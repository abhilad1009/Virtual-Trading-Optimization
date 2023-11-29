import datetime
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp  # type: ignore[import-untyped]
import tqdm
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import argparse
import os

import warnings

warnings.simplefilter('ignore')

FloatArray = npt.NDArray[np.float_]

CHART_DIR = "charts/"
try:
    os.mkdir(CHART_DIR)
except:
    pass

RISK_TOLERANCE = 500
DAILY_VOLUME_LIMIT = 100


####################################################################
#                       Data Functions                             #
####################################################################

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load historical price data

    :param path: path to a CSV of the price data
    :return: data frame of historical prices [operating_day, hour_beginning, node, da_price, rt_price]
    """
    return pd.read_csv(path, parse_dates=["operating_day"])



def compute_expected_spread(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the expected spread for each node and hour for the operating day, by taking the mean of observed
    spread values available as of 24 hours prior to the beginning of the operating day.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: data frame of expected spread values [operating_day, hour_beginning, node, expected_dart_spread]
    """

    # DART spread is DA_price - RT_price
    price_df["node_dart"] = price_df['da_price'] -  price_df['rt_price']
    df_concat_list = []

    for node in price_df.node.unique():
        for hour in range(0,24):
            node_df = price_df[(price_df['node'] == node) & (price_df['hour_beginning'] == hour) ].copy()
            expected_dart_spread = [0 for i in range(len(node_df))]
            dart_spread = node_df['node_dart'].values.tolist()
            running_sum = 0
            counter = 0
            for i in range(0,len(dart_spread)):
                if i-2>=0:
                    expected_dart_spread[i] = np.mean(dart_spread[:i-1])
            node_df['expected_dart_spread']  = expected_dart_spread
            df_concat_list.append(node_df.copy())
    
    merged_df = pd.concat(df_concat_list)
    merged_df.sort_index(inplace=True)

    # print(merged_df)

    return merged_df[['operating_day', 'hour_beginning', 'node', 'expected_dart_spread']]



def compute_spread_variance(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the estimated spread variance for each node and hour for the operating day, by taking the sample variance
    of observed spread values available as of 24 hours prior to the beginning of the operating day.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: data frame of estimated spread variance values [operating_day, hour_beginning, node, dart_spread_var]
    """
    # DART spread is DA_price - RT_price
    price_df["node_dart"] = price_df['da_price'] - price_df['rt_price']
    df_concat_list = []

    for node in price_df.node.unique():
        for hour in range(0,24):
            node_df = price_df[(price_df['node'] == node) & (price_df['hour_beginning'] == hour) ].copy()
            dart_spread_var = [0 for i in range(len(node_df))]
            dart_spread = node_df['node_dart'].values.tolist()
            for i in range(0,len(dart_spread)):
                if i-2>=0:
                    dart_spread_var[i] = np.var(dart_spread[:i-1])
            node_df['dart_spread_var']  = dart_spread_var
            df_concat_list.append(node_df.copy())
    
    merged_df = pd.concat(df_concat_list)
    merged_df.sort_index(inplace=True)

    # print(merged_df)

    return merged_df[['operating_day', 'hour_beginning', 'node', 'dart_spread_var']]



def get_daily_expected_spread_vectors(expected_spread_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the expected spread data frame into a data frame with one row per operating day, where the index is
    the operating day and the elements of each row are the expected spread values for all node and hour combinations
    on that day.

    :param expected_spread_df: data frame of expected spread values
        [operating_day, hour_beginning, node, expected_dart_spread]
    :return: data frame of expected spread vectors with operating day as index
    """
    # Pivot dataframe to convert rows to column values for each operating day
    spread_vector_df = pd.pivot_table(expected_spread_df,index=['operating_day'],columns=expected_spread_df.groupby(['operating_day']).cumcount().add(1),values=['expected_dart_spread'],aggfunc='sum')
    
    # Renaming columns based on node and hour combination, it is hardcoded to 8 nodes and 24 hours
    cols = [r['node']+"_"+str(r['hour_beginning']) for idx,r in expected_spread_df[:8*24].iterrows()]
    spread_vector_df.columns = cols
    # print(spread_vector_df) 
    return spread_vector_df



def get_daily_spread_variance_vectors(spread_var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the spread variance data frame into a data frame with one row per operating day, where the index is
    the operating day and the elements of each row are the estimated spread variance values for all node and hour
    combinations on that day (i.e. the diagonal entries of the covariance matrix).

    :param spread_var_df: data frame of expected spread values
        [operating_day, hour_beginning, node, dart_spread_var]
    :return: data frame of expected spread vectors with operating day as index
    """
    spread_var_vector_df = pd.pivot_table(spread_var_df,index=['operating_day'],columns=spread_var_df.groupby(['operating_day']).cumcount().add(1),values=['dart_spread_var'],aggfunc='sum')
    
    # Renaming columns based on node and hour combination, it is hardcoded to 8 nodes and 24 hours
    cols = [r['node']+"_"+str(r['hour_beginning']) for idx,r in spread_var_df[:8*24].iterrows()]
    spread_var_vector_df.columns = cols
    # print(spread_var_vector_df) 
    return spread_var_vector_df



def get_bids_from_daily_portfolios(portfolio_df: pd.DataFrame, non_negative_bids: bool) -> pd.DataFrame:
    """
    Transform a data frame of daily portfolios to a data frame of bids. Also removes any bids smaller than 0.1 MW.

    :param portfolio_df: data frame of daily bid quantities with operating day as the index
    :param non_negative_bids: If True, we will only keep values >0.1 MW, else we keep values whose absolute value > 0.1 MW, Default False
    :return: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    """
    # get list of nodes and operating hours
    nodes, opening_hours = zip(*(s.split("_") for s in portfolio_df.columns))
    bids = portfolio_df.to_numpy().flatten().tolist()

    # number of times nodes and opening hours need to be repeated to match bids on all dates
    node_repeatition = int(len(bids)/len(nodes))

    bid_df = pd.DataFrame({
        "operating_day":np.repeat(portfolio_df.index.date,8*24).tolist(),
        "hour_beginning":opening_hours*node_repeatition,
        "node":nodes*node_repeatition,
        "bid_mw":bids
    })

    # set bid_type, DEC if bid_mw >= 0 -> da-rt >= 0 ,  INC otherwise
    bid_df['bid_type'] = np.where(bid_df['bid_mw']>=0,"DEC","INC")
    if non_negative_bids:
        # Keep only bids > 0.1 MW
        bid_df['bid_mw'] = np.where((bid_df['bid_mw'] < 0.1),0,bid_df['bid_mw'])
    else:
        # Keep bids where abs(bid) > 0.1 MW, default branch
        bid_df['bid_mw'] = np.where((bid_df['bid_mw'] < 0.1) & (bid_df['bid_mw']>-0.1),0,bid_df['bid_mw'])

    bid_df['operating_day'] = bid_df['operating_day'].astype('datetime64')
    bid_df['hour_beginning'] = bid_df['hour_beginning'].astype('int')

    # print(bid_df)

    return bid_df



####################################################################
#                       Metrics                                    #
####################################################################

def compute_total_pnl(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """
    Compute the total PnL over all operating days in the bid data frame

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: the total PnL
    """
    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']

    return merged_df['pnl'].sum()



def win_loss_rate(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """
    Compute the win/loss rate over all bids in the bid data frame

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: the no_wins/no_losses
    """
    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']

    wins = len(merged_df[merged_df['pnl']>=0])
    losses = len(merged_df)-wins

    return wins/losses



def profit_factor(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """
    Compute the profit factor over all bids in the bid data frame

    We can sum over entire day, and then calculate profit factor over all days as well

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: the gross_profit/gross_loss
    """
    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']\

    gross_profit = merged_df[merged_df['pnl']>=0]['pnl'].sum()
    gross_loss = abs(merged_df[merged_df['pnl']<0]['pnl'].sum())

    return gross_profit/gross_loss



def sharpe_ratio(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """
    Compute the sharpe ratio for the strategy

    This function is wrong on multiple levels, but still useful for quantitative comparison

    This is a very crude approach to sharpe ratio since we are dealing with multiple nodes on multiple hours of the day
    We simply assume PnL as returns
    Risk_free rate is assumed negligible, though we can assume Treasury bond rates, getting return rates on our bids dataframe is tricky
    And thus, we calculate sharpe as : mean(returns)/std(returns)

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: the sharpe ratio
    """
    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']

    daily_returns = merged_df.groupby("operating_day").sum().copy()

    sr = daily_returns['pnl'].mean()/daily_returns['pnl'].std()

    return sr



####################################################################
#                       Plotting                                   #
####################################################################

def plot_drawdown(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> None:
    """
    Plot drawdown of the daily returns

    Since we are considering returns as PnL with no reference for % change, this will also be a bit incorrect

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: None, saves the plot
    """

    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']

    # Sum over node-hour combination to generate daily PnL as return
    daily_returns = merged_df.groupby("operating_day").sum().copy()
    # Crude estimate of % returns
    daily_returns["return_rate"] =  daily_returns['pnl'].pct_change().fillna(0)

    cumulative = (daily_returns["return_rate"] + 1).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    drawdown.plot()
    # plt.show()
    plt.savefig(CHART_DIR +"drawdown.jpg")
    plt.close()



def plot_cummulative_returns(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> None:
    """
    Plot cummulative returns of the portfolio

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: None, saves cummulative returns plot
    """

    merged_df = pd.merge(price_df,bid_df,on=['operating_day','hour_beginning','node'])
    merged_df['pnl'] = merged_df['node_dart']*merged_df['bid_mw']

    # Sum over node-hour combination to generate daily PnL as return
    daily_returns = merged_df.groupby("operating_day").sum().copy().fillna(0)
    t = 0
    cummulative_pnl= []
    # Generate cummulative returns at each day
    for pnl in daily_returns['pnl'].values:
        t+=pnl
        cummulative_pnl.append(t)
    plt.plot(daily_returns.index.values,cummulative_pnl)
    # plt.show()
    plt.savefig(CHART_DIR +"cummulative_returns.jpg")
    plt.close()



def plot_daily_dart_spread(price_df: pd.DataFrame) -> None:
    """
    Plot DART spread, DA - RT prices for each node
    Each node dart spread is saved as a separate figure

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: None, saves dart spread figure
    """
    for node in price_df.node.unique():
        for hour in range(0,24):
            node_df = price_df[(price_df['node'] == node) & (price_df['hour_beginning'] == hour)]
            
            plt.plot(node_df["operating_day"],node_df["node_dart"])
            plt.savefig(CHART_DIR +node+ str(hour)+"_dart_spread.jpg")
            # plt.show()
            plt.close()



def plot_expected_dart_spread(price_df: pd.DataFrame) -> None:
    """
    Plot expected DART spread, rolling_mean(DA - RT) prices for each node
    Each node expected dart spread is saved as a separate figure

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: None, saves expected dart spread figure
    """
    expected_spread = compute_expected_spread(price_df,False)
    for node in expected_spread.node.unique():  
        for hour in range(0,24):
            node_df = expected_spread[(expected_spread['node'] == node) & (expected_spread['hour_beginning'] == hour)]

            plt.plot(node_df["operating_day"],node_df["expected_dart_spread"])
            plt.savefig(CHART_DIR +node+str(hour)+"_expected_dart.jpg")
            # plt.show()
            plt.close()



####################################################################
#                       Objective Functions                        #
####################################################################

def portfolio_objective_fn(
    bid_mw: FloatArray,
    expected_spread: FloatArray,
    spread_variance: FloatArray,
    risk_tol: int
) -> float:
    """
    The objective function to minimize in the portfolio optimizer. This should also use the RISK_TOLERANCE constant
    defined above.

    :param bid_mw: array containing the bid quantities (in MW) for the daily portfolio
    :param expected_spread: array containing the expected spread values for the day
    :param spread_variance: array containing the estimated spread variance values for the day (i.e. the diagonal
        entries of the covariance matrix)
    :return: objective function value to minimize
    """
    covariance_matrix = np.diag(spread_variance)

    # You can check shape of matrices
    # print(bid_mw.shape, covariance_matrix.shape, expected_spread.shape)

    portfolio_objective_fn = np.dot(bid_mw,np.dot(covariance_matrix,bid_mw))  - risk_tol * np.dot(expected_spread ,bid_mw)
    
    # print(portfolio_objective_fn)
    return portfolio_objective_fn



def mw_constraint_fn(bid_mw: FloatArray, max_total_mw: float) -> float:
    """
    The constraint function which must take a non-negative value if and only if the constraint is satisfied.

    :param bid_mw: array containing the bid quantities (in MW) for the daily portfolio
    :param max_total_mw: the maximum number of total MW that can be traded in a day
    :return: constraint function value which must be non-negative iff the constraint is satisfied
    """
    # Here we take the absolute value of bids since we can either sell or buy, 
    # If we don't take absolute value, then we will trade volumes beyond maximum daily MW allowed

    # Approach 1: return 𝑉 - Σ |𝑤𝑖| as inequality constraint value
    return max_total_mw - np.sum(np.abs(bid_mw))
    # return max_total_mw - np.sum(np.sqrt(np.square(bid_mw)))




####################################################################
#                       Optimization Routine                       #
####################################################################


def generate_daily_bids(
    price_df: pd.DataFrame,
    risk_tol: int,
    non_negative_bids: bool,
    daily_volume_limit: int,
    maxiter: int,
    raise_failure: bool,
    first_operating_day: t.Union[str, datetime.date],
    last_operating_day: t.Union[str, datetime.date],
) -> pd.DataFrame:
    """
    Generate bids for the date range, computing the expected DART spreads and estimated variances from
    the price data and limiting each daily portfolio to a maximum size in MW.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param risk_tol: The risk tolerance value
    :param non_negative_bids: If True, we will only keep values >0.1 MW, else we keep values whose absolute value > 0.1 MW, Default False
    :param daily_volume_limit: Maximum daily volume limit on trading
    :param maxiter: Maximum iterations for optimization routine
    :param raise_failure: print if optimization failed to converge
    :param first_operating_day: first operating day for which to generate bids
    :param last_operating_day: last operating day for which to generate bids
    :return: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    """
    expected_spread = compute_expected_spread(price_df)
    spread_variance = compute_spread_variance(price_df)

    daily_expected_spread = get_daily_expected_spread_vectors(expected_spread)
    daily_spread_variance = get_daily_spread_variance_vectors(spread_variance)

    portfolios = []

    options = None
    if maxiter:
        options = {'maxiter':maxiter}
         
    # Convert the optimization routine to function for submitting jobs in parallel
    def minimize_portfolio(day):
        result = sp.optimize.minimize(
            portfolio_objective_fn,
            np.zeros(len(daily_expected_spread.columns)),
            args=(daily_expected_spread.loc[day].values, daily_spread_variance.loc[day].values,risk_tol),
            constraints={
                "type": "ineq",
                "fun": mw_constraint_fn,
                "args": [daily_volume_limit],
            },
            options = options
        )
        if raise_failure:
            if result.success == False:
                print(result.message)

        return pd.DataFrame(
                result.x[None, :],
                columns=daily_expected_spread.columns,
                index=pd.Index([day], name="operating_day")
            )
    dates = pd.date_range(first_operating_day, last_operating_day)

    # Parallelize the porfolio optimization, since calculation in the loop are independant from previous calculations
    portfolios = Parallel(n_jobs=-1)(delayed(minimize_portfolio)(day) for day in tqdm.tqdm(dates))

    return get_bids_from_daily_portfolios(pd.concat(portfolios),non_negative_bids)



def risk_tuned_generate_daily_bids(
    price_df: pd.DataFrame,
    risk_tol_range:  t.Tuple,
    risk_tune_size: float,
    non_negative_bids: bool,
    daily_volume_limit: int,
    maxiter: int,
    raise_failure: bool,
    first_operating_day: t.Union[str, datetime.date],
    last_operating_day: t.Union[str, datetime.date],
) -> pd.DataFrame:
    """
    Generate bids for the date range, computing the expected DART spreads and estimated variances from
    the price data and limiting each daily portfolio to a maximum size in MW.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param risk_tol_range: The risk tolerance range(start,stop,step) to interate over for tuning
    :param risk_tune_size: The fraction of data to use for tuning risk tolerance
    :param non_negative_bids: If True, we will only keep values >0.1 MW, else we keep values whose absolute value > 0.1 MW, Default False
    :param daily_volume_limit: Maximum daily volume limit on trading
    :param maxiter: Maximum iterations for optimization routine
    :param raise_failure: print if optimization failed to converge
    :param first_operating_day: first operating day for which to generate bids
    :param last_operating_day: last operating day for which to generate bids
    :return: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    """

    best_risk = None
    best_pnl = -float('inf')

    dates = pd.date_range(first_operating_day, last_operating_day)
    last_operating_day = dates[int(len(dates)*risk_tune_size)]

    # Iterate over range of risk values and select risk tolerance with maximum PnL
    for risk in range(risk_tol_range[0],risk_tol_range[1],risk_tol_range[2]):
        bid_df = generate_daily_bids(price_df, risk, non_negative_bids, daily_volume_limit, maxiter, raise_failure,first_operating_day,last_operating_day)
        pnl = compute_total_pnl(price_df, bid_df)
        if pnl>best_pnl:
            best_pnl=pnl
            best_risk=risk
    
    print("Best risk tolerance:",best_risk)

    # Use the best risk tolerance value to generate final daily bids
    return generate_daily_bids(price_df, best_risk,non_negative_bids, daily_volume_limit, maxiter, raise_failure,first_operating_day,last_operating_day)



####################################################################
#                       Main Function                              #
####################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_risk_tol",type=bool, help="tune risk tolerance", default=False)
    parser.add_argument("--risk_tol", help="risk tolerance, pass int or range start end step)",nargs='+', type=int,default=[RISK_TOLERANCE])
    parser.add_argument("--risk_tune_size",type=float, help="fraction of data to use for tuning",default=0.2)
    parser.add_argument("--non_negative_bids",type=bool, help="allow only non negative bids",default=False)
    parser.add_argument("--daily_volume_limit",type=int, help="maximum daily trade volume",default=DAILY_VOLUME_LIMIT)
    parser.add_argument("--maxiter",type=int, help="max iterations for optimization",default=None)
    parser.add_argument("--raise_failure",type=bool, help="print if optimzation fails to converge",default=False)
    # Parse arguments
    args = parser.parse_args()

    non_negative_bids = args.non_negative_bids
    daily_volume_limit = args.daily_volume_limit
    maxiter = args.maxiter
    raise_failure = args.raise_failure

    # Load the dataframe
    price_df = load_price_data("prices.csv")

    print(f"Generating bids for 2022 with daily limit of {daily_volume_limit} MW...")


    if args.tune_risk_tol:
        # Tune the risk tolerance for maximum PnL using given range and fraction of data
        risk_tol_range = args.risk_tol
        risk_tune_size = args.risk_tune_size

        print("With Risk Tolerance tuning settings")
        print("Risk Tolerance range",risk_tol_range)
        print("Tuning size:",risk_tune_size)

        bid_df = risk_tuned_generate_daily_bids(price_df, risk_tol_range, risk_tune_size, non_negative_bids, daily_volume_limit, maxiter, raise_failure,"2022-01-01", "2022-12-31")
    else:
        # Used fixed risk tolerance value
        risk_tol = args.risk_tol[0]
        print("With Risk Tolerance:",risk_tol)

        bid_df = generate_daily_bids(price_df, risk_tol, non_negative_bids, daily_volume_limit, maxiter, raise_failure,"2022-01-01", "2022-12-31")


    # Save the generated bids
    bid_df.to_csv("bids.csv", index=False)


    # Compute the metrics
    pnl = compute_total_pnl(price_df, bid_df)
    wlr = win_loss_rate(price_df, bid_df)
    pf = profit_factor(price_df, bid_df)
    sr = sharpe_ratio(price_df, bid_df)

    # Print the metrics
    print(f"The strategy made ${pnl:.2f}")
    print(f"The strategy has win/loss rate of {wlr:.2f}")
    print(f"The strategy has profit factor of {pf:.2f}")
    print(f"The strategy has sharpe ratio of {sr:.2f}")

    # Plot and save the figures
    plot_drawdown(price_df, bid_df)
    plot_cummulative_returns(price_df, bid_df)
    plot_expected_dart_spread(price_df)
    plot_daily_dart_spread(price_df)
