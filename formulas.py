import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import STL


def plot_series(time, series, format="-", start=0, end=None, label=None):
    """[Plot the series data over a time range]

    Args:
        time (data range): [The entire time span of the data in range format]
        series ([integers]): [Series value corresponding to its point on the time axis]
        format (str, optional): [Graph type]. Defaults to "-".
        start (int, optional): [Time to start time series data]. Defaults to 0.
        end ([type], optional): [Where to stop time data]. Defaults to None.
        label ([str], optional): [Label name of series]. Defaults to None.
    """
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast
     This implementation is *much* faster than the previous one"""
  mov = np.cumsum(series)
  mov[window_size:] = mov[window_size:] - mov[:-window_size]
  return mov[window_size - 1:-1] / window_size
    



def calculate_returns(close):
    """
    Compute returns for each ticker and date in close.
    
    Parameters
    ----------
    close : DataFrame
        Close prices for each ticker and date
    
    Returns
    -------
    returns : DataFrame
        Returns for each ticker and date
    """
    # TODO: Implement Function
    
    return (close - close.shift(1))/close.shift(1)


def resample_prices(close_prices, freq='M'):
    """
    Resample close prices for each ticker at specified frequency.
    
    Parameters
    ----------
    close_prices : DataFrame
        Close prices for each ticker and date
    freq : str
        What frequency to sample at
        For valid freq choices, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    Returns
    -------
    prices_resampled : DataFrame
        Resampled prices for each ticker and date
    """
    
    return close_prices.resample(freq).last()

def compute_log_returns(prices):
    """
    Compute log returns for each ticker.
    
    Parameters
    ----------
    prices : DataFrame
        Prices for each ticker and date
    
    Returns
    -------
    log_returns : DataFrame
        Log returns for each ticker and date
    """
    r_t = np.log(prices) - np.log(prices.shift(1))
    return r_t


def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    """
    Compute expected returns for the portfolio, assuming equal investment in each long/short stock.
    
    Parameters
    ----------
    df_long : DataFrame
        Top stocks for each ticker and date marked with a 1
    df_short : DataFrame
        Bottom stocks for each ticker and date marked with a 1
    lookahead_returns : DataFrame
        Lookahead returns for each ticker and date
    n_stocks: int
        The number number of stocks chosen for each month
    
    Returns
    -------
    portfolio_returns : DataFrame
        Expected portfolio returns for each ticker and date
    """
    
    
    return (lookahead_returns*(df_long - df_short)) / n_stocks


def get_top_n(prev_returns, top_n):
    """
    Select the top performing stocks
    
    Parameters
    ----------
    prev_returns : DataFrame
        Previous shifted returns for each ticker and date
    top_n : int
        The number of top performing stocks to get
    
    Returns
    -------
    top_stocks : DataFrame
        Top stocks for each ticker and date marked with a 1
    """
    # TODO: Implement Function
    top_stocks = prev_returns.apply(lambda x: x.nlargest(top_n), axis=1)
    top_stocks = top_stocks.applymap(lambda x: 0 if pd.isna(x) else 1)
    top_stocks = top_stocks.astype(int)
        
    
    return top_stocks

def analyze_alpha(expected_portfolio_returns_by_date):
    """
    Perform a t-test with the null hypothesis being that the expected mean return is zero.
    
    Parameters
    ----------
    expected_portfolio_returns_by_date : Pandas Series
        Expected portfolio returns for each date
    
    Returns
    -------
    t_value
        T-statistic from t-test
    p_value
        Corresponding p-value
    """

    t_statistic,p_value = stats.ttest_1samp(expected_portfolio_returns_by_date, 0)
    return t_statistic,p_value/2

def seasonal_trend_decomp_plot(dataframe,target_series, freq, seasonal_smoother, period):
    """[summary]

    Args:
        dataframe ([pandas dataframe]): [dataframe holding all of your data]
        target_series ([series]): [Name of column in data frame you want to build plot from like 'Adj Close']
        freq ([int]): [How do you want to resample data - 'D', 'W','M']
        seasonal_smoother ([type]): [Length of smoother in whatever units as defined by freq]
        period ([type]): [Periodicity of the sequence (for monthly = 12/year)]

    Returns:
        [STL plot]: [Seasonal-Trend decomposition using LOESS (STL)]
    """

    df = dataframe.set_index('Date')
    df = df.resample(freq).last()
    target = df[target_series]
    stl = STL(target, seasonal = seasonal_smoother)
    res = stl.fit()
    res.plot()
    return 

def resample_series(dataframe, target,freq):
    df = dataframe.set_index('Date')
    df = df.resample(freq).last()
    target = df[target]
    return target