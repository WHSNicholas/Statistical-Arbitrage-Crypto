# -------------------------------------------------------------------------------------------------------------------- #
#                                             Statistical Arbitrage Crypto                                             #
#                                                Quantitative Research                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
"""
Title:    Statistical Arbitrage Crypto
Script:   Quantitative Research

Author:   Nicholas Wong
Created:  25th February 2025
Modified: 25th February 2025

Purpose:  This script builds a statistical arbitrage strategy that tries to discover price-volume patterns that predict
          returns. Cryptocurrency markets are still relatively new and should be fertile grounds for finding market
          inefficiencies using statistical arbitrage techniques. The goal of this project is to research profitable
          momentum and reversal strategies in crypto.

Dependencies: pandas, numpy

Instructions:

Data Sources:

Fonts: "CMU Serif.ttf"

Table of Contents:
1. Data Integration
  1.1. Preamble
  1.2. Data Wrangling
2. Strategy Research
  2.1.
3. Implementation

"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Data Integration
# ----------------------------------------------------------------------------------------------------------------------

# 1.1. Preamble ----------------------------------------------------------------------------------------------
# Required Packages
import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from tornado.gen import moment

# Constants
START = dt.datetime(year=2018, month=1, day=1)
END = dt.datetime(year=2024, month=12, day=31)
SEED = 2025
COINGECKO_API_URL = "https://pro-api.coingecko.com/api/v3/"
COINGECKO_API_KEY = "CG-r7KwanmGv8NrZfAP8EfrtbiD"
HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY}
ANNUAL_FACTOR = np.sqrt(365)


# Functions
def get_coins(n=200):
    """
    Fetches the top `n` cryptocurrencies by trading volume from the CoinGecko API.

    Parameters:
    -----------
    n : int, optional (default=100)
        The number of cryptocurrencies to retrieve, sorted by descending trading volume. The maximum `per_page` allowed
         by CoinGecko is typically 250, so for larger values, additional pagination may be required.

    Returns:
    --------
    list of str
        A list of cryptocurrency IDs (as recognized by CoinGecko) corresponding to the top `n` coins by trading volume.

    Raises:
    -------
    requests.exceptions.HTTPError
        If the API request fails (e.g., due to invalid parameters, rate limits, or server errors).

    Notes:
    ------
    - This function queries the `/coins/markets` endpoint of the CoinGecko API.
    - The retrieved IDs can be used for further API queries (e.g., fetching historical market data).
    - Ensure that a valid API key is provided in the `HEADERS` dictionary if required.

    Example Usage:
    --------------
    >>> top_coins = get_coins(100)
    >>> print(top_coins[:5])  # Display the top 5 coin IDs
    ['bitcoin', 'ethereum', 'tether', 'binancecoin', 'usd-coin']
    """
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": n,
        "page": 1,
        "sparkline": "false"
    }

    response = requests.get(url, params=params, headers=HEADERS)
    response.raise_for_status()

    df = response.json()

    return [coin["id"] for coin in df]

def get_data(coin_id, vs_currency='usd', start_date=START, end_date=END, coverage_threshold=0.60):
    """
    Fetches historical market data (price, volume, and market capitalization) for a given cryptocurrency from the
    CoinGecko API's `/market_chart/range` endpoint.

    Parameters
    ----------
    coin_id : str
        The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum').
    vs_currency : str, optional
        The currency in which prices should be quoted (default is 'usd').
    start_date : datetime, optional
        The start date for retrieving market data (default is `START`).
    end_date : datetime, optional
        The end date for retrieving market data (default is `END`).
    coverage_threshold : float, optional
        The minimum fraction of days in the requested date range that must contain at least one data point. If the
        actual coverage is below this threshold, the function returns `None`. Default is 0.60 (60%).

    Returns
    -------
    pandas.DataFrame or None
        - A DataFrame with the following columns:
            - 'Price'      : Market price at the given timestamp.
            - 'Volume'     : 24-hour rolling trade volume.
            - 'Market Cap' : Market capitalization at the given timestamp.
        - The DataFrame is indexed by date.
        - If the data coverage is below `coverage_threshold`, returns `None`.

    Raises
    ------
    requests.exceptions.HTTPError
        If the API request fails due to an invalid response, rate limits, or server errors.
    KeyError
        If the API response does not contain the expected keys (`'prices'`, `'total_volumes'`, or `'market_caps'`).

    Notes
    -----
    - The `/market_chart/range` endpoint returns time-series data for the requested cryptocurrency.
    - CoinGecko provides **rolling 24-hour volume**, meaning that the volume value at each timestamp represents the
      cumulative trading volume over the previous 24 hours.
    - Market capitalization is based on the latest available circulating supply.
    - For large date ranges, the API may return only **one data point per day** instead of sub-daily timestamps.

    Example Usage
    -------------
    Retrieve Bitcoin's price, volume, and market cap for the first week of January 2023:

    >>> import datetime as dt
    >>> df = get_data(
    ...     "bitcoin",
    ...     vs_currency="usd",
    ...     start_date=dt.datetime(2023, 1, 1),
    ...     end_date=dt.datetime(2023, 1, 7),
    ...     coverage_threshold=0.6
    ... )
    >>> df.head()

    Example Output:
          Date     Price       Volume    Market Cap
    2023-01-01  16500.32  12000000000  320000000000
    2023-01-02  16732.55  12500000000  322500000000
    2023-01-03  17012.10  12800000000  330000000000
    """

    # 1. Build the endpoint URL and parameters
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart/range"
    from_ts = int(start_date.timestamp())
    to_ts   = int(end_date.timestamp())
    params = {
        "vs_currency": vs_currency,
        "from": from_ts,
        "to": to_ts
    }

    # 2. Make the request
    response = requests.get(url, params=params, headers=HEADERS)

    if response.status_code == 429:
        time.sleep(1)
        response = requests.get(url, params=params, headers=HEADERS)

    response.raise_for_status()
    data = response.json()

    # 3. Check for essential keys
    if "prices" not in data or "total_volumes" not in data:
        raise KeyError(f"Response missing 'prices' or 'total_volumes': {data}")

    # 4. Create DataFrames for each data array
    df_prices = pd.DataFrame(data["prices"], columns=["Timestamp_ms", "Price"])
    df_volumes = pd.DataFrame(data["total_volumes"], columns=["Timestamp_ms", "Volume"])
    df_mcaps = pd.DataFrame(data.get("market_caps", []), columns=["Timestamp_ms", "Market Cap"])

    # Empty Market Caps
    if df_mcaps.empty:
        df_mcaps = pd.DataFrame(df_prices["Timestamp_ms"], columns=["Timestamp_ms"])
        df_mcaps["Market Cap"] = float("nan")

    # Timestamp
    df_prices["Timestamp"] = pd.to_datetime(df_prices["Timestamp_ms"], unit='ms')
    df_volumes["Timestamp"] = pd.to_datetime(df_volumes["Timestamp_ms"], unit='ms')
    df_mcaps["Timestamp"]   = pd.to_datetime(df_mcaps["Timestamp_ms"], unit='ms')

    df_prices.sort_values("Timestamp", inplace=True)
    df_volumes.sort_values("Timestamp", inplace=True)
    df_mcaps.sort_values("Timestamp", inplace=True)

    df_merged = pd.merge_asof(
        df_prices,
        df_volumes,
        on="Timestamp",
        direction="nearest"
    )
    df_merged = pd.merge_asof(
        df_merged,
        df_mcaps,
        on="Timestamp",
        direction="nearest"
    )

    # 5. Filter by the requested date range
    mask = (df_merged["Timestamp"] >= start_date) & (df_merged["Timestamp"] <= end_date)
    df_merged = df_merged.loc[mask].copy()
    df_merged.sort_values("Timestamp", inplace=True)

    if df_merged.empty:
        return None

    # Coverage check
    total_days = (end_date.date() - start_date.date()).days + 1

    df_merged["Date"] = df_merged["Timestamp"].dt.date
    daily_counts = df_merged.groupby("Date").size()
    coverage_days = len(daily_counts)
    coverage_ratio = coverage_days / total_days

    df_merged.set_index("Date", inplace=True)

    df_merged.drop(columns=["Timestamp_ms_x", "Timestamp_ms_y", 'Timestamp_ms', 'Timestamp', 'Date'], inplace=True,
                   errors="ignore")

    return df_merged if coverage_ratio >= coverage_threshold else None


# 1.2. Data Wrangling ----------------------------------------------------------------------------------------
# Historical Data
coins = get_coins()
data = {}

for coin in coins:
    try:
        df = get_data(coin)
        if df is not None:
            data[coin] = df
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")

coins = data.keys()

# Extracting Cross Sectional Data
price = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(data['bitcoin'].index)
).rename_axis('Date')

returns = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(data['bitcoin'].index)
).rename_axis('Date')

volume = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(data['bitcoin'].index)
).rename_axis('Date')

market_cap = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(data['bitcoin'].index)
).rename_axis('Date')

for coin in coins:
    price[coin] = data[coin]['Price']
    returns[coin] = data[coin]['Price'].pct_change()
    volume[coin] = data[coin]['Volume']
    market_cap[coin] = data[coin]['Market Cap']


# ----------------------------------------------------------------------------------------------------------------------
# 2. Strategy Research
# ----------------------------------------------------------------------------------------------------------------------
# 2.1. Momentum Strategy: Tanh Normalisation -----------------------------------------------------------------
max_sharpe_1 = -np.inf
sharpe_matrix_1 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(1, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        umd_strat_1 = np.tanh(returns.shift(i).rolling(window=j, min_periods=1).mean())
        umd_strat_1 = umd_strat_1.sub(umd_strat_1.mean(axis=1), axis=0)
        umd_strat_1 = umd_strat_1.div(umd_strat_1.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_1 = (returns * umd_strat_1).sum(axis=1)

        sharpe_1 = portfolio_returns_1.mean() / portfolio_returns_1.std() * ANNUAL_FACTOR
        sharpe_matrix_1[i - 1, j - 1] = sharpe_1

        if sharpe_1 > max_sharpe_1:
            max_sharpe_1 = sharpe_1
            best_params_1 = [i, j]

# Best Strategy
umd_strat_1 = np.tanh(returns.shift(best_params_1[0]).rolling(window=best_params_1[1], min_periods=1).mean())
umd_strat_1 = umd_strat_1.sub(umd_strat_1.mean(axis=1), axis=0)
umd_strat_1 = umd_strat_1.div(umd_strat_1.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_1 = (returns * umd_strat_1).sum(axis=1)
cumulative_returns_1 = (1 + portfolio_returns_1).cumprod()
sharpe_1 = portfolio_returns_1.mean() / portfolio_returns_1.std() * ANNUAL_FACTOR

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_1, cmap='coolwarm')
plt.title("Sharpe Ratio by Momentum Window (j) and Lookback Exclusion (i)")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()

# 2.2. Momentum Strategy 2: Volatility Adjusted --------------------------------------------------------------
max_sharpe_2 = -np.inf
sharpe_matrix_2 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        momentum_signal = returns.shift(i).rolling(window=j, min_periods=2).mean()
        volatility = returns.shift(i).rolling(window=j, min_periods=2).std()

        umd_strat_2 = momentum_signal.div(volatility, axis=0)

        umd_strat_2 = umd_strat_2.sub(umd_strat_2.mean(axis=1), axis=0)
        umd_strat_2 = umd_strat_2.div(umd_strat_2.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_2 = (returns * umd_strat_2).sum(axis=1)

        sharpe_2 = portfolio_returns_2.mean() / portfolio_returns_2.std() * ANNUAL_FACTOR
        sharpe_matrix_2[i - 1, j - 1] = sharpe_2

        if sharpe_2 > max_sharpe_2:
            max_sharpe_2 = sharpe_2
            best_params_2 = [i, j]

# Best Strategy
momentum_signal_2 = returns.shift(best_params_2[0]).rolling(window=best_params_2[1], min_periods=2).mean()
volatility = returns.shift(best_params_2[0]).rolling(window=best_params_2[1], min_periods=2).std()

umd_strat_2 = momentum_signal_2.div(volatility, axis=0)

umd_strat_2 = umd_strat_2.sub(umd_strat_2.mean(axis=1), axis=0)
umd_strat_2 = umd_strat_2.div(umd_strat_2.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_2 = (returns * umd_strat_2).sum(axis=1)
cumulative_returns_2 = (1 + portfolio_returns_2).cumprod()
sharpe_2 = portfolio_returns_2.mean() / portfolio_returns_2.std() * ANNUAL_FACTOR

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_2, cmap='coolwarm')
plt.title("Sharpe Ratio by Momentum Window (j) and Lookback Exclusion (i) Volatility Adjusted")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()



# 2.3. Momentum Strategy 3: Rank Based --------------------------------------------------------------
max_sharpe_3 = -np.inf
sharpe_matrix_3 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        momentum_signal = returns.shift(i).rolling(window=j, min_periods=2).mean().rank(1)

        umd_strat_3 = momentum_signal.sub(momentum_signal.mean(axis=1), axis=0)
        umd_strat_3 = umd_strat_3.div(umd_strat_3.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_3 = (returns * umd_strat_3).sum(axis=1)

        sharpe_3 = portfolio_returns_3.mean() / portfolio_returns_3.std() * ANNUAL_FACTOR
        sharpe_matrix_3[i - 1, j - 1] = sharpe_3

        if sharpe_3 > max_sharpe_3:
            max_sharpe_3 = sharpe_3
            best_params_3 = [i, j]

# Best Strategy
momentum_signal_3 = returns.shift(best_params_3[0]).rolling(window=best_params_3[1], min_periods=2).mean().rank(1)

umd_strat_3 = momentum_signal_3.sub(momentum_signal_3.mean(axis=1), axis=0)
umd_strat_3 = umd_strat_3.div(umd_strat_3.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_3 = (returns * umd_strat_3).sum(axis=1)
cumulative_returns_3 = (1 + portfolio_returns_3).cumprod()
sharpe_3 = portfolio_returns_3.mean() / portfolio_returns_3.std() * ANNUAL_FACTOR

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_3, cmap='coolwarm')
plt.title("Sharpe Ratio by Momentum Window (j) and Lookback Exclusion (i) Rank Based")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()

# 2.3. Momentum Strategy 3: Rank Based --------------------------------------------------------------
max_sharpe_4 = -np.inf
sharpe_matrix_4 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        momentum_signal_4 = returns.shift(i).ewm(span=j, adjust=False).mean()
        volatility = returns.shift(i).ewm(span=j, adjust=False).std()

        volume_mean = volume.shift(i).ewm(span=j, adjust=False).mean()
        volume_std  = volume.shift(i).ewm(span=j, adjust=False).std()
        volume_zscore = volume.sub(volume_mean, 0).div(volume_std)

        umd_strat_4 = (momentum_signal_4.mul(volume_zscore, 0).div(volatility, axis=0)).rank(1)

        umd_strat_4 = umd_strat_4.sub(umd_strat_4.mean(axis=1), axis=0)
        umd_strat_4 = umd_strat_4.div(umd_strat_4.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_4 = (returns * umd_strat_4).sum(axis=1)
        cumulative_returns_4 = (1 + portfolio_returns_4).cumprod()
        sharpe_4 = portfolio_returns_4.mean() / portfolio_returns_4.std() * ANNUAL_FACTOR
        sharpe_matrix_4[i - 1, j - 1] = sharpe_4

        if sharpe_4 > max_sharpe_4:
            max_sharpe_4 = sharpe_4
            best_params_4 = [i, j]

# Best Strategy
momentum_signal_4 = returns.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).mean()
volatility = returns.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).std()

volume_mean = volume.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).mean()
volume_std = volume.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).std()
volume_zscore = volume.sub(volume_mean, 0).div(volume_std)

umd_strat_4 = (momentum_signal_4.mul(volume_zscore, 0).div(volatility, axis=0)).rank(1)

umd_strat_4 = umd_strat_4.sub(umd_strat_4.mean(axis=1), axis=0)
umd_strat_4 = umd_strat_4.div(umd_strat_4.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_4 = (returns * umd_strat_4).sum(axis=1)
cumulative_returns_4 = (1 + portfolio_returns_4).cumprod()
sharpe_4 = portfolio_returns_4.mean() / portfolio_returns_4.std() * ANNUAL_FACTOR

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_4, cmap='coolwarm')
plt.title("Sharpe Ratio by Momentum Window (j) and Lookback Exclusion (i) Rank Based")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()



# ----------------------------------------------------------------------------------------------------------------------
# 3. Implementation
# ----------------------------------------------------------------------------------------------------------------------

