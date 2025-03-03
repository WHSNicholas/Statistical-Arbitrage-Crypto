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

Dependencies: pandas, numpy, requests, time, datetime, matplotlib, seaborn, pypfopt

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
from pypfopt import EfficientFrontier


# Constants
START = dt.datetime(year=2019, month=1, day=1)
END = dt.datetime(year=2024, month=12, day=31)
SEED = 2025
COINGECKO_API_URL = "https://pro-api.coingecko.com/api/v3/"
COINGECKO_API_KEY = ""
HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY}
ANNUAL_FACTOR = np.sqrt(365)
MKT_COST = 0.0007


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


def get_daily_data(coin_id, vs_currency='usd', start_date=START, end_date=END, coverage_threshold=0.60):
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

    # 6. Coverage check
    total_days = (end_date.date() - start_date.date()).days + 1

    df_merged["Date"] = df_merged["Timestamp"].dt.date
    daily_counts = df_merged.groupby("Date").size()
    coverage_days = len(daily_counts)
    coverage_ratio = coverage_days / total_days

    df_merged.set_index("Date", inplace=True)

    df_merged.drop(columns=["Timestamp_ms_x", "Timestamp_ms_y", 'Timestamp_ms', 'Timestamp', 'Date'], inplace=True,
                   errors="ignore")

    return df_merged if coverage_ratio >= coverage_threshold else None


def get_hourly_data(coin_id, vs_currency='usd', start_date=START, end_date=END, coverage_threshold=0.60):
    """
    Fetches historical hourly market data (price, volume, and market capitalization) for a given cryptocurrency from the
    CoinGecko API's `/market_chart/range` endpoint. The function breaks the overall date range into 90-day chunks to
    ensure hourly data is returned (since ranges >90 days return daily data).

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
        If the API response does not contain the expected keys ('prices', 'total_volumes', or 'market_caps').
    """

    # 1. Build the endpoint URL and parameters
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart/range"

    # Partitioning the range to 90day chunks
    delta = dt.timedelta(days=90)
    current_start = start_date

    all_prices = []
    all_volumes = []
    all_mcaps = []

    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        from_ts = int(current_start.timestamp())
        to_ts = int(current_end.timestamp())
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

        all_prices.extend(data["prices"])
        all_volumes.extend(data["total_volumes"])

        if "market_caps" in data and data["market_caps"]:
            all_mcaps.extend(data["market_caps"])
        else:
            pass

        # Move to the next chunk
        current_start = current_end
        time.sleep(1)

    # 4. Create DataFrames for each data array
    df_prices = pd.DataFrame(all_prices, columns=["Timestamp_ms", "Price"])
    df_volumes = pd.DataFrame(all_volumes, columns=["Timestamp_ms", "Volume"])
    if all_mcaps:
        df_mcaps = pd.DataFrame(all_mcaps, columns=["Timestamp_ms", "Market Cap"])
    else:
        df_mcaps = pd.DataFrame(df_prices["Timestamp_ms"], columns=["Timestamp_ms"])
        df_mcaps["Market Cap"] = float("nan")

    # Timestamps
    df_prices["Timestamp"] = pd.to_datetime(df_prices["Timestamp_ms"], unit='ms')
    df_volumes["Timestamp"] = pd.to_datetime(df_volumes["Timestamp_ms"], unit='ms')
    df_mcaps["Timestamp"] = pd.to_datetime(df_mcaps["Timestamp_ms"], unit='ms')

    df_prices.sort_values("Timestamp", inplace=True)
    df_volumes.sort_values("Timestamp", inplace=True)
    df_mcaps.sort_values("Timestamp", inplace=True)

    # Merge data using asof merge on Timestamp
    df_merged = pd.merge_asof(
        df_prices, df_volumes, on="Timestamp", direction="nearest"
    )
    df_merged = pd.merge_asof(
        df_merged, df_mcaps, on="Timestamp", direction="nearest"
    )

    # 5. Filter by the requested date range
    mask = (df_merged["Timestamp"] >= start_date) & (df_merged["Timestamp"] <= end_date)
    df_merged = df_merged.loc[mask].copy()
    df_merged.sort_values("Timestamp", inplace=True)

    if df_merged.empty:
        return None

    # 6. Coverage check
    total_days = (end_date.date() - start_date.date()).days + 1

    df_merged["Date"] = df_merged["Timestamp"].dt.date
    df_merged['Timestamp'] = df_merged['Timestamp'].dt.round('h')

    daily_counts = df_merged.groupby("Date").size()
    coverage_days = len(daily_counts)
    coverage_ratio = coverage_days / total_days

    df_merged.set_index("Timestamp", inplace=True)
    df_merged.drop(columns=["Timestamp_ms_x", "Timestamp_ms_y", "Timestamp_ms", "Timestamp", "Date"],
                   inplace=True, errors="ignore")

    return df_merged if coverage_ratio >= coverage_threshold else None


# 1.2. Data Wrangling ----------------------------------------------------------------------------------------
# Historical Data
coins = get_coins()
daily_data = {}
hourly_data = {}

# for coin in coins:
#     try:
#         df = get_daily_data(coin)
#         if df is not None:
#             daily_data[coin] = df
#             print(f"Including {coin}")
#         else:
#             print(f"Excluding {coin}")
#     except Exception as e:
#         print(f"Error fetching data for {coin}: {e}")

coins = daily_data.keys()

# for coin in coins:
#     try:
#         df = get_hourly_data(coin)
#         if df is not None:
#             hourly_data[coin] = df
#             print(f"Including {coin}")
#         else:
#             print(f"Excluding {coin}")
#     except Exception as e:
#         print(f"Error fetching data for {coin}: {e}")
#     time.sleep(1)

# Saving Data
# for coin in coins:
#     hourly_data[coin].to_csv(f"Hourly Data/{coin}_hourly_data.csv", index=True)
#     daily_data[coin].to_csv(f"Daily Data/{coin}_daily_data.csv", index=True)

# Getting Data from Saved CSV Files
coins = ['bitcoin', 'tether', 'ethereum', 'usd-coin', 'ripple', 'solana', 'dogecoin', 'litecoin', 'binancecoin',
         'thorchain', 'cardano', 'tron', 'chainlink', 'avalanche-2', 'wrapped-bitcoin', 'polkadot', 'hedera-hashgraph',
         'maker', 'wbnb', 'weth', 'bitcoin-cash', 'shiba-inu', 'uniswap', 'aave', 'near', 'stellar', 'render-token',
         'filecoin', 'nervos-network', 'mantra-dao', 'ethereum-classic', 'golem', 'fetch-ai', 'cosmos', 'lido-dao',
         'raydium', 'fio-protocol', 'blockstack', 'the-sandbox', 'curve-dao-token', 'injective-protocol', 'dai', 'gala',
         'binance-bitcoin', 'internet-computer', 'algorand', 'eos', 'wrapped-avax', 'pancakeswap-token', 'staked-ether',
         'chiliz', 'stp-network', 'pendle', 'zcash', 'the-graph', 'decentraland', 'monero', 'arweave', 'sun-token',
         'vechain', 'true-usd', 'compound-governance-token', 'alchemy-pay', 'sushi', 'dash', 'mask-network',
         'quant-network', 'conflux-token', 'just', 'auction', 'axie-infinity', 'pax-gold']

# Importing Saved Data
for coin in coins:
    hourly_data[coin] = pd.read_csv(f"Hourly Data/{coin}_hourly_data.csv", index_col=0, parse_dates=True)
    daily_data[coin] = pd.read_csv(f"Daily Data/{coin}_daily_data.csv", index_col=0, parse_dates=True)

# Extracting Cross Sectional Daily Data
daily_price = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(daily_data['bitcoin'].index)
).rename_axis('Date')

daily_returns = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(daily_data['bitcoin'].index)
).rename_axis('Date')

daily_volume = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(daily_data['bitcoin'].index)
).rename_axis('Date')

daily_market_cap = pd.DataFrame(
    columns=coins,
    index=pd.to_datetime(daily_data['bitcoin'].index)
).rename_axis('Date')

for coin in coins:
    daily_price[coin] = daily_data[coin]['Price']
    daily_returns[coin] = daily_data[coin]['Price'].pct_change()
    daily_volume[coin] = daily_data[coin]['Volume']
    daily_market_cap[coin] = daily_data[coin]['Market Cap']

# # Extracting Cross Sectional Hourly Data
# hourly_price = pd.DataFrame(
#     columns=coins,
#     index=pd.to_datetime(hourly_data['bitcoin'].index)
# ).rename_axis('Timestamp')
#
# hourly_returns = pd.DataFrame(
#     columns=coins,
#     index=pd.to_datetime(hourly_data['bitcoin'].index)
# ).rename_axis('Timestamp')
#
# hourly_volume = pd.DataFrame(
#     columns=coins,
#     index=pd.to_datetime(hourly_data['bitcoin'].index)
# ).rename_axis('Timestamp')
#
# hourly_market_cap = pd.DataFrame(
#     columns=coins,
#     index=pd.to_datetime(hourly_data['bitcoin'].index)
# ).rename_axis('Timestamp')
#
# for coin in coins:
#     hourly_price[coin] = hourly_data[coin]['Price']
#     hourly_returns[coin] = hourly_data[coin]['Price'].pct_change()
#     hourly_volume[coin] = hourly_data[coin]['Volume']
#     hourly_market_cap[coin] = hourly_data[coin]['Market Cap']

# 1.3. Transaction Costs -------------------------------------------------------------------------------------
def apply_transaction_costs(signal_df, returns_df, cost_rate):
    """
    Computes net daily returns by subtracting transaction costs proportional to turnover.

    Parameters
    ----------
    signal_df : pd.DataFrame
        The (T x N) DataFrame of daily weights/signals. Each row sums (in absolute value) to 1
        if you normalized them that way.
    returns_df : pd.DataFrame
        The (T x N) daily returns of each asset.
    cost_rate : float
        The fraction of capital paid for each 1.0 turnover.
        E.g., 0.002 for 20bps (market orders).

    Returns
    -------
    net_returns : pd.Series
        A Pandas Series of daily net returns, index-aligned with signal_df.
    """
    # Gross daily PnL from the strategy
    gross_returns = (signal_df * returns_df).sum(axis=1)

    turnover = signal_df.diff().abs().sum(axis=1)

    if len(turnover) > 0:
        turnover.iloc[0] = signal_df.iloc[0].abs().sum()

    daily_cost = turnover * cost_rate
    net_returns = gross_returns - daily_cost

    return net_returns

# ----------------------------------------------------------------------------------------------------------------------
# 2. Strategy Research
# ----------------------------------------------------------------------------------------------------------------------
# 2.1. Momentum Strategy: Tanh Normalisation -----------------------------------------------------------------
max_sharpe_1 = -np.inf
sharpe_matrix_1 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(1, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        strat_1_m = np.tanh(daily_returns.shift(i).rolling(window=j, min_periods=1).mean())
        strat_1_m = strat_1_m.sub(strat_1_m.mean(axis=1), axis=0)
        strat_1_m = strat_1_m.div(strat_1_m.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_1 = apply_transaction_costs(strat_1_m, daily_returns, MKT_COST)
        sharpe_1 = portfolio_returns_1.mean() / portfolio_returns_1.std() * ANNUAL_FACTOR
        sharpe_matrix_1[i - 1, j - 1] = sharpe_1

        if sharpe_1 > max_sharpe_1:
            max_sharpe_1 = sharpe_1
            best_params_1 = [i, j]

# Best Strategy
strat_1_m = np.tanh(daily_returns.shift(best_params_1[0]).rolling(window=best_params_1[1], min_periods=1).mean())
strat_1_m = strat_1_m.sub(strat_1_m.mean(axis=1), axis=0)
strat_1_m = strat_1_m.div(strat_1_m.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_1 = apply_transaction_costs(strat_1_m, daily_returns, MKT_COST)
cumulative_returns_1 = (1 + portfolio_returns_1).cumprod()
sharpe_1 = portfolio_returns_1.mean() / portfolio_returns_1.std() * ANNUAL_FACTOR
print(f'Sharpe Ratio for (1) Tanh Normalised Momentum Strategy: {sharpe_1}')

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_1, cmap='coolwarm')
plt.title("Sharpe Ratio by Lookback Period for Tanh Normalised Momentum Strategy")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()

# 2.2. Momentum Strategy 2: Volatility Adjusted --------------------------------------------------------------
max_sharpe_2 = -np.inf
sharpe_matrix_2 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        strat_2_m = daily_returns.shift(i).rolling(window=j, min_periods=2).mean()
        volatility = daily_returns.shift(i).rolling(window=j, min_periods=2).std()

        strat_2_m = strat_2_m.div(volatility, axis=0)

        strat_2_m = strat_2_m.sub(strat_2_m.mean(axis=1), axis=0)
        strat_2_m = strat_2_m.div(strat_2_m.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_2 = apply_transaction_costs(strat_2_m, daily_returns, MKT_COST)
        sharpe_2 = portfolio_returns_2.mean() / portfolio_returns_2.std() * ANNUAL_FACTOR
        sharpe_matrix_2[i - 1, j - 1] = sharpe_2

        if sharpe_2 > max_sharpe_2:
            max_sharpe_2 = sharpe_2
            best_params_2 = [i, j]

# Best Strategy
strat_2_m = daily_returns.shift(best_params_2[0]).rolling(window=best_params_2[1], min_periods=2).mean()
volatility = daily_returns.shift(best_params_2[0]).rolling(window=best_params_2[1], min_periods=2).std()

strat_2_m = strat_2_m.div(volatility, axis=0)

strat_2_m = strat_2_m.sub(strat_2_m.mean(axis=1), axis=0)
strat_2_m = strat_2_m.div(strat_2_m.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_2 = apply_transaction_costs(strat_2_m, daily_returns, MKT_COST)
cumulative_returns_2 = (1 + portfolio_returns_2).cumprod()
sharpe_2 = portfolio_returns_2.mean() / portfolio_returns_2.std() * ANNUAL_FACTOR
print(f'Sharpe Ratio for (2) Volatility Adjusted Momentum Strategy: {sharpe_2}')

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_2, cmap='coolwarm')
plt.title("Sharpe Ratio by Lookback Period for Volatility Adjusted Momentum Strategy")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()


# 2.3. Momentum Strategy 3: Rank Based -----------------------------------------------------------------------
max_sharpe_3 = -np.inf
sharpe_matrix_3 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        # Momentum Signal: Mean returns for last j days excluding the most recent i days
        strat_3_m = daily_returns.shift(i).rolling(window=j, min_periods=2).mean().rank(1)

        strat_3_m = strat_3_m.sub(strat_3_m.mean(axis=1), axis=0)
        strat_3_m = strat_3_m.div(strat_3_m.abs().sum(axis=1), axis=0)

        # Performance Metrics
        portfolio_returns_3 = apply_transaction_costs(strat_3_m, daily_returns, MKT_COST)
        sharpe_3 = portfolio_returns_3.mean() / portfolio_returns_3.std() * ANNUAL_FACTOR
        sharpe_matrix_3[i - 1, j - 1] = sharpe_3

        if sharpe_3 > max_sharpe_3:
            max_sharpe_3 = sharpe_3
            best_params_3 = [i, j]

# Best Strategy
strat_3_m = daily_returns.shift(best_params_3[0]).rolling(window=best_params_3[1], min_periods=2).mean().rank(1)

strat_3_m = strat_3_m.sub(strat_3_m.mean(axis=1), axis=0)
strat_3_m = strat_3_m.div(strat_3_m.abs().sum(axis=1), axis=0)

# Performance Metrics
portfolio_returns_3 = apply_transaction_costs(strat_3_m, daily_returns, MKT_COST)
cumulative_returns_3 = (1 + portfolio_returns_3).cumprod()
sharpe_3 = portfolio_returns_3.mean() / portfolio_returns_3.std() * ANNUAL_FACTOR
print(f'Sharpe Ratio for (3) Rank Based Momentum Strategy: {sharpe_3}')

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_3, cmap='coolwarm')
plt.title("Sharpe Ratio by Lookback Period for Rank Based Momentum Strategy")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()


# 2.4. Momentum Strategy 4: Volume and Volatility Adjusted ---------------------------------------------------
max_sharpe_4 = -np.inf
sharpe_matrix_4 = np.zeros((30, 365))

for i in range(1, 31):
    for j in range(2, 366):
        strat_4_m = daily_returns.shift(i).ewm(span=j, adjust=False).mean()
        volatility = daily_returns.shift(i).ewm(span=j, adjust=False).std()

        volume_mean = daily_volume.shift(i).ewm(span=j, adjust=False).mean()
        volume_std  = daily_volume.shift(i).ewm(span=j, adjust=False).std()
        volume_zscore = daily_volume.shift(1).sub(volume_mean, 0).div(volume_std)

        strat_4_m = (strat_4_m.mul(volume_zscore, 0).div(volatility, axis=0)).rank(1)

        strat_4_m = strat_4_m.sub(strat_4_m.mean(axis=1), axis=0)
        strat_4_m = strat_4_m.div(strat_4_m.abs().sum(axis=1), axis=0)

        # Smoothing
        strat_4_m_smoothed = strat_4_m.copy()

        for col in strat_4_m.columns:
            strat_4_m_smoothed[col] = strat_4_m[col].ewm(alpha=0.1).mean()

        strat_4_m = strat_4_m_smoothed

        # Performance Metrics
        portfolio_returns_4 = apply_transaction_costs(strat_4_m, daily_returns, MKT_COST)
        cumulative_returns_4 = (1 + portfolio_returns_4).cumprod()
        sharpe_4 = portfolio_returns_4.mean() / portfolio_returns_4.std() * ANNUAL_FACTOR
        sharpe_matrix_4[i - 1, j - 1] = sharpe_4

        if sharpe_4 > max_sharpe_4:
            max_sharpe_4 = sharpe_4
            best_params_4 = [i, j]

# Best Strategy
# best_params_4 = [4, 30]
strat_4_m = daily_returns.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).mean()
volatility = daily_returns.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).std()

volume_mean = daily_volume.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).mean()
volume_std = daily_volume.shift(best_params_4[0]).ewm(span=best_params_4[1], adjust=False).std()
volume_zscore = daily_volume.shift(1).sub(volume_mean, 0).div(volume_std)

strat_4_m = (strat_4_m.mul(volume_zscore, 0).div(volatility, axis=0)).rank(1)

strat_4_m = strat_4_m.sub(strat_4_m.mean(axis=1), axis=0)
strat_4_m = strat_4_m.div(strat_4_m.abs().sum(axis=1), axis=0)

# Smoothing
strat_4_m_smoothed = strat_4_m.copy()

for col in strat_4_m.columns:
    strat_4_m_smoothed[col] = strat_4_m[col].ewm(alpha=0.1).mean()

strat_4_m = strat_4_m_smoothed

# Performance Metrics
portfolio_returns_4 = apply_transaction_costs(strat_4_m, daily_returns, MKT_COST)
cumulative_returns_4 = (1 + portfolio_returns_4).cumprod()
sharpe_4 = portfolio_returns_4.mean() / portfolio_returns_4.std() * ANNUAL_FACTOR
print(f'Sharpe Ratio for (4) Volatility and Volume Adjusted Momentum Strategy: {sharpe_4}')

# Heatmap
plt.figure(figsize=(18, 4))
sns.heatmap(sharpe_matrix_4, cmap='coolwarm')
plt.title("Sharpe Ratio by Lookback Period for Volatility and Volume Adjusted Momentum Strategy")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()

# 2.5. Reversal Strategy 1: ---------------------------------------------------
max_sharpe_5 = -np.inf
sharpe_matrix_5 = np.zeros(365)

for i in range(1, 365+1):
    # Reversal Signal
    strat_5_r = -daily_returns.shift(1).rolling(i, min_periods=1).mean(0)
    volatility = daily_returns.shift(1).ewm(span=30, adjust=False).std()

    strat_5_r = np.tanh(strat_5_r.div(volatility, axis=0))

    # Adjusting Volume z-Score
    log_volume = np.log(daily_volume + 1)
    volume_mean = daily_volume.shift(1).ewm(span=30, adjust=False).mean()
    volume_std = daily_volume.shift(1).ewm(span=30, adjust=False).std()
    volume_zscore = -daily_volume.shift(1).sub(volume_mean, 0).div(volume_std)

    # Normalisation and Market Neutrality
    strat_5_r = np.tanh(strat_5_r.mul(volume_zscore, axis=0))
    strat_5_r = strat_5_r.sub(strat_5_r.mean(axis=1), axis=0)
    strat_5_r = strat_5_r.div(strat_5_r.abs().sum(axis=1), axis=0)

    # Smoothing
    strat_5_r_smoothed = strat_5_r.copy()

    for col in strat_5_r.columns:
        strat_5_r_smoothed[col] = strat_5_r[col].ewm(alpha=0.1).mean()

    strat_5_r = strat_5_r_smoothed

    # Performance Metrics
    portfolio_returns_5 = apply_transaction_costs(strat_5_r, daily_returns, MKT_COST)
    cumulative_returns_5 = (1 + portfolio_returns_5).cumprod()
    sharpe_5 = portfolio_returns_5.mean() / portfolio_returns_5.std() * ANNUAL_FACTOR
    sharpe_matrix_5[i - 1] = sharpe_5

    if sharpe_5 > max_sharpe_5:
        max_sharpe_5 = sharpe_5
        best_params_5 = i

# Best Strategy
strat_5_r = -daily_returns.shift(1).rolling(16, min_periods=1).mean(0)
volatility = daily_returns.shift(1).ewm(span=30, adjust=False).std()

strat_5_r = np.tanh(strat_5_r.div(volatility, axis=0))

# Adjusting Volume z-Score
log_volume = np.log(daily_volume + 1)
volume_mean = daily_volume.shift(1).ewm(span=30, adjust=False).mean()
volume_std = daily_volume.shift(1).ewm(span=30, adjust=False).std()
volume_zscore = -daily_volume.shift(1).sub(volume_mean, 0).div(volume_std)

# Normalisation and Market Neutrality
strat_5_r = np.tanh(strat_5_r.mul(volume_zscore, axis=0))
strat_5_r = strat_5_r.sub(strat_5_r.mean(axis=1), axis=0)
strat_5_r = strat_5_r.div(strat_5_r.abs().sum(axis=1), axis=0)

# Smoothing
strat_5_r_smoothed = strat_5_r.copy()

for col in strat_5_r.columns:
    strat_5_r_smoothed[col] = strat_5_r[col].ewm(alpha=0.1).mean()

strat_5_r = strat_5_r_smoothed

# Performance Metrics
portfolio_returns_5 = apply_transaction_costs(strat_5_r, daily_returns, MKT_COST)
cumulative_returns_5 = (1 + portfolio_returns_5).cumprod()
sharpe_5 = portfolio_returns_5.mean() / portfolio_returns_5.std() * ANNUAL_FACTOR
print(f'Sharpe Ratio for (5) Reversal Strategy: {sharpe_5}')


# 2.6. Breakout Strategy 1: ---------------------------------------------------
max_sharpe_6 = -np.inf
sharpe_matrix_6 = np.zeros((75, 75))

for long in range(1, 75):
    for short in range(1, 75):
        # Compute rolling high and low
        rolling_high = daily_price.shift(2).rolling(window=long*5, min_periods=1).max()
        rolling_low  = daily_price.shift(2).rolling(window=short*5, min_periods=1).min()

        # Compute breakout strength
        long_strength = (daily_price.shift(1) - rolling_high).clip(lower=0)
        short_strength = (rolling_low - daily_price.shift(1)).clip(lower=0)

        # Signal
        strat_6_b = long_strength - short_strength
        strat_6_b = strat_6_b.rank(axis=1)

        strat_6_b = strat_6_b.sub(strat_6_b.mean(axis=1), axis=0)
        strat_6_b = strat_6_b.div(strat_6_b.abs().sum(axis=1), axis=0)

        # Smoothing
        strat_6_b_smoothed = strat_6_b.copy()

        for col in strat_6_b.columns:
            strat_6_b_smoothed[col] = strat_6_b[col].ewm(alpha=0.1).mean()

        strat_6_b = strat_6_b_smoothed

        # Performance Metrics
        portfolio_returns_6 = apply_transaction_costs(strat_6_b, daily_returns, MKT_COST)
        cumulative_returns_6 = (1 + portfolio_returns_6).cumprod()

        # Calculate the annualized Sharpe ratio
        sharpe_6 = portfolio_returns_6.mean() / portfolio_returns_6.std() * ANNUAL_FACTOR

        sharpe_matrix_6[long - 1, short - 1] = sharpe_6

        if sharpe_6 > max_sharpe_6:
            max_sharpe_6 = sharpe_6
            best_params_6 = [long, short]

# Heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(sharpe_matrix_6, cmap='coolwarm')
plt.title("Sharpe Ratio by Lookback Period for Volatility and Volume Adjusted Momentum Strategy")
plt.xlabel('Momentum Window (days)')
plt.ylabel('Lookback Exclusion (days)')
plt.show()

# Define parameters:
long = 260  # Rolling highest close
short = 25  # Rolling lowest close

# Compute rolling high and low
rolling_high = daily_price.shift(2).rolling(window=long, min_periods=1).max()
rolling_low = daily_price.shift(2).rolling(window=short, min_periods=1).min()

# Compute breakout strength
long_strength = (daily_price.shift(1) - rolling_high).clip(lower=0)
short_strength = (rolling_low - daily_price.shift(1)).clip(lower=0)

# Signal
strat_6_b = long_strength - short_strength
strat_6_b = strat_6_b.rank(axis=1)

strat_6_b = strat_6_b.sub(strat_6_b.mean(axis=1), axis=0)
strat_6_b = strat_6_b.div(strat_6_b.abs().sum(axis=1), axis=0)

# Smoothing
strat_6_b_smoothed = strat_6_b.copy()

for col in strat_6_b.columns:
    strat_6_b_smoothed[col] = strat_6_b[col].ewm(alpha=0.1).mean()

strat_6_b = strat_6_b_smoothed

# Performance Mettrics
portfolio_returns_6 = apply_transaction_costs(strat_6_b, daily_returns, MKT_COST)
cumulative_returns_6 = (1 + portfolio_returns_6).cumprod()

# Calculate the annualized Sharpe ratio
sharpe_6 = portfolio_returns_6.mean() / portfolio_returns_6.std() * ANNUAL_FACTOR
print("Sharpe Ratio:", sharpe_6)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Implementation
# ----------------------------------------------------------------------------------------------------------------------
# Combine the strategy returns into a DataFrame
strategy_returns = pd.DataFrame({
    'Momentum 1': portfolio_returns_1,
    'Momentum 2': portfolio_returns_2,
    'Momentum 3': portfolio_returns_3,
    'Momentum 4': portfolio_returns_4,
    'Reversal': portfolio_returns_5,
    'Breakout': portfolio_returns_6,
})

# Compute the mean returns and covariance matrix
mu = strategy_returns.mean()
Sigma = strategy_returns.cov()

# Optimising Portfolio Weights
ef = EfficientFrontier(mu, Sigma, weight_bounds=(-1, 1))
weights = ef.max_sharpe(risk_free_rate=0)
cleaned_weights = ef.clean_weights()
print("Optimized Weights:", cleaned_weights)

# Combine the strategy returns
weights_series = pd.Series(cleaned_weights)
combined_returns = strategy_returns.dot(weights_series)
cumulative_returns = (1 + combined_returns).cumprod()

# Compute performance metrics
sharpe_portfolio = combined_returns.mean() / combined_returns.std() * ANNUAL_FACTOR
print("Combined Portfolio Annualized Sharpe Ratio:", sharpe_portfolio)

# Plot the cumulative returns of the combined strategy
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns, label="Combined Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Combined Strategy Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()


def calculate_drawdowns(returns):
    """
    Given a Pandas Series of daily returns, this function calculates the cumulative returns, the running maximum of the
    cumulative returns, and the drawdowns (as a fraction).
    """

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    return cum_returns, running_max, drawdowns

for strategy in strategy_returns.columns:
    cum_returns, running_max, drawdowns = calculate_drawdowns(strategy_returns[strategy])
    max_drawdown = drawdowns.min()  # minimum value (most negative) is the maximum drawdown
    print(f"Max drawdown for {strategy}: {max_drawdown:.2%}")

    # Optionally, plot the drawdowns
    plt.figure(figsize=(10, 4))
    plt.plot(drawdowns.index, drawdowns, label=f"{strategy} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title(f"Drawdown for {strategy}")
    plt.legend()
    plt.grid(True)
    plt.show()