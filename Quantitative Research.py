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
from binance.client import Client
import datetime as dt

# Constants
START = dt.datetime(year=2018, month=1, day=1)
END = dt.datetime(year=2025, month=12, day=31)
SEED = 2025

# Functions
def format_binance(data):
    data = pd.DataFrame(
        data,
        columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])

    columns_float = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades',
                     'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']

    for col in columns_float:
        data[col] = data[col].astype(float)

    # Convert from POSIX timestamp
    data['Open Time'] = data['Open Time'].map(lambda x: dt.datetime.fromtimestamp(x / 1000))
    data['Close Time'] = data['Close Time'].map(lambda x: dt.datetime.fromtimestamp(x / 1000))
    return data



# 1.2. Data Wrangling ----------------------------------------------------------------------------------------
# Binance Client
client = Client()

# Tickers
stable_coins = {'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT', 'FDUSDUSDT'}

tickers = (
    pd.DataFrame(client.get_ticker())
    .pipe(lambda df: df[df['symbol'].str.endswith('USDT')])  # Keep USDT pairs
    .pipe(lambda df: df.assign(quoteVolume=pd.to_numeric(df['quoteVolume'])))  # Convert to numeric
    .pipe(lambda df: df[~df['symbol'].isin(stable_coins)])  # Remove stablecoins
    .pipe(lambda df: df.sort_values(by='quoteVolume', ascending=False).head(1000)['symbol'].values)  # Keep top 100
)

# Candlestick Data
data = {}
for ticker in tickers:
    df = client.get_historical_klines(
        symbol=ticker,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=str(START),
        end_str=str(END)
    )

    df = format_binance(df)
    data[ticker] = df
    data[ticker] = data[ticker].set_index(pd.to_datetime(data[ticker]['Open Time']).dt.strftime("%Y-%m-%d"))
    data[ticker] = data[ticker].rename_axis('Date')

del df, stable_coins


# ----------------------------------------------------------------------------------------------------------------------
# 2. Strategy Research
# ----------------------------------------------------------------------------------------------------------------------
# 2.1. Momentum Strategy -------------------------------------------------------------------------------------
# Daily Returns
returns = pd.DataFrame(
    columns=tickers,
    index=pd.to_datetime(data['BTCUSDT']['Open Time']).dt.strftime("%Y-%m-%d")
).rename_axis('Date')

for ticker in tickers:
    returns[ticker] = (data[ticker]['Close'] - data[ticker]['Open']) / data[ticker]['Open']







# ----------------------------------------------------------------------------------------------------------------------
# 3. Implementation
# ----------------------------------------------------------------------------------------------------------------------

