# Statistical Arbitrage Crypto Trading Algorithm
## 1. Project Background
Statistical arbitrage is a data-driven trading strategy that seeks to identify and exploit inefficiencies in financial markets by uncovering predictive relationships between price, volume, and other market variables. It is widely used in quantitative hedge funds and high-frequency trading firms due to its ability to systematically generate alpha. Despite its success in traditional equity and futures markets, statistical arbitrage remains relatively underexplored in the cryptocurrency space. Given the decentralized nature, 24/7 trading cycles, and high volatility of digital assets, crypto markets present unique inefficiencies that may be well-suited for statistical arbitrage techniques.

This project aims to research and develop profitable trading strategies by leveraging statistical arbitrage principles in the crypto markets. Specifically, it focuses on two fundamental price behaviors:
1. Momentum – the tendency of assets to continue moving in the same direction after a significant price movement.
2. Reversal – the tendency of assets to mean-revert following excessive price deviations.

By analyzing historical price, volume, and market structure data, the project seeks to uncover robust, data-driven trading signals that can be applied to a systematic trading framework. The ultimate objective is to design and optimize a set of crypto trading strategies that can consistently identify and capture statistical inefficiencies while managing risk and transaction costs effectively.

## 2. Executive Summary
Using daily crypto data from 1st Jan 2019 to 31st Dec 2024, we backtested 6 various trading signals and combined them using mean-variance portfolio theory (Markowitz Theory) and achieved a total **Sharpe Ratio of 2.34** and a **Maximum Drawdown of -27.22%**. Our strategies involved:
1. Tanh Normalised Momentum Strategy
2. Volatility Adjusted Momentum Strategy
3. Rank Based Momentum Strategy
4. Volume and Volatility Adjusted Momentum Strategy
5. Volume and Volatility Adjusted Reversal Strategy
6. Price Breakout Strategy

## 3. Methods and Strategies
### 3.1. Crypto Data
Using the CoinGecko API with the 'Analyst Pro' subscription, we collected daily data for the top 200 coins by market cap, then filtered out those coins with data less than 60% coverage between 1st January 2019 and 31st Dec 2024. This resulted in a total of 72 coins with _daily close price_, _volume_ and _market capitalisation_ which have been stored in the directory 'Daily Data'. We assume that the price data is the close price quoted for the entire day at 23:59:59. We calculate the percentage change per day for the returns of the coins for each day and collate them into a single dataframe _daily_returns_, together with _daily_price_, _daily_volume_ and _daily_market_cap_.

### 3.2. Momentum Strategies
We tested four momentum strategies, each building on the previous to create a more refined signal. All signals were de-meaned and normalised for market neutrality and hence assumed that shorting was possible. We also assumed that limit orders were used with a transaction cost of 7bp. This led to us smoothing all signals using an exponentially weighted mean with alpha selected for highest sharpe, so that the turnover was minimised and hence we could reduce transaction costs. 

We begun with a simple returns based momentum strategy with a uniform lookback period of 365 days and exclusion window of 18 days which were then passed through the tanh function to help curb outliers. The lookback and exclusion were selected so that they gave the highest sharpe ratio from backtesting. 

Next we refined the simple momentum strategy by dividing the signal by the rolling standard deviation on the same window as the momentum window. The ideal windows were an exclusion window of 2 days and a lookback period of 25 days.

We found that if instead of the tanh normalisation, if a rank based normalisation was used instead, the results were much better. Hence for the fourth and final momentum strategy we combined the rank based normalisation with a division of the volatility (reducing volatility) and multiplying by the z-score of the volume on a lookback period. The hypothesis being that coins with higher trading volumes meant that the price momentum is more likely to be due to actual fundamental movements rather than noise. The results of these strategies are discussed below.

### 3.3. Reversal Strategies
We found that while momentum existed at a longer lookback period, the contrarion exists within the price movements on a short term basis. We reversed the signal by computing the rolling mean of the returns, divided by the volatility (to again minimise volatility) and found that the highest sharpe existed at a lookback period of 2 days. Further we refined the signal by multiplying by the negative z-score of the volume on a 30 day window with the hypothesis that if reversals happen with low volumes, it suggests a lack of conviction within the price movements and hence signal a higher chance of mean-reversion.


### 3.4. Breakout Strategies
The final breakout strategy is defined by first computing a rolling high and rolling low price on different windows (to be selected later) and then calculated as the difference between the current past day price and the rolling high and low, passed through a Rectified Linear Unit (ReLU). We calculate the difference between the long signal and the short signal and used a rank based approach to generate the final signal. The lookback periods for the long and short signals were selected for the highest sharpe ratio and were found to be 260 days (long) and 30 days (short). The idea is that if the price movements broke past their moving averages, then it indicated a fundamental movement and hence a signal.

## 4. Results and Analysis
Each of these strategies were used to create an overall portfolio using Markowitz Mean-Variance Portfolio theory. It does so by finding the optimal weights of each strategy such that we maximise the returns while minimising the variance. The Sharpe ratios and maximum drawdowns for each strategy is outlined below. 

| **Volume Condition**  | **Sharpe Ratio** | **Annualised Returns** | **Annualised Volatility**  | **Maximum Drawdown** | **Weight in Overall Portfolio** |
|----------------------|--------------------|---------------------------|-----------------------------|
| **1. Tanh Normalised Momentum** | 1.14055 |  0.9542  |  0.8366  |  -45.53%  |  0.0980  |
| **2. Volatility Adjusted Momentum** |  1.1753  |  0.4143  |  0.3525  | -40.54%  |  -0.27341  |
