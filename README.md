# Overview

This python module acts as an environment in which you can plug in a reinforcement learning model to train it to buy and sell crypto. In the timeframe of a day, it passes cryptocurrency prices and indicators to the agent, and the agent can buy or sell the cryptocurrency. The model will calculate the total value of the portfolio and earnings will be the reward.

## Specifications
`environment.py` contains a class `CryptoEnv`, which simulates the environment of a MDP.

- **State**: A tuple containing an array of size `5 x 1440`, which is an array of 1440 minutes (24 hours) of `(open, high, low, close, volume)` data, then contains `BTCamount` and `BUSDamount`. (1440 is a adjustable parameter)

- **Reward**: Returns the gain/loss by the player at this timestep, i.e. `BTCamount * todayClose - BTCamount * prevClose`

- **Action**: At this moment, the player can input a number to denote the amount of bitcoin that the agent wants to buy/sell.

Every episode of the simulation consists of 1440 timesteps, which loads the most recent 1440 entries of the price data. This means two days of price data is loaded for every day of simulation.

## Agent

The demo agent will be DQN based, and it can take the discrete actions: `+100/+50/+10/0/-10/-50/-100`, which buys BTC using `x` amount of BUSD

Training data is price data of BTC to BUSD trading pair, obtained from binance's public API. Since the data available goes from Sept 19, 2019 to Nov 30, 2022 (1169 days, inclusive), we will train the agent with 2.5 years of data and test it with 0.5 years of data. The training data will be from 1 Oct 2019 to 31 May 2022 (974 days), and the testing data will be from 1 June 2022 to 30 Nov 2022 (154 days). Some days are bugged due to lack of data on binance api, we will fix the problem later