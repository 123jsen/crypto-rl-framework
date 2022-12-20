# Overview

This python module acts as an environment in which you can plug in a reinforcement learning model to train it to buy and sell crypto. In the timeframe of a day, it passes cryptocurrency prices and indicators to the agent, and the agent can buy or sell the cryptocurrency. The model will calculate the total value of the portfolio and earnings will be the reward.

## Specifications
`environment.py` contains a class `CryptoEnv`, which simulates the environment of a MDP.

- **State**: Contains an array of size `5 x 1440`, which is an array of 1440 minutes (24 hours) of `(open, high, low, close, volume)` data, then contains `bitcoinAmount` and `cashAmount`. (1440 is a adjustable parameter)

- **Reward**: Returns the gain/loss by the player at this timestep, i.e. `bitcoinAmount * todayClose - bitcoinAmount * prevClose`

- **Action**: At this moment, the player can input a number to denote the amount of bitcoin that the agent wants to buy/sell.
