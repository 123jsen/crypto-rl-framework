import numpy as np


class CryptoEnv:
    """Acts as the environment for RL, provides step function that returns next state and reward"""
    def __init__(self, ticker_symbol, datetime):
        time = 0

        pass

    def reset():
        pass

    def step(action):
        pass

    @staticmethod
    def load(pathname):
        """Load from .csv assuming format of binance K-line data"""
        data = np.genfromtxt(pathname, delimiter=',')

        # The columns are [Open High Low Close Volume]
        return data[:,1:6]

