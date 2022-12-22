import numpy as np
from datetime import date, timedelta

EPISODE_LEN = 1440


class CryptoEnv:
    def __init__(self):
        self.time = 0   # time step (in minutes) for simulation
        self.BTC = 0
        self.BUSD = 1000

        self.data = np.empty(0)

    def reset(self, targetDate):
        self.time = 0
        self.BTC = 0
        self.BUSD = 1000

        delta = timedelta(days=1)
        self.data = np.concatenate((CryptoEnv.loadByDate(targetDate - delta),
                                   CryptoEnv.loadByDate(targetDate)))

        priceState = self.data[self.time:EPISODE_LEN+self.time]
        return (priceState, self.BTC, self.BUSD)

    def step(self, BUSD):
        """Buy/Sell BTC at new timestep with BUSD amount (calculate with closing price), Returns nextState, done, reward, totalValue"""

        # Previous values
        prevTotalValue = self.BUSD + self.BTC * \
            self.data[EPISODE_LEN+self.time-1][3]

        # Progress time
        self.time += 1
        priceState = self.data[self.time:EPISODE_LEN+self.time]

        # Update values
        if (BUSD > self.BUSD):   # insufficient money for purchase
            BUSD = self.BUSD

        # amount of BTC purchase, can be +/-
        purchaseAmount = BUSD / priceState[-1][3]
        if (purchaseAmount < -self.BTC):  # purchase sold less than total possible
            purchaseAmount = self.BTC

        self.BUSD -= BUSD
        self.BTC += purchaseAmount

        newTotalValue = self.BUSD + self.BTC * priceState[-1][3]
        reward = newTotalValue - prevTotalValue

        # Check done
        if (self.time == EPISODE_LEN):
            done = 1
        else:
            done = 0

        return ((priceState, self.BTC, self.BUSD), done, reward, newTotalValue)

    @staticmethod
    def loadByDate(targetDate: date):
        # Change this path name
        data = CryptoEnv.load(
            f"./training-data/BTCBUSD-1m-{targetDate.year}-{targetDate.month}/BTCBUSD-1m-{targetDate.year}-{targetDate.month}.csv")
        startIndex = (targetDate.day - 1) * 60 * 24
        endIndex = targetDate.day * 60 * 24

        assert(endIndex < len(data) + 1)  # end index is inside

        return data[startIndex:endIndex]

    @staticmethod
    def load(pathname: str):
        """Load from .csv assuming format of binance K-line data"""
        data = np.genfromtxt(pathname, delimiter=',')

        # The columns are [Open High Low Close Volume]
        return data[:, 1:6]
