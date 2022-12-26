import numpy as np
import random
from environment import CryptoEnv
from datetime import date, timedelta

from collections import deque
from model import simpleModel

# 720 * 5 for stock price, 1 for current BTC, 1 for current BUSD
STATE_SIZE = CryptoEnv.stateLen * 5 + 2
ACTION_SIZE = 7


def flatten(state):
    """Returns flattened state"""
    state = np.concatenate((state[0].flatten(), [state[1]], [state[2]]))
    return np.reshape(state, (1, -1))


class Agent:
    def __init__(self, gamma=0.95, lr=0.001, epsilonDecay=0.995, epsilonMin=0.01):
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1.0
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin

        # Other structures
        self.memory = deque(maxlen=2000)
        self.model = simpleModel(stateSize=STATE_SIZE,
                                 actionSize=ACTION_SIZE, lr=self.lr)

    """The state in memory is a """

    def remember(self, episodeDate, time, money, action, done, reward):
        self.memory.append((episodeDate, time, money, action, done, reward))

    def train(self, batchSize, targetNetwork):
        minibatch = random.sample(self.memory, batchSize)
        for i, (episodeDate, time, money, action, done, reward) in enumerate(minibatch):

            delta = timedelta(days=1)
            data = np.concatenate((CryptoEnv.loadByDate(episodeDate - delta),
                                   CryptoEnv.loadByDate(episodeDate)))

            state = (data[time:CryptoEnv.stateLen + time], money[0], money[1])
            nextState = (data[time + 1:CryptoEnv.stateLen +
                         time + 1], money[2], money[3])

            state = flatten(state)
            nextState = flatten(nextState)

            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(targetNetwork.predict(nextState, verbose=0)[0]))
            targetF = self.model.predict(state, verbose=0)
            targetF[0][action] = target
            self.model.fit(state, targetF, epochs=1, verbose=0)
            if (i % 8 == 0):
                print(f'Training {i/batchSize * 100:.1f}%', end='\r')
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay
        print('Training 100.0%', end='\r')
        print()

    def act(self, flatState):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        act_values = self.model.predict(flatState, verbose=0)
        return np.argmax(act_values[0])

    def save(self, pathname):
        self.model.save_weights(pathname)

    def load(self, pathname):
        self.model.load_weights(pathname)
