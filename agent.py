import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 1440 * 5 for stock price, 1 for current BTC, 1 for current BUSD
STATE_SIZE = 1440 * 5 + 2
ACTION_SIZE = 7

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
        self.model = self._buildModel()

    # DQN using simple FCNN
    def _buildModel(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=STATE_SIZE))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, nextState, done, reward):
        self.memory.append((state, action, nextState, done, reward))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, nextState, done, reward in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(nextState)[0]))
                targetF = self.model.predict(state)
                targetF[0][action] = target
                self.model.fit(state, targetF, epochs=1, verbose=0)
            if self.opsilon > self.epsilonMin:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
