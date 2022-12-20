from collections import deque

class Agent:
    def __init__(self, gamma=0.95, lr=0.001):
        self.gamma = gamma
        self.lr = lr
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

    # Model will be LSTM/CNN
    def _build_model(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        pass

    def act(self, state):
        pass