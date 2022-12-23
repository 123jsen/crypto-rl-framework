from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def simpleModel(stateSize, actionSize, lr):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=stateSize))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actionSize, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model
