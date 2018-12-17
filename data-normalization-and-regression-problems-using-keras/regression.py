import numpy as np

from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda
from keras import optimizers
import math

from matplotlib import pyplot as plt

def createDataFromRange(range):
    f = lambda x1, x2: x1 + x2 ** 2
    X = []
    Y = []
    for x in range:
        X.append([x])
        Y.append([f(x, x)])

    return np.array(X), np.array(Y)

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation('linear'))
model.add(Dense(64))
model.add(Lambda(lambda x: x ** 2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['accuracy'])

r = range(-100, 100)
X, Y = createDataFromRange(r)

model.fit(X, Y, epochs=2000)
V = model.predict(X)

plt.scatter(X.reshape(-1, 1), Y)
plt.plot(X.reshape(-1, 1), V)
plt.show()

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
