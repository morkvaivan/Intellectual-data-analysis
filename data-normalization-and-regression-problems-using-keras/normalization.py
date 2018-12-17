from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import plot_model

import numpy

numpy.random.seed(7)

dataset = numpy.loadtxt("data.csv", delimiter=",")

# normalize
for i in range(1,10):
    m = max(dataset[:, i])
    dataset[:, i] *= 1 / m
    
X = dataset[: ,0:8]
Y = dataset[: ,0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)

model = Sequential()
model.add(Dense(8, input_shape=(8,)))
model.add(Dense(5,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01, momentum=0.1, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=200)

scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

V = model.predict(X)
