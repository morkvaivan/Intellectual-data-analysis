from perceptron import Perceptron
import numpy as np

X = np.matrix([[0.0, 0.0],
               [0.0, 1.0],
               [1.0, 0.0],
               [1.0, 1.0],])

Y = np.matrix([[0.0], [0.0], [0.0], [1.0]])

ls = [2, 4, 1]

neuralNetwork = Perceptron(ls=ls)
layersOfWeights = neuralNetwork.fit(X, Y)

n = len(ls)

print('\n Weights after training:')

for i in range(n - 1):
    print('Layer ' + str(i + 1) + ':\n' + str(layersOfWeights[i]) + '\n')

print('\n Output values:')
print(neuralNetwork.getOutputValues())
