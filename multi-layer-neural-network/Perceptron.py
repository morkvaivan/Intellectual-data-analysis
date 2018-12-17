import numpy as np

class Perceptron(object):
    def __init__(self, a = 0.5, numIter = 400, ls = [2, 8, 8, 1]):
        self.a = a
        self.numIter = numIter
        self.ls = ls

    def fit(self, X, Y):
        n = len(self.ls)

        W = []

        for i in range(n - 1):
            W.append(np.random.randn(self.ls[i], self.ls[i + 1]) * 0.1)

        B = []
        for i in range(1, n):
            B.append(np.random.randn(self.ls[i]) * 0.1)

        O = []
        for i in range(n):
            O.append(np.zeros([self.ls[i]]))

        D = []
        for i in range(1, n):
            D.append(np.zeros(self.ls[i]))

        actF = []
        dF = []
        for i in range(n - 1):
            actF.append(lambda x : np.tanh(x))

            dF.append(lambda y : 1 - np.square(y))

        actF.append(lambda x : x)
        dF.append(lambda x : np.ones(x.shape))

        for c in range(self.numIter):
            for i in range(len(X)):
                t = Y[i, :]

                O[0] = X[i, :]
                for j in range(n - 1):
                    O[j + 1] = actF[j](np.dot(O[j], W[j]) + B[j])

                D[-1] = np.multiply((t - O[-1]), dF[-1](O[-1]))

                for j in range(n - 2, 0, -1):
                    D[j - 1] = np.multiply(np.dot(D[j], W[j].T), dF[j](O[j]))

                for j in range(n - 1):
                    W[j] = W[j] + self.a * np.outer(O[j], D[j])
                    B[j] = B[j] + self.a * D[j]

        self.outputValues = []

        for i in range(len(X)):
            t = Y[i, :]
            O[0] = X[i, :]

            for j in range(n - 1):
                O[j + 1] = actF[j](np.dot(O[j], W[j]) + B[j])

            self.outputValues.append(O[n-1][0,0])

        return W

    def getOutputValues(self):
        return self.outputValues














