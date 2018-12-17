from perceptron import Perceptron
import matplotlib.pyplot as plt
import pandas as pd

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
y = [0, 0, 0, 1]

for x in range(0,4):
    y[x] = (x1[x] and x2[x])

myData = {'x1': x1, 'x2': x2, 'y': y}

df = pd.DataFrame(data=myData)

print(df)

df.tail()

X = df.iloc[:, 0:2].values
Y = df.iloc[:, 2].values

ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit(X, Y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.savefig('image')

print("\n")
print(ppn.predict(X))
