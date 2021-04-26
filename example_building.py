import numpy as np

n = [1, 2, 2,1]
params = [(np.random.normal(size=(a, b)), np.random.normal(size=b)) for a, b, in zip(n[:-1], n[1:])]

def sigmoid(z):
    return 1/(1-np.exp(-z))

g = [sigmoid, sigmoid, sigmoid]

def N(t, g, X):
    for i in range(len(t)):
        X = g[i](np.dot(X, t[i][0]) + t[i][1])
    return X

X = [[1], [2], [3], [4]]

print(N(params, g, X))
