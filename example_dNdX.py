import numpy as np

n = [1, 2, 2,1]
params = [(np.random.normal(size=(a, b)), np.random.normal(size=b)) for a, b, in zip(n[:-1], n[1:])]

def sigmoid(z):
    return 1/(1-np.exp(-z))

def dsigmoiddz(z):
    return sigmoid(z)*(1-sigmoid(z))

g = [sigmoid, sigmoid, sigmoid]
dgdz = [dsigmoiddz, dsigmoiddz, dsigmoiddz]

def dNdx(p, g, dgdz, X):
    Z = []
    L = len(p)
    for i in range(L):
        Z.append(np.dot(X, p[i][0]) + p[i][1])
        X = g[i](Z[i])

    d = dgdz[L-1](Z[L-1])
    for i in range(L-1,0,-1):
        d = np.dot(d, p[i][0].T)*dgdz[i-1](Z[i-1])
    return np.dot(d, p[0][0].T)

X = [[1], [2], [3], [4]]

print(dNdx(params, g, dgdz, X))
