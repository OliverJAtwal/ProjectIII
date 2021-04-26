import numpy as np

n = [1, 2, 2,1]
params = [(np.random.normal(size=(a, b)), np.random.normal(size=b)) for a, b, in zip(n[:-1], n[1:])]

def sigmoid(z):
    return 1/(1-np.exp(-z))

def dsigmoiddz(z):
    return sigmoid(z)*(1-sigmoid(z))

g = [sigmoid, sigmoid, sigmoid]
dgdz = [dsigmoiddz, dsigmoiddz, dsigmoiddz]

def dJdT(T, g, dgdz, X):
    Z = []
    A = [X]
    L = len(T)
    m = len(X)
    for i in range(L):
        Z.append(np.dot(A[i], T[i][0]) + T[i][1])
        A.append(g[i](Z[i]))

    dNdx = dgdz[L-1](Z[L-1])
    for i in range(L-1,0,-1):
        dNdx = np.dot(dNdx, T[i][0].T)*dgdz[i-1](Z[i-1])
    dNdx = np.dot(dNdx, T[0][0].T)

    dLdz = 4*X*(dNdx+2*X*A[L])*dgdz[L-1](Z[L-1])
    grads = []
    for i in range(L-1,-1,-1):
        grads.append(((1/m)*np.dot(A[i].T, dLdz), np.mean(dLdz, axis=0)))
        dLdz = np.dot(dLdz, T[i][0].T)*dgdz[i-1](Z[i-1])
    return grads[::-1]

X = np.array([[1], [2], [3], [4]])

print(dJdT(params, g, dgdz, X))
