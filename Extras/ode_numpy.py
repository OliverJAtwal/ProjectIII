# NUMPY implementation
# Note that this is highly unstable due the to use of gradient descent and ReLU
# This is a proof of concept that the ideas we are using can be implemented in a low level setting
# Please see the JAX code for a reliable ODE solver

import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    if z < 0:
        return 0
    else:
        return z

def dreludz(z):
    if z < 0:
        return 0
    else:
        return 1

def N(p, g, X):
    for i in range(len(p)):
        X = g[i](np.dot(X, p[i][0]) + p[i][1])
    return X

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

def loss(p, g, dgdz, X):
    l1 = (dNdx(p, g, dgdz, X) + 2*X*N(p, g, X))**2
    return np.mean(l1)

def GD(T, g, dgdz, X, epochs=1000, lr=0.0001):
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('epoch: {} loss: {:.8f}'.format(epoch, loss(T, g, dgdz, X)))
        grads = dJdT(T, g, dgdz, X)
        T = [(w - lr*vw, b - lr*vb) for (w, b), (vw, vb) in zip(T, grads)]
    return T

n = [1, 16, 16, 1]
params = [(np.random.normal(scale=1/a, size=(a, b)), np.random.normal(scale=1/a, size=b)) for a, b, in zip(n[:-1], n[1:])]

relu_vect = np.vectorize(relu)
dreludz_vect = np.vectorize(dreludz)
g = [relu_vect, relu_vect, relu_vect]
dgdz = [dreludz_vect, dreludz_vect, dreludz_vect]

X = np.array([np.linspace(-3, 3, 101)]).T

trained_params = GD(params, g, dgdz, X, 1000)

Y = N(trained_params, g, X)

plt.plot(X, Y, label=r'$N(x)$')
plt.plot(X, Y/Y[50], label=r'$\hat{f}(x)$')
plt.plot(X, np.exp(-X**2), label='Exact')
plt.legend()
plt.show()
