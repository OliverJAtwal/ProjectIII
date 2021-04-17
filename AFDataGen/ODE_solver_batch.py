import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.nn import swish, sigmoid, relu, leaky_relu, softplus
import matplotlib.pyplot as plt
from time import time

class ODE_Solver:

    def f(self, p, x):
        for i in range(len(p)):
            x = self.activations[i](jnp.dot(x, p[i][0]) + p[i][1])

        return jnp.reshape(x, ())

    def __init__(self, layer_sizes, activations):
        self.params = [(np.random.normal(scale=2/m, size=(m, n)), np.random.normal(scale=2/m, size=n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes
        self.activations = activations

        self.f_vect = vmap(self.f, (None, 0))
        self.dfdx = grad(self.f, 1)
        self.dfdx_vect = vmap(self.dfdx, (None, 0))
        self.d2fdx2 = grad(self.dfdx, 1)
        self.d2fdx2_vect = vmap(self.d2fdx2, (None, 0))

    def train(self, xs, loss_function, epochs=1000, lr=0.1, momentum=0.9):
        jit_loss = jit(loss_function)
        grad_loss = jit(grad(loss_function))
        velocity = [(np.zeros((m, n)), np.zeros(n)) for m, n, in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        losses = []

        for epoch in range(epochs):
            losses.append(jit_loss(self.params, xs))

            pv = [(w - momentum*vw, b - momentum*vb) for (w, b), (vw, vb) in zip(self.params, velocity)]
            grads = grad_loss(pv, xs)
            velocity = [(momentum*vw + lr*dw, momentum*vb + lr*db) for (vw, vb), (dw, db) in zip(velocity, grads)]
            self.params = [(w - vw, b - vb) for (w, b), (vw, vb) in zip(self.params, velocity)]

        return losses

    def predict(self, xs):
        return self.f_vect(self.params, xs)

if __name__ == '__main__':

    def linear(x):
        return x

    layers = [1, 16, 16, 1]

    a_function = 'sigmoid'
    intialisation = 'He'

    activations = [sigmoid, sigmoid, linear]

    def loss(p, x):
        l1 = model.dfdx_vect(p, x) + 2*x*model.f_vect(p, x)
        l2 = model.f(p, 0) - 0.5
        return jnp.mean(l1**2) + l2**2

    xs = jnp.linspace(-3,3,101)

    y_data = pd.DataFrame()
    mse_data = pd.DataFrame()
    loss_data = pd.DataFrame()
    time_data = pd.DataFrame()

    for i in range(50):
        model = ODE_Solver(layers, activations)
        t0 = time()
        losses = model.train(xs, loss, epochs=5000, lr=0.001, momentum=0.99)
        ys = model.predict(xs)/model.f(model.params, 0)
        elapsed = time() - t0

        print("MSE: {}".format(np.mean((ys-jnp.exp(-xs**2))**2)))
        print("Loss: {}".format(losses[-1]))
        print("Time: {}".format(elapsed))

        y_data['y{}'.format(i)] = ys
        mse_data['MSE{}'.format(i)] = (ys-jnp.exp(-xs**2))**2
        loss_data['LOSS{}'.format(i)] = pd.Series(losses)
        time_data['Time{}'.format(i)] = pd.Series(elapsed)

    root = "/Users/oliver/Desktop/AFDataGen/{}/{}/".format(intialisation, a_function)
    y_data.to_csv('{}y_data.csv'.format(root))
    mse_data.to_csv('{}mse_data.csv'.format(root))
    loss_data.to_csv('{}loss_data.csv'.format(root))
    time_data.to_csv('{}time_data.csv'.format(root))
