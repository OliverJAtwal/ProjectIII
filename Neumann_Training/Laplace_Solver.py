# 2D PDE Solver
# Requires Python 3.6
# & Jax

import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
from jax.nn import swish, sigmoid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Laplace_Solver:

    def f(self, p, x, y):
        x = jnp.array([x, y])
        for i in range(len(p)):
            x = self.activations[i](jnp.dot(x, p[i][0]) + p[i][1])

        return jnp.reshape(x, ())

    def __init__(self, layer_sizes, activations):
        self.params = [(np.random.uniform(-(m)**(-1/2), (m)**(-1/2), size=(m, n)), np.random.uniform(-(m)**(-1/2), (m)**(-1/2), size=n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes
        self.activations = activations

        # Vectorised Functions - Outputs a matrix of values for vectors x and y
        # Vectorised over x AND y
        self.f_vect = vmap(vmap(self.f, (None, 0, None)), (None, None, 0))
        self.dfdx = grad(self.f, 1)
        self.dfdy = grad(self.f, 2)
        self.dfdx_vect = vmap(vmap(self.dfdx, (None, 0, None)), (None, None, 0))
        self.dfdy_vect = vmap(vmap(self.dfdy, (None, 0, None)), (None, None, 0))
        self.d2fdx2 = grad(self.dfdx, 1)
        self.d2fdy2 = grad(self.dfdx, 2)
        self.d2fdx2_vect = vmap(vmap(self.d2fdx2, (None, 0, None)), (None, None, 0))
        self.d2fdy2_vect = vmap(vmap(self.d2fdy2, (None, 0, None)), (None, None, 0))

        # Axis Vectorised Functions - Outputs a vector of values for a fixed x/y and vector y/x
        # Vectorised over x OR y
        self.f_vectx = vmap(self.f, (None, 0, None))
        self.f_vecty = vmap(self.f, (None, None, 0))
        self.dfdx_vectx = vmap(self.dfdx, (None, 0, None))
        self.dfdx_vecty = vmap(self.dfdx, (None, None, 0))
        self.dfdy_vectx = vmap(self.dfdy, (None, 0, None))
        self.dfdy_vecty = vmap(self.dfdy, (None, None, 0))
        self.d2fdx2_vectx = vmap(self.d2fdx2, (None, 0, None))
        self.d2fdx2_vecty = vmap(self.d2fdx2, (None, None, 0))
        self.d2fdy2_vectx = vmap(self.d2fdy2, (None, 0, None))
        self.d2fdy2_vecty = vmap(self.d2fdy2, (None, None, 0))

    def train(self, xs, ys, loss_function, epochs=1001, learning_rate=0.1, momentum=0.9):
        jit_loss = jit(loss_function)
        grad_loss = jit(grad(loss_function))
        velocity = [(np.zeros((m, n)), np.zeros(n)) for m, n, in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        losses = []

        for epoch in range(epochs):
            if epoch % 100 == 0:
                print('epoch: %3d loss: %.6f' % (epoch, 1000*jit_loss(self.params, xs, ys)))

            losses.append(jit_loss(self.params, xs, ys))
            # Nesterov accelerated gradient decent
            pv = [(w - momentum*vw, b - momentum*vb) for (w, b), (vw, vb) in zip(self.params, velocity)]
            grads = grad_loss(pv, xs, ys)
            velocity = [(momentum*vw + learning_rate*dw, momentum*vb + learning_rate*db) for (vw, vb), (dw, db) in zip(velocity, grads)]
            self.params = [(w - vw, b - vb) for (w, b), (vw, vb) in zip(self.params, velocity)]

    def predict(self, xs, ys):
        return self.f_vect(self.params, xs, ys)

if __name__ == '__main__':

    def linear(x):
        return x

    layers = [2, 16, 16, 1]
    activations = [jnp.tanh, jnp.tanh, linear]

    model = TwoD_PDE_Solver(layers, activations)

    def cost(p, x, y):
        l2 = model.f_vectx(p, x, 0.0)
        l3 = model.f_vecty(p, 0.0, y)
        l4 = model.f_vectx(p, x, 1.0) - jnp.sin(jnp.pi*x)
        l5 = model.f_vecty(p, 1.0, y)
        return jnp.mean(l2**2+l3**2+l4**2+l5**2)

    xs = jnp.linspace(0.0,1.0,101)
    ys = jnp.linspace(0.0,1.0,101)

    model.train(xs, ys, cost, learning_rate=0.005, epochs=3001, momentum=0.99)
