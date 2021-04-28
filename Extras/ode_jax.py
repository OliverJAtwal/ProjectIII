import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.nn import swish
import matplotlib.pyplot as plt

class ODE_Solver:

    def f(self, p, x):
        for i in range(len(p)):
            x = self.activations[i](jnp.dot(x, p[i][0]) + p[i][1])

        return jnp.reshape(x, ())

    def __init__(self, layer_sizes, activations):
        self.params = [(np.random.uniform(-2/m, 2/m, size=(m, n)), np.random.uniform(-2/m, 2/m, size=n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.layer_sizes = layer_sizes
        self.activations = activations

        self.f_vect = vmap(self.f, (None, 0))
        self.dfdx = grad(self.f, 1)
        self.dfdx_vect = vmap(self.dfdx, (None, 0))
        self.d2fdx2 = grad(self.dfdx, 1)
        self.d2fdx2_vect = vmap(self.d2fdx2, (None, 0))

    def train(self, xs, loss_function, epochs=1000, learning_rate=0.1, momentum=0.9):
        jit_loss = jit(loss_function)
        grad_loss = jit(grad(loss_function))
        velocity = [(np.zeros((m, n)), np.zeros(n)) for m, n, in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        losses = []

        for epoch in range(epochs):
            if epoch % 100 == 0:
                print('epoch: %3d loss: %.8f' % (epoch, 1000*jit_loss(self.params, xs)))

            losses.append(jit_loss(self.params, xs))

            # Nesterov accelerated gradient decent
            pv = [(w - momentum*vw, b - momentum*vb) for (w, b), (vw, vb) in zip(self.params, velocity)]
            grads = grad_loss(pv, xs)
            velocity = [(momentum*vw + learning_rate*dw, momentum*vb + learning_rate*db) for (vw, vb), (dw, db) in zip(velocity, grads)]
            self.params = [(w - vw, b - vb) for (w, b), (vw, vb) in zip(self.params, velocity)]

        return losses

    def predict(self, xs):
        return self.f_vect(self.params, xs)

if __name__ == '__main__':

    def linear(x):
        return x

    layers = [1, 16, 16, 1]
    activations = [swish, swish, swish]

    model = ODE_Solver(layers, activations)

    def loss(p, x):
        l1 = model.dfdx_vect(p, x) + 2*x*model.f_vect(p, x)
        l2 = model.f(p, 0.0) - 1.0
        return jnp.mean(l1**2) + l2**2

    xs = jnp.linspace(-3,3,101)
    losses = model.train(xs, loss, epochs=1001, learning_rate=0.001, momentum=0.99)

    ys = model.predict(xs)/model.f(model.params, 0)

    plt.plot(xs, ys, label=r'$N(x)$')
    plt.plot(xs, ys/ys[50], label=r'$\hat{f}(x)$')
    plt.plot(xs, jnp.exp(-xs**2), label='exact')
    plt.legend()
    plt.show()

    print("Mean Absolute Error: {}".format(np.mean(np.abs(ys/ys[50]-jnp.exp(-xs**2)))))

    fig = plt.figure()
    ax = fig.gca()
    ax.set_yscale('log')
    plt.plot(xs, np.abs(ys/ys[50]-jnp.exp(-xs**2)))
    plt.title("Mean Squared Error")
    plt.ylabel("error")
    plt.xlabel("x")
    plt.show()

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Losses")
    ax.set_yscale('log')
    plt.plot(range(len(losses)), losses)
    plt.ylabel("J")
    plt.xlabel("epochs")
    plt.show()
