from Laplace_Solver import Laplace_Solver

import jax.numpy as jnp
from jax import vmap
from jax.nn import swish, sigmoid
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def linear(x):
    return x

layers = [2, 16, 16, 16, 1]
activations = [jnp.tanh, jnp.tanh, jnp.tanh, jnp.tanh]

model = Laplace_Solver(layers, activations)

def cost(p, x, y):
    l1 = model.d2fdx2_vect(p, x, y) + model.d2fdy2_vect(p, x, y)
    return jnp.mean(l1**2)

xs = jnp.linspace(0.0,1.0,101)
ys = jnp.linspace(0.0,1.0,101)

model.train(xs, ys, cost, learning_rate=0.01, epochs=1001, momentum=0.99)

def solution(x,y):
    return jnp.sin(2*jnp.pi*x)-jnp.sin(2*jnp.pi*y)-x*y*(x-1)*(y-1)*model.f(model.params, x, y)

solution_vect = vmap(vmap(solution, (0, None)), (None, 0))
prediction = solution_vect(xs, ys)

X, Y = np.meshgrid(xs, ys)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, prediction, cmap=cm.coolwarm)
plt.show()

def f(x, y):
    return jnp.sin(2*jnp.pi*x)*jnp.sinh(2*jnp.pi*y)/(2*jnp.pi*jnp.sinh(2*np.pi))

def g(x, y):
    return f(x, y) + f(x, y-1) - f(y, x) - f(y, x-1)

f_vect = vmap(vmap(g, (0, None)), (None, 0))
analytical = f_vect(xs, ys)

error = np.abs(prediction - analytical)
print('MAE: {}'.format(np.mean(error)))

plt.pcolormesh(xs, ys, error, cmap='viridis', norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
