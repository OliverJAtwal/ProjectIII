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

def cost1(p, x, y):
    l1 = model.d2fdx2_vect(p, x, y) + model.d2fdy2_vect(p, x, y)
    return jnp.mean(l1**2)

def cost2345(p, x, y):
    l2 = model.f_vectx(p, x, 0.0)
    l3 = model.f_vecty(p, 0.0, y)
    l4 = model.f_vectx(p, x, 1.0) - jnp.sin(jnp.pi*x)
    l5 = model.f_vecty(p, 1.0, y)
    return jnp.mean(l2**2+l3**2+l4**2+l5**2)

xs = jnp.linspace(0.0,1.0,101)
ys = jnp.linspace(0.0,1.0,101)

model.train(xs, ys, cost2345, learning_rate=0.01, epochs=2001, momentum=0.99)
model.train(xs, ys, cost1, learning_rate=0.0000001, epochs=1001, momentum=0.5)

prediction = model.predict(xs, ys)

X, Y = np.meshgrid(xs, ys)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, prediction, cmap=cm.coolwarm)
plt.show()

def f(x, y):
    return jnp.sin(jnp.pi*x)*jnp.sinh(jnp.pi*y)/jnp.sinh(jnp.pi)

f_vect = vmap(vmap(f, (0, None)), (None, 0))
analytical = f_vect(xs, ys)

error = np.abs(prediction - analytical)
print('MAE: {}'.format(np.mean(error)))

plt.pcolormesh(xs, ys, error, cmap='viridis', norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
