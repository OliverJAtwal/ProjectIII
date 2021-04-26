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
activations = [swish, swish, swish, swish]

model = Laplace_Solver(layers, activations)

def cost(p, x, y):
    l1 = model.d2fdx2_vect(p, x, y) + model.d2fdy2_vect(p, x, y)
    l2 = model.dfdy_vectx(p, x, 0.0) - jnp.sin(2*jnp.pi*x)
    l3 = model.dfdx_vecty(p, 0.0, y) + jnp.sin(2*jnp.pi*y)
    l4 = model.dfdy_vectx(p, x, 1.0) - jnp.sin(2*jnp.pi*x)
    l5 = model.dfdx_vecty(p, 1.0, y) + jnp.sin(2*jnp.pi*y)
    return jnp.mean(l2**2+l3**2+l4**2+l5**2)

xs = jnp.linspace(0.0,1.0,101)
ys = jnp.linspace(0.0,1.0,101)

model.train(xs, ys, cost, learning_rate=0.001, epochs=10001, momentum=0.99)

prediction = model.predict(xs, ys)

X, Y = np.meshgrid(xs, ys)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, prediction, cmap=cm.coolwarm)
plt.show()

def f(x, y):
    return jnp.sin(jnp.pi*x)*jnp.sinh(jnp.pi*y)/jnp.sinh(jnp.pi)

f_vect = vmap(vmap(f, (0, None)), (None, 0))
analytical = f_vect(xs, ys) + f_vect(xs, ys).T

error = np.abs(prediction - analytical)
print('MAE: {}'.format(np.mean(error)))

plt.pcolormesh(xs, ys, error, cmap='viridis', norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
