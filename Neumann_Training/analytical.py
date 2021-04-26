import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

import jax.numpy as jnp
from jax import vmap

def f(x, y):
    return jnp.sin(2*jnp.pi*x)*jnp.sinh(2*jnp.pi*y)/(2*jnp.pi*jnp.sinh(2*np.pi))

def g(x, y):
    return f(x, y) + f(x, y-1) - f(y, x) - f(y, x-1)

xs = np.linspace(0,1.0,101)
ys = np.linspace(0,1.0,101)

X, Y = np.meshgrid(xs, ys)

f_vect = vmap(vmap(g, (0, None)), (None, 0))
analytical = f_vect(xs, ys)

plt.pcolormesh(xs, ys, analytical, cmap='viridis')
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, analytical, cmap=cm.gnuplot)
plt.show()
