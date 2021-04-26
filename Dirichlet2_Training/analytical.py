import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(x, y):
    return jnp.sin(jnp.pi*x)*jnp.sinh(jnp.pi*y)/jnp.sinh(jnp.pi)

xs = jnp.linspace(0.0,1.0,101)
ys = jnp.linspace(0.0,1.0,101)

f_vect = vmap(vmap(f, (0, None)), (None, 0))
analytical = f_vect(xs, ys) + f_vect(xs, ys).T

plt.pcolormesh(xs, ys, analytical, cmap='viridis')
plt.colorbar()
plt.show()

X, Y = np.meshgrid(xs, ys)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, analytical, cmap=cm.gnuplot)
plt.show()
