# comparison of the NN fixed method against the runge kutta integrations
# see that the nn method is significantly slower and less precise

from ode_jax import ODE_Solver
from scipy.integrate import solve_ivp
from jax.nn import swish
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from time import time

def linear(x):
    return x

layers = [1, 16, 16, 1]
activations = [swish, swish, swish]

nn_model = ODE_Solver(layers, activations)

def loss(p, x):
    l1 = nn_model.dfdx_vect(p, x) - jnp.cos(x)
    l2 = nn_model.f(p, 0) - 1
    return jnp.mean(l1**2) + l2**2

xs = jnp.linspace(0,2,200)

epochs = 1001
t0 = time()
losses = nn_model.train(xs, loss, epochs=epochs, learning_rate=0.01, momentum=0.99)
t1 = time()
print("NN train time: {}".format(t1-t0))

def de(t, y): return np.cos(t)

t2 = time()
rk8_model = solve_ivp(de, [0, 2], [1], dense_output=True, method='DOP853')
t3 = time()
print("RK8 train time: {}".format(t3-t2))


t4 = time()
rk4_model = solve_ivp(de, [0, 2], [1], dense_output=True, method='RK45')
t5 = time()
print("RK4 train time: {}".format(t5-t4))

plt.plot(xs, nn_model.predict(xs), label='NN', color='blue')
plt.plot(xs, rk8_model.sol(xs)[0], label='RK8', color='red')
plt.plot(xs, rk4_model.sol(xs)[0], label='RK4', color='green')
plt.plot(xs, (1+np.sin(xs)), label='exact')
plt.legend()
plt.show()

print("NN mean abs error: {}".format(np.mean(np.abs((1+np.sin(xs))-nn_model.predict(xs)))))
print("RK8 mean abs error: {}".format(np.mean(np.abs((1+np.sin(xs))-rk8_model.sol(xs)[0]))))
print("RK4 mean abs error: {}".format(np.mean(np.abs((1+np.sin(xs))-rk4_model.sol(xs)[0]))))

ax = plt.gca()
ax.set_yscale('log')
ax.set_title("Absolute Error")
plt.plot(xs, np.abs((1+np.sin(xs))-nn_model.predict(xs)), label='NN', color='blue')
plt.plot(xs, np.abs((1+np.sin(xs))-rk8_model.sol(xs)[0]), label='RK8', color='red')
plt.plot(xs, np.abs((1+np.sin(xs))-rk4_model.sol(xs)[0]), label='RK4', color='green')
plt.legend()
plt.ylabel("log error")
plt.xlabel("x")
plt.show()
