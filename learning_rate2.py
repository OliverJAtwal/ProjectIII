import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

rates = [0.3, 0.05, 0.01]
iterations = 501

def f(x):
    return x**4 - 2*x**3+2

fv = np.vectorize(f)

def gd(x, a):
    return x - a*(4*x**3 - 6*x**2)

xs = np.linspace(0,2,100)
ys = f(xs)

for a in rates:

    xi = [0.2]

    for i in range(0,iterations):
        xi.append(gd(xi[i], a))

    yi = fv(xi)

    title = "Gradient Descent for Learning Rate {} (iteration {})".format(a, i)
    plt.title(title)
    plt.scatter(xi, yi, color='r')
    plt.plot(xi, yi, color='r')
    plt.plot(xs, ys, zorder=-1)
    plt.show()

    abs_error = [abs(i-1.5) for i in xi]

    title = "Absolute Error for Learning Rate {} (iteration {})".format(a, i)
    plt.title(title)
    plt.plot(range(iterations+1), abs_error)
    plt.yscale('log')
    plt.show()
