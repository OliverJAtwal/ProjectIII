import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2*x**3+2

fv = np.vectorize(f)

def momentumn_step(v, x, b, a):
    v = b*v + a*(4*(x-b*v)**3 - 6*(x-b*v)**2)
    return (v, x - v)

xs = np.linspace(-1,2.25,100)
ys = f(xs)

bs = [0.5, 0.8, 0.99]
a = 0.05

iterations = 100

for b in bs:
    xi = [-0.9]
    vi = [0]

    for i in range(0,iterations):
        v_new, x_new = momentumn_step(vi[i], xi[i], b, a)
        xi.append(x_new)
        vi.append(v_new)

    yi = fv(xi)

    title = "Nesterov Descent for Learning Rate {}, Momentum {} (iteration {})".format(a, b, iterations)
    plt.title(title)
    plt.scatter(xi, yi, color='r')
    plt.plot(xi, yi, color='r')
    plt.plot(xs, ys, zorder=-1)
    plt.show()

    title = "Velocities for Learning Rate {}, Momentum {} (iteration {})".format(a, b, iterations)
    plt.title(title)
    plt.plot(range(0,iterations+1), vi)
    plt.show()

    abs_error = [abs(i-1.5) for i in xi]

    title = "Absolute Error for Learning Rate {}, Momentum {} (iteration {})".format(a, b, iterations)
    plt.title(title)
    plt.plot(range(iterations+1), abs_error)
    plt.yscale('log')
    plt.show()
