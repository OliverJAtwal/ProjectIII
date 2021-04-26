import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x0 = 1.9
y0 = 1.9

iterations = 200

def f(x, y):
    return (2*x**2 - 1.05*x**4 + (1/6)*x**6 + x*y + y**2)**(1/2)

def dfdx(x, y):
    return (x**5-(21*x**3)/5+4*x+y)/(2*(x**6/6-(21*x**4)/20+2*x**2+y*x+y**2)**(1/2))

def dfdy(x, y):
    return (2*y+x)/(2*(y**2+x*y+x**6/6-(21*x**4)/20+2*x**2)**(1/2))

def gd(x, y, its, a):
    for i in range(its):
        x.append(x[i]-a*dfdx(x[i], y[i]))
        y.append(y[i]-a*dfdy(x[i], y[i]))
    return (x, y)

def momentumn(x, y, its, a, b):
    vx = [0]
    vy = [0]
    for i in range(its):
        vx.append(b*vx[i]+(1-b)*dfdx(x[i],y[i]))
        vy.append(b*vy[i]+(1-b)*dfdy(x[i],y[i]))
        x.append(x[i]-a*vx[i+1])
        y.append(y[i]-a*vy[i+1])
    return (x, y)

def nag(x, y, its, a, b):
    vx = [0]
    vy = [0]
    for i in range(its):
        vx.append(b*vx[i]+a*dfdx(x[i]-b*vx[i],y[i]-b*vy[i]))
        vy.append(b*vy[i]+a*dfdy(x[i]-b*vx[i],y[i]-b*vy[i]))
        x.append(x[i]-vx[i+1])
        y.append(y[i]-vy[i+1])
    return (x, y)

def adagrad(x, y, its, a):
    sx = [0]
    sy = [0]
    for i in range(its):
        sx.append(sx[i]+dfdx(x[i],y[i])**2)
        sy.append(sy[i]+dfdy(x[i],y[i])**2)
        x.append(x[i]-a*sx[i+1]**(-1/2)*dfdx(x[i],y[i]))
        y.append(y[i]-a*sy[i+1]**(-1/2)*dfdy(x[i],y[i]))
    return (x, y)

def rmsprop(x, y, its, a, b):
    sx = [0]
    sy = [0]
    for i in range(its):
        sx.append(b*sx[i]+(1-b)*dfdx(x[i],y[i])**2)
        sy.append(b*sy[i]+(1-b)*dfdy(x[i],y[i])**2)
        x.append(x[i]-a*sx[i+1]**(-1/2)*dfdx(x[i],y[i]))
        y.append(y[i]-a*sy[i+1]**(-1/2)*dfdy(x[i],y[i]))
    return (x, y)

def adam(x, y, its, a, b1, b2):
    vx = [0]
    vy = [0]
    sx = [0]
    sy = [0]
    for i in range(its):
        vx.append(b1*vx[i]+(1-b1)*dfdx(x[i],y[i]))
        vy.append(b1*vy[i]+(1-b1)*dfdy(x[i],y[i]))
        sx.append(b2*sx[i]+(1-b2)*dfdx(x[i],y[i])**2)
        sy.append(b2*sy[i]+(1-b2)*dfdy(x[i],y[i])**2)
        x.append(x[i]-a*sx[i+1]**(-1/2)*vx[i+1])
        y.append(y[i]-a*sy[i+1]**(-1/2)*vy[i+1])
    return (x, y)

xs = np.linspace(-2,2,1001)
ys = np.linspace(-2,2,1001)

X, Y = np.meshgrid(xs, ys)

Z = f(X, Y)

GDx, GDy = gd([x0], [x0], iterations, 0.05)
Mx, My = momentumn([x0], [x0], iterations, 0.1, 0.9)
NAGx, NAGy = nag([x0], [x0], iterations, 0.02, 0.9)
ADAx, ADAy = adagrad([x0], [x0], iterations, 0.1)
RMSx, RMSy = rmsprop([x0], [x0], iterations, 0.05, 0.9)
ADAMx, ADAMy = adam([x0], [x0], iterations, 0.05, 0.9, 0.9)

plt.figure(figsize=(12,6))

plt.contourf(X, Y, Z, 10)
plt.plot(GDx, GDy, label='Gradient Descent')
plt.plot(Mx, My, label='Momentumn')
plt.plot(NAGx, NAGy, label='Nesterov')
plt.plot(ADAx, ADAy, label='AdaGrad')
plt.plot(RMSx, RMSy, label='RMS Prop')
plt.plot(ADAMx, ADAMy, label='ADAM')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))

plt.plot(range(iterations+1), np.square(GDx) + np.square(GDy), label='Gradient Descent')
plt.plot(range(iterations+1), np.square(Mx) + np.square(My), label='Momentumn')
plt.plot(range(iterations+1), np.square(NAGx) + np.square(NAGy), label='Nesterov')
plt.plot(range(iterations+1), np.square(ADAx) + np.square(ADAy), label='AdaGrad')
plt.plot(range(iterations+1), np.square(RMSx) + np.square(RMSy), label='RMS Prop')
plt.plot(range(iterations+1), np.square(ADAMx) + np.square(ADAMy), label='ADAM')
plt.yscale('log')
plt.legend()
plt.show()
