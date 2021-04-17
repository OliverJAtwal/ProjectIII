import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DLmethods = ['gd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam']

xs = np.linspace(-3,3,101)
ys = np.exp(-xs**2)

for m in DLmethods:
    print(m)

    url = '/Users/oliver/Desktop/Mathematics/Y3/Project/DLDataGen/{}/'.format(m)

    y_data = pd.read_csv('{}y_data.csv'.format(url))
    abs_error_data = pd.DataFrame()
    for y in y_data:
        abs_error_data[y] = np.abs(y_data[y] - ys)
    loss_data = pd.read_csv('{}loss_data.csv'.format(url))
    time_data = pd.read_csv('{}time_data.csv'.format(url))

    print('Median Time: {}'.format(time_data.quantile(q=0.5, axis=1)[0]))
    print('Median Loss: {}'.format(loss_data.quantile(q=0.5, axis=1)[4999]))
    print('Mean Median Error: {}'.format(np.mean(abs_error_data.quantile(q=0.5, axis=1))))
    print('Failed: {}'.format(loss_data.shape[1] - loss_data.dropna(axis=1).shape[1]))
    print('------------------------')

fig, ax = plt.subplots()

for m in reversed(DLmethods):
    loss_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DLDataGen/{}/loss_data.csv'.format(m))
    ax.plot(range(5000), loss_data.quantile(q=0.5, axis=1), '-', label=m)
    ax.fill_between(range(5000), loss_data.quantile(q=0.75, axis=1), loss_data.quantile(q=0.25, axis=1), alpha=0.4)

plt.yscale('log')
plt.legend()
plt.show()
