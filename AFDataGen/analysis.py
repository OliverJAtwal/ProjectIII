import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Init = ['Random', 'Xavier', 'He']
AF = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'swish']

xs = np.linspace(-3,3,101)
ys = np.exp(-xs**2)

for i in Init:
    print('------------------------')
    print(i)
    print('------------------------')
    for a in AF:
        print(a)

        url = '/Users/oliver/Desktop/Mathematics/Y3/Project/AFDataGen/{}/{}/'.format(i,a)

        y_data = pd.read_csv('{}y_data.csv'.format(url))
        abs_error_data = pd.DataFrame()
        for y in y_data:
            abs_error_data[y] = np.abs(y_data[y] - ys)
        loss_data = pd.read_csv('{}loss_data.csv'.format(url))
        time_data = pd.read_csv('{}time_data.csv'.format(url))

        print('Mean Median Error: {}'.format(np.mean(abs_error_data.quantile(q=0.5, axis=1))))
        print('Median Loss: {}'.format(loss_data.quantile(q=0.5, axis=1)[4999]))
        print('Median Time: {}'.format(time_data.quantile(q=0.5, axis=1)[0]))
        print('Failed: {}'.format(loss_data.shape[1] - loss_data.dropna(axis=1).shape[1]))
        print('------------------------')
