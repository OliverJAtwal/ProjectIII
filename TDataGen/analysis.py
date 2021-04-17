import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sgd_y_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/sgd_y_data.csv', index_col=0)
sgd_mse_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/sgd_mse_data.csv', index_col=0)
sgd_loss_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/sgd_loss_data.csv', index_col=0)

mbgd_y_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/mbgd_y_data.csv', index_col=0)
mbgd_mse_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/mbgd_mse_data.csv', index_col=0)
mbgd_loss_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/mbgd_loss_data.csv', index_col=0)

bgd_y_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/bgd_y_data.csv', index_col=0)
bgd_mse_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/bgd_mse_data.csv', index_col=0)
bgd_loss_data = pd.read_csv('/Users/oliver/Desktop/Mathematics/Y3/Project/DataGen/bgd_loss_data.csv', index_col=0)

xs = np.linspace(-3,3,101)
ys = np.exp(-xs**2)

sgd_y_data['q1'] = sgd_y_data.quantile(q=0.25, axis=1)
sgd_mse_data['q1'] = sgd_mse_data.quantile(q=0.25, axis=1)
sgd_loss_data['q1'] = sgd_loss_data.fillna('ffill').quantile(q=0.25, axis=1)
sgd_y_data['median'] = sgd_y_data.quantile(q=0.5, axis=1)
sgd_mse_data['median'] = sgd_mse_data.quantile(q=0.5, axis=1)
sgd_loss_data['median'] = sgd_loss_data.fillna('ffill').quantile(q=0.5, axis=1)
sgd_y_data['q3'] = sgd_y_data.quantile(q=0.75, axis=1)
sgd_mse_data['q3'] = sgd_mse_data.quantile(q=0.75, axis=1)
sgd_loss_data['q3'] = sgd_loss_data.fillna('ffill').quantile(q=0.75, axis=1)

mbgd_y_data['q1'] = mbgd_y_data.quantile(q=0.25, axis=1)
mbgd_mse_data['q1'] = mbgd_mse_data.quantile(q=0.25, axis=1)
mbgd_loss_data['q1'] = mbgd_loss_data.fillna('ffill').quantile(q=0.25, axis=1)
mbgd_y_data['median'] = mbgd_y_data.quantile(q=0.5, axis=1)
mbgd_mse_data['median'] = mbgd_mse_data.quantile(q=0.5, axis=1)
mbgd_loss_data['median'] = mbgd_loss_data.fillna('ffill').quantile(q=0.5, axis=1)
mbgd_y_data['q3'] = mbgd_y_data.quantile(q=0.75, axis=1)
mbgd_mse_data['q3'] = mbgd_mse_data.quantile(q=0.75, axis=1)
mbgd_loss_data['q3'] = mbgd_loss_data.fillna('ffill').quantile(q=0.75, axis=1)

bgd_y_data['q1'] = bgd_y_data.quantile(q=0.25, axis=1)
bgd_mse_data['q1'] = bgd_mse_data.quantile(q=0.25, axis=1)
bgd_loss_data['q1'] = bgd_loss_data.fillna('ffill').quantile(q=0.25, axis=1)
bgd_y_data['median'] = bgd_y_data.quantile(q=0.5, axis=1)
bgd_mse_data['median'] = bgd_mse_data.quantile(q=0.5, axis=1)
bgd_loss_data['median'] = bgd_loss_data.fillna('ffill').quantile(q=0.5, axis=1)
bgd_y_data['q3'] = bgd_y_data.quantile(q=0.75, axis=1)
bgd_mse_data['q3'] = bgd_mse_data.quantile(q=0.75, axis=1)
bgd_loss_data['q3'] = bgd_loss_data.fillna('ffill').quantile(q=0.75, axis=1)

fig, ax = plt.subplots()
ax.plot(xs, sgd_y_data['median'], '-', label='SGD')
ax.plot(xs, mbgd_y_data['median'], '-', label='MBGD')
ax.plot(xs, bgd_y_data['median'], '-', label='BGD')
plt.legend()
plt.show()

sgd_abs_error_data = pd.DataFrame()
for y in sgd_y_data:
    sgd_abs_error_data[y] = np.abs(sgd_y_data[y] - ys)
print('SGD MMAE: {}'.format(np.mean(sgd_abs_error_data.quantile(q=0.5, axis=1))))

mbgd_abs_error_data = pd.DataFrame()
for y in mbgd_y_data:
    mbgd_abs_error_data[y] = np.abs(mbgd_y_data[y] - ys)
print('MBGD MMAE: {}'.format(np.mean(mbgd_abs_error_data.quantile(q=0.5, axis=1))))

bgd_abs_error_data = pd.DataFrame()
for y in bgd_y_data:
    bgd_abs_error_data[y] = np.abs(bgd_y_data[y] - ys)
print('BGD MMAE: {}'.format(np.mean(bgd_abs_error_data.quantile(q=0.5, axis=1))))

fig, ax = plt.subplots()
ax.plot(xs, sgd_abs_error_data.quantile(q=0.5, axis=1), '-', label='SGD')
ax.plot(xs, mbgd_abs_error_data.quantile(q=0.5, axis=1), '-', label='MBGD')
ax.plot(xs, bgd_abs_error_data.quantile(q=0.5, axis=1), '-', label='BGD')
plt.yscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(xs, sgd_mse_data['median'], '-', label='SGD')
ax.fill_between(xs, sgd_mse_data['q3'], sgd_mse_data['q1'], alpha=0.4)
ax.plot(xs, mbgd_mse_data['median'], '-', label='MBGD')
ax.fill_between(xs, mbgd_mse_data['q3'], mbgd_mse_data['q1'], alpha=0.4)
ax.plot(xs, bgd_mse_data['median'], '-', label='BGD')
ax.fill_between(xs, bgd_mse_data['q3'], bgd_mse_data['q1'], alpha=0.4)
plt.yscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(range(len(sgd_loss_data)), sgd_loss_data['median'], '-', label='SGD')
ax.fill_between(range(len(sgd_loss_data)), sgd_loss_data['q3'], sgd_loss_data['q1'], alpha=0.4)
ax.plot(range(len(mbgd_loss_data)), mbgd_loss_data['median'], '-', label='MBGD')
ax.fill_between(range(len(mbgd_loss_data)), mbgd_loss_data['q3'], mbgd_loss_data['q1'], alpha=0.4)
ax.plot(range(len(bgd_loss_data)), bgd_loss_data['median'], '-', label='BGD')
ax.fill_between(range(len(bgd_loss_data)), bgd_loss_data['q3'], bgd_loss_data['q1'], alpha=0.4)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
