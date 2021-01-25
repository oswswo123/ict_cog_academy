# %% dataset에서 batch를 뽑아서 Cost function 만들기

import numpy as np
import matplotlib.pyplot as plt


def make_dataset(mean, std, n_sample, t_th1, t_th0):
    x_data = np.random.normal(loc=mean, scale=std, size=(n_sample, ))
    y_data = t_th1 * x_data + t_th0
    return x_data, y_data
    

def get_batch(x_data, y_data, batch_size, n_sample):
    random_indices = np.arange(n_sample)
    np.random.shuffle(random_indices)
    random_indices = random_indices[:batch_size]
    x_batch = x_data[random_indices]
    y_batch = y_data[random_indices]
    return x_batch, y_batch

    
mean, std = 0, 1
n_sample = 100
batch_size = 8
t_th1, t_th0 = 2, 1

x_data, y_data = make_dataset(mean, std, n_sample, t_th1, t_th0)
x_batch, y_batch = get_batch(x_data, y_data, batch_size, n_sample)

th1 = np.linspace(t_th1-2, t_th1+2, 100)
th0 = np.linspace(t_th0-2, t_th0+2, 100)
Th1, Th0 = np.meshgrid(th1, th0)

cost = np.zeros(shape=(100, 100))
for data_idx in range(batch_size):
    x, y = x_batch[data_idx], y_batch[data_idx]
    
    loss = (y - (Th1*x + Th0))**2
    cost += loss
cost = cost / batch_size

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_wireframe(Th1, Th0, cost)

ax2 = fig.add_subplot(1, 2, 2)
levels = np.geomspace(cost.min(), cost.max(), 20)
ax2.contour(Th1, Th0, cost, levels=levels)

# %% batch를 통해 learning 구현

import numpy as np
import matplotlib.pyplot as plt


def make_dataset(mean, std, n_sample, t_th1, t_th0):
    x_data = np.random.normal(loc=mean, scale=std, size=(n_sample, ))
    y_data = t_th1 * x_data + t_th0
    return x_data, y_data
    

def get_batch(x_data, y_data, batch_size, n_sample):
    random_indices = np.arange(n_sample)
    np.random.shuffle(random_indices)
    random_indices = random_indices[:batch_size]
    x_batch = x_data[random_indices]
    y_batch = y_data[random_indices]
    return x_batch, y_batch


mean, std = 0, 1
n_sample = 100
batch_size = 16

t_th1, t_th0 = 2, 1

th1 = np.linspace(t_th1-2, t_th1+2, 300)
th0 = np.linspace(t_th0-2, t_th0+2, 300)
Th1, Th0 = np.meshgrid(th1, th0)

x_data, y_data = make_dataset(mean, std, n_sample, t_th1, t_th0)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
th1, th0 = 0.5, -0.5
learning_rate = 0.1
th1_list, th0_list = [th1], [th0]

for ax_idx, ax in enumerate(axes.flat):
    x_batch, y_batch = get_batch(x_data, y_data, batch_size, n_sample)
    
    cost = np.zeros_like(Th1)
    for data_idx in range(batch_size):
        x, y = x_batch[data_idx], y_batch[data_idx]
        loss = (y - (Th1*x + Th0))**2
        cost += loss
    cost = cost / batch_size
    
    prediction = th1 * y_batch + th0
    diff = y_batch - prediction
    
    dth1 = -2*np.mean(diff*x_batch)
    dth0 = -2*np.mean(diff)
    
    th1 = th1 - (learning_rate * dth1)
    th0 = th0 - (learning_rate * dth0)
    
    th1_list.append(th1)
    th0_list.append(th0)
    
    levels = np.geomspace(cost.min(), cost.max(), 20)
    ax.contour(Th1, Th0, cost, levels=levels)
    ax.plot(th1_list, th0_list, marker='o', markersize=5)

fig.tight_layout()
