# %% MVLR for One Smaple : Implementation for n Features

import numpy as np
import matplotlib.pyplot as plt

import basic_nodes as nodes

n_point = 100
feature_dim = 5
noise = 0.2
t_th = [3 for _ in range(feature_dim+1)]
t_th = np.array(t_th)

parameters = {i:{'mean':0, 'std':1} for i in range(1, feature_dim+1)}

x_data = np.ones(shape=(n_point, 1))
y_data = x_data * t_th[0]                             
for idx in range(1, feature_dim+1):
    new_x_data = np.random.normal(loc=parameters[idx]['mean'],
                                  scale=parameters[idx]['std'],
                                  size=(n_point, 1))
    x_data = np.hstack([x_data, new_x_data])
    y_data += t_th[idx] * new_x_data    
y_data += noise * np.random.normal(0, 1, size=(n_point, 1))
    
epochs = 5
lr = 0.01
th_list = np.random.normal(0, 1, size=(feature_dim+1, 1))
th_acc = np.array([th_list]).reshape(-1, 1)
loss_list = list()

node1 = [None] + [nodes.MulNode() for _ in range(feature_dim)]
node2 = [None] + [nodes.PlusNode() for _ in range(feature_dim)]
node3 = nodes.MinusNode()
node4 = nodes.SquareNode()

fig, axes = plt.subplots(2, 1, figsize=(20, 12))

for epoch in range(epochs):
    for (X, y) in zip(x_data, y_data):
        z1_list = [None] * (feature_dim+1)
        z2_list, dz2_list, dz1_list, dth_list = \
            z1_list.copy(), z1_list.copy(), z1_list.copy(), z1_list.copy()
        
        for idx in range(1, feature_dim+1):
            z1_list[idx] = node1[idx].forward(th_list[idx], X[idx])
        
        for idx in range(1, feature_dim+1):
            if idx == 1:
                z2_list[idx] = node2[idx].forward(th_list[0], z1_list[idx])
            else:
                z2_list[idx] = node2[idx].forward(z2_list[idx-1], z1_list[idx])
        
        z3 = node3.forward(y, z2_list[-1])
        loss = node4.forward(z3)
        
        dz3 = node4.backward(1)
        _, dz2_list[-1] = node3.backward(dz3)
        
        for idx in reversed(range(1, feature_dim+1)):
            dz2_list[idx-1], dz1_list[idx] = node2[idx].backward(dz2_list[idx])
        dth_list[0] = dz2_list[0]
        
        for idx in reversed(range(1, feature_dim+1)):
            dth_list[idx], _ = node1[idx].backward(dz1_list[idx])
        
        for idx in range(feature_dim+1):
            th_list[idx] = th_list[idx] - (dth_list[idx] * lr)
        
        th_acc = np.hstack([th_acc, np.array(th_list).reshape(-1, 1)])
        loss_list.append(loss)

for idx in range(feature_dim+1):
    axes[0].plot(th_acc[idx], label=r'$\theta_{}$'.format({idx}))
axes[1].plot(loss_list)
axes[0].grid()
axes[1].grid()
