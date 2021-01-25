# %% Linear Regreesion 구현

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

t_w, t_b = 3, -2
n_point = 1000
x_data = np.random.normal(loc=0, scale=1, size=(100, ))
y_data = t_w*x_data + t_b + 0.5*np.random.normal(loc=0, scale=1, size=(100, ))

fig, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].scatter(x_data, y_data)

w, b = np.random.normal(0, 1, size=(2, ))
losses = []
epochs = 10
learning_rate = 0.01

cmap = cm.get_cmap('rainbow', lut=len(x_data))

# zip : 여러개의 객체를 묶어서 전달할 때 쓰는 함수
# 괄호로 묶인 객체들을 tuple형태로 묶어서 인자로 전달
for idx, (x, y) in enumerate(zip(x_data, y_data)):
    #### forward propagation(start) #####
    prediction = w*x + b
    loss = (y - prediction)**2
    losses.append(loss)
    #### forward propagation(end) #####
    
    
    #### backward prdiction(start) #####
    dl_dw = -2*x*(y - prediction)
    dl_db = -2*(y - prediction)
    #### backward prdiction(end) #####
    
    
    #### paramter update(start) #####
    w = w - learning_rate*dl_dw
    b = b - learning_rate*dl_db
    #### paramter update(end) #####
    
    
    #### predictor visualization(start) #####
    x_range = np.linspace(x_data.min(), x_data.max(), 2)
    y_range = w*x_range + b
    
    axes[0].plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    #### predictor visualization(end) #####

axes[1].plot(losses)
axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)

# %% Linear Regreesion의 Gradient 구하는 과정을 좀 더 세분화
'''
실제 딥 러닝에서는 여러 layer를 지나며 학습이 진행됨
따라서 위 코드처럼 gradient를 한번에 구하는 것은 불가능할 때가 많다
그러니 Chain Rule을 이용하여 Backward propagation을 좀 더 세분화 해보자
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# target weight 와 target bias set
t_w, t_b = 3, -2
n_point = 500
x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
y_data = t_w*x_data + t_b + 0.5*np.random.normal(loc=0, scale=1, size=(n_point, ))

fig, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].scatter(x_data, y_data)

# predictor의 weight와 bias는 random하게 정해짐
w, b = np.random.normal(0, 1, size=(2, ))
losses = []
learning_rate = 0.01

cmap = cm.get_cmap('rainbow', lut=len(x_data))

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    #### forward propagation(start) #####
    # affine function = z
    z = w * x + b
    loss = (y - z)**2
    losses.append(loss)
    #### forward propagation(end) #####
    
    
    #### backward prdiction(start) #####
    # 편미분을 하고, Chain Rule을 이용하여 과정을 조금 더 세분화
    dl_dz = -1 * 2 * (y - z)
    
    dz_dw = x
    dz_db = 1
    
    dl_dw = dl_dz * dz_dw
    dl_db = dl_dz * dz_db
    #### backward prdiction(end) #####
    
    
    #### paramter update(start) #####
    # parameter를 update할때는 반드시 gradient에 -1을 곱해서!
    w = w + ( -1 * dl_dw * learning_rate )
    b = b + ( -1 * dl_db * learning_rate )
    #### paramter update(end) #####
    
    
    #### predictor visualization(start) #####
    x_range = np.linspace(x_data.min(), x_data.max(), 2)
    y_range = w*x_range + b
    
    axes[0].plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    #### predictor visualization(end) #####

axes[1].plot(losses)
axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)

# %% Linear Regreesion을 Function으로 세분화

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_dataset(n_point, t_w, t_b, noise_coeff):
    # target weight 와 target bias set
    x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
    y_data = t_w*x_data + t_b + noise_coeff*np.random.normal(loc=0, scale=1, size=(n_point, ))
    return x_data, y_data


def forward_propagation(x, y, w, b, losses):
    # affine function = z
    z = w * x + b
    loss = (y - z)**2
    losses.append(loss)
    return z, losses
    

def backward_propagation(x, y, z):
    # 편미분을 하고, Chain Rule을 이용하여 과정을 조금 더 세분화
    dl_dz = -2 * (y - z)
    
    dz_dw = x
    dz_db = 1
    
    dl_dw = dl_dz * dz_dw
    dl_db = dl_dz * dz_db
    return dl_dw, dl_db


def parameter_update(w, b, dl_dw, dl_db, learning_rate):
    # parameter를 update할때는 반드시 gradient에 -1을 곱해서!
    w = w - ( dl_dw * learning_rate )
    b = b - ( dl_db * learning_rate )
    return w, b


# create dataset
n_point = 100
t_w, t_b = 3, -2
noise_coeff = 0.5
x_data, y_data = create_dataset(n_point, t_w, t_b, noise_coeff) 
    
fig, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].scatter(x_data, y_data)

# predictor의 weight와 bias는 random하게 정해짐
w, b = np.random.normal(0, 1, size=(2, ))
losses = []
learning_rate = 0.01

cmap = cm.get_cmap('rainbow', lut=len(x_data))

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    z, losses = forward_propagation(x, y, w, b, losses)
    dl_dw, dl_db = backward_propagation(x, y, z)
    w, b = parameter_update(w, b, dl_dw, dl_db, learning_rate)    
    
    #### predictor visualization(start) #####
    x_range = np.linspace(x_data.min(), x_data.max(), 2)
    y_range = w*x_range + b
    
    axes[0].plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    #### predictor visualization(end) #####

axes[1].plot(losses)
axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)

# %% Linear Regreesion을 다른 package의 Function으로 구현

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from linear_regression_tool import create_dataset
from linear_regression_tool import set_weight_bias
from linear_regression_tool import forward_propagation
from linear_regression_tool import backward_propagation
from linear_regression_tool import parameter_update

n_point = 100
t_w, t_b = 3, -2
noise = 0.3
learning_rate = 0.01
losses = []

x_data, y_data = create_dataset(n_point, t_w, t_b, noise) 
w, b = set_weight_bias()
    
fig, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].scatter(x_data, y_data)
cmap = cm.get_cmap('rainbow', lut=len(x_data))

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    # forward
    z, losses = forward_propagation(x, y, w, b, losses)
    
    # backward
    dl_dw, dl_db = backward_propagation(x, y, z)
    
    # parameter update
    w, b = parameter_update(w, b, dl_dw, dl_db, learning_rate)    
    
    #### predictor visualization(start) #####
    x_range = np.linspace(x_data.min(), x_data.max(), 2)
    y_range = w*x_range + b
    
    axes[0].plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    #### predictor visualization(end) #####

axes[1].plot(losses)
axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)
