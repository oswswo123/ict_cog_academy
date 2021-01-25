# %% SVLR for one Sample (Batch gradient descent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes


def create_dataset(n_point, t_th1, t_th0, noise):
    x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
    y_data = t_th1 * x_data + t_th0 + \
        (noise * np.random.normal(0, 1, size=(n_point, )))
    return x_data, y_data


n_point = 80
t_th1, t_th0 = 3, 2
lr = 0.01
noise = 0.3
epochs = 5

x_data, y_data = create_dataset(n_point, t_th1, t_th0, noise)

th1, th0 = np.random.normal(loc=0, scale=1, size=(2, ))
th1_list, th0_list, loss_list = [th1], [th0], list()

node1 = nodes.MulNode()
node2 = nodes.PlusNode()
node3 = nodes.MinusNode()
node4 = nodes.SquareNode()

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
axes[0].scatter(x_data, y_data)
cmap = cm.get_cmap('rainbow', lut=len(x_data * epochs))

for epoch in range(epochs):
    for idx, (x, y) in enumerate(zip(x_data, y_data)):
        z1 = node1.forward(th1, x)
        z2 = node2.forward(z1, th0)
        z3 = node3.forward(y, z2)
        loss = node4.forward(z3)
        
        dz3 = node4.backward(1)
        dy, dz2 = node3.backward(dz3)
        dth0, dz1 = node2.backward(dz2)
        dth1, dx = node1.backward(dz1)
        
        th1 = th1 - (dth1 * lr)
        th0 = th0 - (dth0 * lr)
        
        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(loss)
        
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_range = th1 * x_range + th0
        axes[0].plot(x_range, y_range, color=cmap(epoch*n_point + idx), alpha=0.2)

axes[1].plot(th1_list)        
axes[1].plot(th0_list)
axes[2].plot(loss_list)
axes[0].grid()
axes[1].grid(axis='y')
axes[2].grid(axis='y')

# %% SVLR for one Sample (Batch gradient descent using cost function)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes


def create_dataset(n_point, t_th1, t_th0, noise):
    x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
    y_data = t_th1 * x_data + t_th0 + \
        (noise * np.random.normal(0, 1, size=(n_point, )))
    return x_data, y_data


n_point = 80
t_th1, t_th0 = 3, 2
lr = 0.01
noise = 0.3
epochs = 500

x_data, y_data = create_dataset(n_point, t_th1, t_th0, noise)

th1, th0 = np.random.normal(loc=0, scale=1, size=(2, ))
th1_list, th0_list, cost_list = [th1], [th0], list()

node1 = nodes.MulNode()
node2 = nodes.PlusNode()
node3 = nodes.MinusNode()
node4 = nodes.SquareNode()
node5 = nodes.MeanNode()

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
axes[0].scatter(x_data, y_data)
cmap = cm.get_cmap('rainbow', lut=epochs)

for epoch in range(epochs):
    X, Y = x_data, y_data
    
    Z1 = node1.forward(th1, X)
    Z2 = node2.forward(th0, Z1)
    Z3 = node3.forward(Y, Z2)
    loss = node4.forward(Z3)
    cost = node5.forward(loss)
    
    dL = node5.backward(1)
    dZ3 = node4.backward(dL)
    _, dZ2 = node3.backward(dZ3)
    dTh0, dZ1 = node2.backward(dZ2)
    dTh1, _ = node1.backward(dZ1)
    
    th1 = th1 - (np.sum(dTh1) * lr)
    th0 = th0 - (np.sum(dTh0) * lr)
    
    th1_list.append(th1)
    th0_list.append(th0)
    cost_list.append(cost)
        
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_range = th1 * x_range + th0
    axes[0].plot(x_range, y_range, color=cmap(epoch), alpha=0.2)

axes[1].plot(th1_list)
axes[1].plot(th0_list)
axes[2].plot(cost_list)
axes[0].grid()
axes[1].grid(axis='y')
axes[2].grid(axis='y')

# %% LeNet-5와 비슷하게 MNIST Classification

# 현재는 base 가상환경은 tensorflow가 없고
# iml 가상환경은 spyder가 설치가 되어있지 않아
# tensorflow import에서 문제가 생긴다.
# 둘중 하나를 하면 해결될
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class MNIST_Classifier(Model):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3),
                             activation='relu')
        self.conv1_pool = AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2))
    
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3),
                            activation='relu')
        self.conv2_pool = AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2))
        
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=32, activation='relu')
        self.dense3 = Dense(units=10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

train_ds, ds_info = tfds.load(name='mnist',
                              shuffle_files=True,
                              as_supervised=True,
                              with_info=True,
                              split='train',
                              batch_size=32)

model = MNIST_Classifier()
loss_object = SparseCategoricalCrossentropy()

optimizer = SGD(learning_rate=0.01)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

epochs = 5
losses, accs = list(), list()

for epoch in range(epochs):
    for images, labels in train_ds:
        images = tf.cast(images, tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(labels, predictions)
        
        losses.append(train_loss.result().numpy())
        accs.append(train_acc.result().numpy())
    
    train_loss.reset_states()
    train_acc.reset_states()
    
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

axes[0].plot(losses)
axes[1].plot(accs)

        
























