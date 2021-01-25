# %% Linear Regression (using tensorflow & keras)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

epochs = 100
learning_rate = 0.01
batch_size = 32
noise_factor = 0.2

train_losses = list()

x_data = np.random.normal(loc=0, scale=1, size=(100, 1))
y_data = x_data + 2 + noise_factor*np.random.normal(loc=0, scale=1, size=(100, 1))

train_x = tf.data.Dataset.from_tensor_slices(x_data)
train_y = tf.data.Dataset.from_tensor_slices(y_data)

train_x = train_x.batch(batch_size)
train_y = train_y.batch(batch_size)

model = Sequential()
model.add(Dense(units=1, activation='linear'))

loss_object = MeanSquaredError()
optimizer = SGD()
train_loss = Mean()

for epoch in range(epochs):
    for x, y in zip(train_x, train_y):
        
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
    
    train_losses.append(train_loss.result().numpy())
    train_loss.reset_states()
    
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(train_losses)
ax.set_xlabel('Epoch', fontsize=30)
ax.set_ylabel('MSE', fontsize=30)

# %% Logistic Regression (using tensorflow & keras)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

epochs = 100
learning_rate = 0.01
batch_size = 32

train_losses, train_accs = list(), list()

x_data = np.random.normal(loc=0, scale=1, size=(100, 1))
y_data = (x_data > 0.5).astype(np.int).flatten()

train_x = tf.data.Dataset.from_tensor_slices(x_data)
train_y = tf.data.Dataset.from_tensor_slices(y_data)

train_x = train_x.batch(batch_size)
train_y = train_y.batch(batch_size)

model = Sequential()
model.add(Dense(units=2, activation='softmax'))

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=learning_rate)
train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

for epoch in range(epochs):
    for x, y in zip(train_x, train_y):
        
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(y, predictions)
    
    train_losses.append(train_loss.result().numpy())
    train_accs.append(train_acc.result().numpy())
    
    print(colored('Epoch: ', 'red', 'on_white', attrs=['blink']), epoch)
    template = "Loss: {:.4f}, Accuracy: {:.2f}%"
    print(template.format(train_loss.result(), train_acc.result()))
    
    train_loss.reset_states()
    train_acc.reset_states()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].plot(train_losses)
axes[0].set_ylabel('Cross Entropy', fontsize=15)

axes[1].plot(train_accs)
axes[1].set_ylabel('Accuracy', fontsize=15)
axes[1].set_xlabel('Epoch', fontsize=30)