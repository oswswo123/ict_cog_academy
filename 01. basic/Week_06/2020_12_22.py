# %% Logistic Regression 연습

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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
    for (x, y) in zip(train_x, train_y):
        
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(y, predictions)
        
    train_losses.append(train_loss.result().numpy())
    train_accs.append(train_acc.result().numpy())
    
    print(colored('Epoch: ', 'red', 'on_white', attrs=['blink']), epoch+1)
    termlate = "Loss: {:.4f}, Accuracy: {:.2f}%"
    print(termlate.format(train_loss.result(), train_acc.result()))
    
    train_loss.reset_states()
    train_acc.reset_states()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(train_losses)
axes[0].set_ylabel('Cross Entropy', fontsize=15)

axes[1].plot(train_accs)
axes[1].set_ylabel('Accuracy', fontsize=15)
axes[1].set_xlabel('Epoch', fontsize=30)

# %% MNIST Classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

epochs = 100
learning_rate = 0.01
batch_size = 32

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = tf.data.Dataset.from_tensor_slices(train_x)
train_y = tf.data.Dataset.from_tensor_slices(train_y)

train_x = train_x.batch(batch_size)
train_y = train_y.batch(batch_size)

model = Sequential()
model.add(Flatten())
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

train_losses, train_accs = list(), list()
loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=learning_rate)
train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

for epoch in range(epochs):
    for (images, labels) in zip(train_x, train_y):
        
        # tf.GradientTape()은 forward를 진행하면서
        # backward에서 필요할만한 parameter들을 기록함
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(labels, predictions)
        
    train_losses.append(train_loss.result().numpy())
    train_accs.append(train_acc.result().numpy())
    
    print(colored('Epoch: ', 'red', 'on_white', attrs=['blink']), epoch+1)
    termlate = 'Loss: {:.4f}, Accuracy: {:.4f}'
    print(termlate.format(train_loss.result(), train_acc.result()))
    
    train_loss.reset_states()
    train_acc.reset_states()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(train_losses)
axes[0].set_ylabel('Cross Entropy', fontsize=15)

axes[1].plot(train_accs)
axes[1].set_ylabel('Accuracy', fontsize=15)
axes[1].set_xlabel('Epoch', fontsize=30)
