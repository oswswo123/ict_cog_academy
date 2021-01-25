# %% MNIST Classification (train + vaildation)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Model sub-classing
class MNIST_Classification(Model):
    def __init__(self):
        super(MNIST_Classification, self).__init__()
        
        self.flatten = Flatten()
        self.dense1 = Dense(units=32, activation='sigmoid')
        self.dense2 = Dense(units=10, activation='softmax')
    
    def __call__(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# hyperparameter를 set
epochs = 20
learning_rate = 0.01
batch_size = 256

# dataset load
(train_x, train_y), (val_x, val_y) = mnist.load_data()

# load된 dataset은 numpy ndarray형태
# dataset을 tensor로 바꿔줌
train_x = tf.data.Dataset.from_tensor_slices(train_x)
train_y = tf.data.Dataset.from_tensor_slices(train_y)
val_x = tf.data.Dataset.from_tensor_slices(val_x)
val_y = tf.data.Dataset.from_tensor_slices(val_y)

train_x = train_x.batch(batch_size)
train_y = train_y.batch(batch_size)
val_x = val_x.batch(batch_size)
val_y = val_y.batch(batch_size)

# model 생성
model = MNIST_Classification()

# loss, optimizer같은 object들 생성
loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=learning_rate)
train_loss = Mean()
train_acc = SparseCategoricalAccuracy()
val_loss = Mean()
val_acc = SparseCategoricalAccuracy()

train_losses, train_accs = list(), list()
val_losses, val_accs = list(), list()

for epoch in range(epochs):
    # train data를 통해 model의 parameter를 update
    for (images, labels) in zip(train_x, train_y):
        # forward
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        
        # backward
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(labels, predictions)
    
    # validation data를 통해 model의 성능을 측정
    # loss와 accuracy만 필요하므로, GradientTape도, gradient도 필요없다
    for (images, labels) in zip(val_x, val_y):
        predictions = model(images)
        loss = loss_object(labels, predictions)
        
        val_loss(loss)
        val_acc(labels, predictions)
    
    train_losses.append(train_loss.result().numpy())
    train_accs.append(train_acc.result().numpy())
    val_losses.append(val_loss.result().numpy())
    val_accs.append(val_acc.result().numpy())
    
    print(colored('Epoch:', 'red', 'on_white', attrs=['blink']), epoch+1)
    termlate = '''
    Train loss: {:.4f}       Train Accuracy: {:.2f}%
    Validation loss: {:.4f}  Validation Accuracy: {:.2f}%
    '''
    print(termlate.format(train_loss.result(),
                          train_acc.result()*100,
                          val_loss.result(),
                          val_acc.result()*100))
    
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()

fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

ax.plot(train_losses, color='tab:blue', label='Train Loss')
ax.plot(val_losses, color='tab:blue', label='Validation Loss', linestyle=':')
ax2.plot(train_accs, color='tab:orange', label='Train Accuracy')
ax2.plot(val_accs, color='tab:orange', label='Validation Accuracy', linestyle=':')
ax.legend(fontsize=15, loc='center right')
ax2.legend(fontsize=15, loc='upper right')

ax.set_ylabel('Cross Entropy', color='tab:blue', fontsize=15)
ax2.set_ylabel('Accuracy', color='tab:orange', fontsize=15)
ax.set_xlabel('Epoch', fontsize=30)

ax.grid(axis='y')
# %% Model을 만드는 방법 1 (Sequential method)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Flatten())
model.add(Dense(units=20, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

# %% Model을 만드는 방법 2 (Model sub-classing)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Model class를 상속받음 (Model sub-classing)
class MNIST_Classifier(Model):
    def __init__(self):
        # 먼저 부모 클래스인 Model대로 초기화를 시킴
        super(MNIST_Classifier, self).__init__()
        
        # 이후 자식 클래스인 MNIST_Classifier의 초기화를 실행
        self.flatten = Flatten()
        self.dense1 = Dense(units=20, actiation='sigmoid')
        self.dense2 = Dense(units=10, activation='softmax')
    
    # 해당 class의 이름이 불릴 때 실행되는 method (special method)
    def __call__(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MNIST_Classifier()

model_output = model(images)
print(model_output.shape)

# %% 시각화하는 또다른 방법

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

ax.plot(train_losses, color='tab:blue', linestyle=':', label='Train Loss')
ax2.plot(train_accs, color='tab:orange', linestyle=':', label='Train Accuracy')
ax.legend(font_size=10, loc='center right')

ax.set_ylabel('Cross Entropy', fontsize=15, color='tab:blue')
ax2.set_ylabel('Accuracy', fontsize=15, color='tab:orange')
ax.set_xlabel('Epoch', fontsize=30)

ax.tick_params(color='tab:blue', axis='y', labelsize=20)
ax2.tick_params(color='tab:orange', axis='y', labelsize=20)

ax.grid(axis='y')