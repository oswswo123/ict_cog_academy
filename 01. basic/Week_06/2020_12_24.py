# %% MNIST Classification - LeNet (using CNN)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Model sub-classing
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=(2, 2))
        self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), activation='tanh')
        self.maxpooling1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')
        self.maxpooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv3 = Conv2D(filters=120, kernel_size=(5, 5), activation='tanh')
        
        # classification
        self.flatten = Flatten()
        self.dense1 = Dense(units=32, activation='sigmoid')
        self.dense2 = Dense(units=10, activation='softmax')
    
    def __call__(self, x):
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        
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

# conv layer는 batch, height, width, depth의 4차원 data가 필요함
# 기존의 mnist data는 batch, height, witdh의 3차원 data
# 그래서 차원을 확장해줌
train_x = np.expand_dims(train_x, axis=3).astype(np.float32)
val_x = np.expand_dims(val_x, axis=3).astype(np.float32)

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
model = LeNet5()

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

# %% Model sub-classing을 Sequential로 묶어내기

# Model sub-classing + Sequential    
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.fe = Sequential()
        self.fe.add(ZeroPadding2D(padding=(2, 2)))
        self.fe.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh'))
        self.fe.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.fe.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
        self.fe.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.fe.add(Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'))
        self.fe.add(Flatten())
        
        self.classifier = Sequential()
        self.classifier.add(Dense(units=20, activation='sigmoid'))
        self.classifier.add(Dense(units=10, activation='softmax'))

    def __call__(self, x):
        x = self.fe(x)
        x = self.classifier(x)
        return x


# %% Custom Layer 만들기 (Layer sub-classing + Model sub-classing)

# 여러 layer를 만들어 낼 때 간단하게 사용가능 (custom layer 생성)
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

class ConvLayer(Layer):
    def __init__(self, filters):
        super(ConvLayer, self).__init__()
        
        self.conv = Conv2D(filters=filters, kernel_size=(5, 5), activation='tanh')
        self.conv_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    
    def __call__(self, x):
        x = self.conv(x)
        x = self.conv_pool(x)
        return x

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # feature extractor
        self.conv1 = ConvLayer(filters=6)
        self.conv2 = ConvLayer(filters=16)
        self.conv3 = Conv2D(filters=120, kernel_size=(5, 5), activation='tanh')
        self.flatten = Flatten()
        
        # classifier
        self.dense1 = Dense(units=20, activation='sigmoid')
        self.dense2 = Dense(units=10, activation='softmax') 
        
    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.dense2(x)
        return x