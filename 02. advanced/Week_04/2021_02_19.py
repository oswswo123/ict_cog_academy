#%% data load (오래 걸려서 분리)

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

#%% EDA

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5 ,5, i+1)
    plt.imshow(X_train[i])

# label을 분석
import numpy as np

print(np.unique(y_train, return_counts=True))
# inbalanced data 확인
# stratification : 층화계층

#%% Normalize

X_train_norm, X_test_norm = X_train / 255, X_test / 255

#%% model implement

import tensorflow as tf

input_layer = tf.keras.Input(shape=X_train_norm.shape[1:])
x1 = tf.keras.layers.Flatten()(input_layer)
x2 = tf.keras.layers.Dense(128, activation='relu')(x1)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x2)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

print(model.summary())
# print(model(X_train_norm)) 잘 됨

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

output = model.fit(X_train_norm, y_train,
                   epochs=2,
                   batch_size=100,
                   callbacks=tf.keras.callbacks.History())

# history는 epochs의 갯수만큼 나온다
print(output.history)

# 여기까지 복습

#%% Callback 만들기

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('----------------')
        print('epoch {} start'.format(epoch+1))
        print('----------------')
    
    def on_epoch_end(self, epoch, logs=None):
        print('\n----- epoch {} end -----'.format(epoch+1))
        
output = model.fit(X_train_norm, y_train,
                   epochs=2,
                   batch_size=100,
                   callbacks=[MyCallback()])

# Callback을 overriding 해서 여러 기능을 custom 가능
# ex) 매 epoch 종료시 email 발신, batch 구간마다 text print

#%% Tensorboard Callback

output = model.fit(X_train_norm, y_train,
                   epochs=2,
                   batch_size=100,
                   callbacks=[tf.keras.callbacks.TensorBoard()])

# 아래 명령어를 console에 입력하여 tensorboard load
# Tensorboard 사이트를 통해 결과 분석도 가능
# %load_ext tensorboard
# %tensorboard --logdir=./logs/fit
# localhost:6006
