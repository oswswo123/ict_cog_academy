#%% CNN modeling

import matplotlib.pyplot as plt
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# convolution 연산을 하기위해 data를 3차원으로 변형
X_train_3d = X_train.reshape(60000, 28, 28, 1)

inputs = tf.keras.Input(shape=(28, 28, 1))
conv1 = tf.keras.layers.Conv2D(4, (3, 3), use_bias=False)
outputs = conv1(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# parameter는 weight 3x3개 + bias 1개인 filter가 4개이므로 40
model.summary()

exam = model(X_train_3d[0][tf.newaxis])

plt.imshow(conv1.weights[0][...,-1].numpy().reshape(3, 3), cmap='gray')

plt.imshow(exam[0][..., 0], cmap='gray')

#%% 실제로 쓰는 형태

X_train_3d = X_train.reshape(60000, 28, 28, 1)

inputs = tf.keras.Input(shape=(28, 28, 1))
conv1 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=True)(inputs)
conv2 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=True)(conv1)
flatten_layer = tf.keras.layers.Flatten()(conv2)
dense1 = tf.keras.layers.Dense(64, activation='relu')(flatten_layer)
outputs = tf.keras.layers.Dense(10, activation='softmax')(dense1)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)