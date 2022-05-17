#%%

from sklearn.datasets import load_boston  # regression
from sklearn.datasets import load_iris    # classification

boston = load_boston()
iris = load_iris()

import tensorflow as tf

# model에 activation function을 넣는 방법은 3가지
# - 문자열
# - 객체
# - 함수
from functools import partial
relu2 = partial(tf.keras.activations.relu, alpha=1)

model_boston = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=(13, ), activation='relu'),   # 문자열
    tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),    # 객체
    tf.keras.layers.Dense(16, activation=relu2),                        # 함수
    # 2015년 이후 BN이 중요해지면서 layer처럼 activation을 넣는 경우도 있음
    # tf.keras.layers.Activation(activation='relu') 
    tf.keras.layers.Dense(1)
    ])

# model을 compile (tensorflow 2.0부터는 안시켜도 됨)
model_boston.compile(loss=tf.keras.losses.mse, optimizer='adam')

model_boston.fit(boston.data, boston.target, epochs=5)

#%%

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train_ohe = tf.keras.utils.to_categorical(y_train)     # y를 one hot encoding으로 변경

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, input_shape=(784, ), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

# loss로 sparse categorical crossentropy를 사용하면 one hot encoding 하지 않고도 적용 가능
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, y_train_ohe, epochs=5)

model.predict(X_test)