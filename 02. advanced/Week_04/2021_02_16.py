#%% Sequential 방식으로 model 만들기

import tensorflow as tf
import numpy as np

# 함수형 패러다임 방식 (composite functions)
# 3-layers model 혹은 2-hidden layers model
model = tf.keras.models.Sequential([
    # tf.keras.Input((4, )),      # 이렇게도 input layer를 만들 수 있음
    tf.keras.layers.Dense(16, input_shape=(4, )),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(3),       # output layer
    ])

# 객체형 방식 (가능은 하나 추천은 하지 않음)
# model = tf.keras.models.Sequential()
# layer1 = tf.keras.layers.Dense(16, input_shape=(4, ))
# layer2 = tf.keras.layers.Dense(16)
# layer3 = tf.keras.layers.Dense(3)
# model.add(layer1)
# model.add(layer2)

# 모델 구조 요약
# parameter은 weight와 bias의 수의 합
model.summary()

# numpy array도 해당 model에 집어넣을 수 있긴 함
a = np.arange(20).reshape(5, 4)
print(model(a))

#%% sci-kit learn의 iris data를 tensorflow로

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris

data = load_iris()

model = Sequential([
    Dense(16, input_shape=(4, )),
    Dense(16),
    Dense(3, activation='softmax')
    ])

print(model(data.data))