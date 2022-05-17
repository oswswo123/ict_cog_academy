#%%

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# composite function 방식으로 만든 model
# g(x) = f₄(f₃(f₂(f₁(x)))) 의 개념
input_layer = tf.keras.layers.Input(shape=(28*28, ))
x = tf.keras.layers.Dense(128)(input_layer)
x = tf.keras.layers.Dense(128)(x)
output_layer = tf.keras.layers.Dense(10)(x)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Sequential 방식으로 만든 model
# 두 모델은 같은 구조이다
model_s = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28*28, )),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(10)
    ])

# model_s와 달리 model은 input layer가 list임을 확인할 수 있다
# 즉, input과 output을 여러개로 할 수 있다
print(model.summary())
print(model_s.summary())

# 콘솔창에 입력해보자
# tf.keras.utils.plot_model(model, show_shapes=True)
# tf.keras.utils.plot_model(model_s, show_shapes=True)

#%% multiple input & output layer model

import tensorflow as tf

# input이 여러개인 model
inputs_1 = tf.keras.Input(shape=(28*28, ))
inputs_2 = tf.keras.Input(shape=(28*28, ))
inputs = tf.keras.layers.Concatenate()([inputs_1, inputs_2])
x = tf.keras.layers.Dense(128)(inputs)
y = tf.keras.layers.Dense(128)(inputs)
z = tf.keras.layers.Concatenate()([x, y])
x = tf.keras.layers.Dense(128)(z)
outputs = tf.keras.layers.Dense(10)(x)

model = tf.keras.models.Model(inputs=[inputs_1, inputs_2], outputs=outputs)

# 콘솔창에 입력해보자
# tf.keras.utils.plot_model(model, show_shapes=True)

#%% fashion mnist classificaion

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalization, Standardization, Regularization는 모두 정규화로 해석됨
# 그러나 실제로는 다른 용어

# Normalization : 최소값을 0, 최대값을 1로 하여 범위를 줄이는 방법 (min-max scaling)
# x' = x - min(x) / max(x) - min(x)
# 학습전에 scaling 해서
# ML에서는 scale이 큰 feature의 영향이 비대해지는 것을 방지
# DL에선 Local Minima에 빠질 위험을 줄이고 학습 속도를 향상시킴
X_train = X_train / 255
X_test = X_test / 255

# Standardization : 평균을 0, 분산을 1로 만드는 방법
# x' = x - μ / σ (μ : 평균, σ : 표준편차)
# 학습전에 scaling 해서
# ML에서는 scale이 큰 feature의 영향이 비대해지는 것을 방지
# DL에선 Local Minima에 빠질 위험을 줄이고 학습 속도를 향상시킴
# 정규분포를 표준정규분포로 변환하는 것과 같음
# -1 ~ 1 사이에 68%, -2 ~ 2 사이에 95%, -3 ~ 3 사이에 99%가 있음
# 표준화로 번역하기도 한다

# Regularization : weight를 조정하는데 규제(제약)를 거는 기법
# overfitting을 막기위해 사용함
# L1 regularization(LASSO, 마름모), L2 regularization등이 있음(Lidge, 원)

inputs = tf.keras.layers.Input(shape=(28*28, ))
flatten = tf.keras.layers.Flatten()(inputs)
dense1 = tf.keras.layers.Dense(16, activation='relu', name='moon')(flatten)
dense2 = tf.keras.layers.Dense(16, activation='relu', name='sun')(dense1)
outputs = tf.keras.layers.Dense(10, activation='softmax')(dense2)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='test')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

# batch_size는 정답과 비교하기 전에 한번에 문제 푸는 문제의 수
# 학습은 모든 data에 대해서 진행됨
# batch_size가 1000이면 1000개의 data를 풀고 정답을 한번 확인함
# batch_size가 1이면 매 data마다 정답을 확인함
# batch_size가 클수록 train속도는 빨라지나 학습의 정확도가 낮아짐
out = model.fit(X_train, y_train, epochs=20, batch_size=100)

print(out.history)

import pandas as pd

d = pd.DataFrame(out.history)
d.plot()

# 만들어진 model은 function으로 encapsulation 한다
# - 관리 편의성 / 재활용
# - estimator
# - wrapper