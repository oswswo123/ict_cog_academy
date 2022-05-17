#%% Autoencoder (Dense)

import tensorflow as tf

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train, X_test = X_train.reshape(60000, -1), X_test.reshape(10000, -1)
X_train, X_test = X_train / 255, X_test / 255

representation = 64

input_layer = tf.keras.Input(shape=(28*28, ))
encoder = tf.keras.layers.Dense(512, activation='relu')(input_layer)
encoder = tf.keras.layers.Dense(256, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(128, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(representation, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(128, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(256, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(512, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(28*28, activation='sigmoid')(decoder)

autoencoder = tf.keras.models.Model(
    inputs=input_layer,
    outputs=decoder
    )

#%%

autoencoder.compile(
    loss='binary_crossentropy',
    optimizer='adam'
    )

autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=512
    )

#%%

prediction = autoencoder(X_test)

import matplotlib.pyplot as plt

plt.imshow(prediction[:10][0].numpay().reshape(28, 28))

'''
    Flatten 하는 과정에서 상당히 많은 정보가 손실됨
    Dense는 AE에 그다지 적합하지 않음을 확인할 수 있음
'''

#%% Autoencoder (Convolution)

import tensorflow as tf

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train, X_test = X_train.reshape(-1, 28, 28, 1) / 255, X_test.reshape(-1, 28, 28, 1) / 255

input_layers = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(input_layers)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(encoder)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
decoder = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x)

model = tf.keras.models.Model(
    inputs=input_layers,
    outputs=decoder
    )

#%%

model.compile(
    loss='mse',
    optimizer='adam'
    )

model.fit(X_train, X_train, epochs=50, batch_size=512)

#%%

prediction = model.predict(X_test)

plt.imshow(prediction[0].reshape(28, 28))

'''
    굉장히 유사하게 새로운 이미지가 생성됨
    단, model이 너무 Deep해지면 지나치게 비슷한 이미지가 생성됨을 주의
'''