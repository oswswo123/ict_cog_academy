#%%

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# X_train, X_test = X_train / 255.0, X_test / 255.0
# X_train = X_train.reshape(-1, 32, 32, 1)
# X_test = X_test.reshape(-1, 32, 32, 1)

input_layer = tf.keras.layers.Input(shape=(32, 32, 1))
x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='valid')(input_layer)
x = tf.keras.layers.Activation(activation='tanh')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='valid')(x)
x = tf.keras.layers.Activation(activation='tanh')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation='tanh')(x)
x = tf.keras.layers.Dense(64, activation='tanh')(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

print(model.summary())
