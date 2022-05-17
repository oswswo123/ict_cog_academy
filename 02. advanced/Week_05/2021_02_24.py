#%% pooling

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_3d = X_train.reshape(-1, 28, 28, 1)
X_test_3d = X_test.reshape(-1, 28, 28, 1)

X_train_3d, X_test_3d = X_train_3d / 255.0, X_test_3d / 255.0

# Sequential 방식이 아닌 Composite Function 방식으로 만들어진 model
input_layer = tf.keras.Input(shape=(28, 28, 1))

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))(input_layer)
activation1 = tf.keras.layers.Activation('relu')(conv1)
pooling1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(activation1)

conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3))(pooling1)
activation2 = tf.keras.layers.Activation('relu')(conv2)
pooling2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(activation2)

conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3))(pooling2)
activation3 = tf.keras.layers.Activation('relu')(conv3)

flatten_layer = tf.keras.layers.Flatten()(activation3)
dense1 = tf.keras.layers.Dense(64, activation='relu')(flatten_layer)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense1)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

print(model.summary())
