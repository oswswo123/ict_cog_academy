#%%

import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_train = (X_train - 127.5) / 127.5

BATCH = 256
LATENT_DIM = 100
train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(BATCH)

class DepthWiseSeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, input_channel, output_channel, kernels_per_layers):
        super().__init__(self)
        self.depthwise = tf.keras.layers.DepthwiseConv2D(input_channel, output_channel, kernels_per_layers)
        self.pointwise = tf.keras.layers.Layer.Conv2D(input_channel, output_channel, kernel_size=(1, 1))
        
    def call(self, x):
        func = self.depthwise(x)
        return self.pointwise(func)

class PointWiseConv2D(tf.keras.layers.Layer):
    def __init__(self, input_channel, output_channel):
        super().__init__(self)
        self.pointwise = tf.keras.layers.Layer.Conv2D(input_channel, output_channel, kernel_size=(1, 1))
        
    def call(self, x):
        return self.pointwise(x)

class GAN(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super().__init__(self)
        
        # Generator Model
        input_layer = tf.keras.layers.Input(shape=(100, ))
        
        x = tf.keras.layers.Dense(units=7*7*32)(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
        
        x = tf.keras.layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        x = tf.keras.layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        x = tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_layer = tf.keras.layers.LeakyReLU()(x)
        
        self.generator = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer)
        
        # Discriminator Model
        input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_layer)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)
        
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)
        
        x = tf.keras.layers.Flatten()(x)
        output_layer = tf.keras.layers.Dense(1)(x)
        
        self.discriminator = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer
            )
        
    def discriminator_loss(real_, fake_):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_), real_)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_), fake_)
        total = real_loss + fake_loss
        return total

    def generator_loss(fake_):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_), fake_)

    def train_step(X_train):
        noises = tf.random_normal([BATCH, LATENT_DIM])
        
        with tf.GradientTape() as tape:
            generated_images = self.generator(noises, training=True)
            
            real_output = self.discriminator(images, train)
            # 보충 필요
            
        pass
    
    def call(self):
        pass



g = GAN(LATENT_DIM)


