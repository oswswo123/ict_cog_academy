#%%

import tensorflow as tf
import tensorflow_probability as tfp

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train, X_test = X_train.reshape(-1, 28, 28, 1) / 255, X_test.reshape(-1, 28, 28, 1) / 255
# 흑백 이미지 만드는 방법 : np.where(X_train > 0.5, 1.0, 0.0)

#%%

class CVAE(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super().__init__(self)
        self.latent_dim = latent_dim
        
        # encoder
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim+latent_dim)
            ])
        
        # decoder
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7*7*32, input_shape=(latent_dim, ), activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same')
            ])
    
    def call(self):
        pass

#%%
'''
    1. layer custom
    2. callbacks
    3. tf.GradientTape
    4. model custom
    
    이들을 통해 내가 원하는 모델을 만들어내자
'''