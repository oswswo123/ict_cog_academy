#%%

import tensorflow as tf
import pandas as pd
from scipy import io

temp = list()
for i in range(1,35):
    temp.append(io.loadmat(f'D:/Program Files/다운로드(D드라이브)/Annotations/wild_cat/annotation_{i:04}')['box_coord'][0])

dataset = pd.DataFrame([f'D:/Program Files/다운로드(D드라이브)/101_ObjectCategories/wild_cat/image_{i:04}.jpg' for i in range(1, 35)], columns=['file'])
dataset['target'] = 'wild_cat'
dataset['coord'] = temp

#%%

img_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()

myvgg = tf.keras.applications.VGG16(include_top=False)
myvgg.trainable = False

input_layer = myvgg.input
x = tf.keras.layers.GlobalAveragePooling2D()(myvgg.output)
class_ = tf.keras.layers.Dense(1, activation='sigmoid')(x)
box_ = tf.keras.layers.Dense(4)(x)
output_layer = [class_, box_]

model = tf.keras.models.Model(
    inputs=input_layer,
    outputs=output_layer
    )

#%% multiple loss train

# output이 여러개라면 loss도 여러개일 수 있다
# 이 떄 두 개의 loss를 동시에 줄이는 학습은 굉장히 어렵다
# 때문에 이들에게 weight를 부여한다
losses = {
    'a' : 'binary_categorical_crossentropy',
    'b' : 'mean_squared_error'
    }
losses_weight = {
    'a' : 1.0,
    'b' : 2.0
    }

model.compile(
    loss=losses,
    loss_weight=losses_weight,
    optimizer='adam',
    metrics=['acc', 'mse']
    )

