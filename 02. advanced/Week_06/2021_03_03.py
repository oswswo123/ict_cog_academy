#%% Transfer learning (전이 학습)
'''
    기존에 만들어진 모델을 통해 새로운 Data에 적용시키는 기법
    - ex) 개와 고양이는 비슷하니까 개를 분류하는 모델로 고양이를 분류해보자
'''
import tensorflow as tf

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# data load
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    seed=41,
    image_size=(180, 180),
    batch_size=32
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    seed=41,
    image_size=(180, 180),
    batch_size=32
    )

#%% EDA

# 예시 1
for i, j in train_ds:
    print(i, j)
    break

# 예시 2
for i, j in train_ds.take(2):
    print(i, j)
    
# 예시 3
print(next(iter(train_ds)))

#%% pipeline 생성

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

#%% 기존의 Model load

# include_top : Dense layer까지 가져올 것인가
vgg = tf.keras.applications.VGG16(
    include_top=False,
    input_shape=(180, 180, 3),
    )

# base인 vgg를 frozen 시켜서 training 시키지 않음
vgg.trainable = False

# Global Average Pooling layer
gap = tf.keras.layers.GlobalAveragePooling2D()

# Normalization layer
norm = tf.keras.layers.experimental.preprocessing.Rescaling(1/255)

#%% Model 설계 (Global Average Pooling model)

input_layer = tf.keras.Input(shape=(180, 180, 3))
x = norm(input_layer)
x = vgg(x)
x = gap(x)
output_layer = tf.keras.layers.Dense(5)(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['acc']
    )

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    )

#%% Model 설계 (Flatten-Dense model)

input_layer = tf.keras.Input(shape=(180, 180, 3))
x = norm(input_layer)
x = vgg(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_layer = tf.keras.layers.Dense(5)(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['acc']
    )

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    )

#%% 결과 확인

import pandas as pd
data = pd.DataFrame(history.history)
data.plot.line()

print(history.history['val_acc'])

#%% tensorflow hub를 이용한 transfer learning model

import tensorflow_hub as hub

mobile = hub.KerasLayer(
    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4'
    )


# data load
# mobilenet은 image shape가 (224, 224, 3)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    seed=41,
    image_size=(224, 224),
    batch_size=32
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    seed=41,
    image_size=(224, 224),
    batch_size=32
    )

for i, j in train_ds.take(1):
    pass

# print(mobile(i))

#%% 

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
outputs = mobile(inputs)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

print(model.summary())

'''
정리
    transfer learning : 기존의 모델을 새로운 data에 적용하는 학습 기법
    
    transfer learning을 구현하는 방법
    - application
    - tensorflow_hub
'''