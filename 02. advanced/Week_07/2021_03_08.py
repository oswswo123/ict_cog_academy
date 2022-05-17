#%%

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(train, validation, test), info = tfds.load(
    'tf_flowers',
    # 0% ~ 70% : train,  70% ~ 80% : validation,  80% ~ 100% : test
    split=['train[:70%]','train[70%:80%]','train[80%:]'],
    as_supervised=True,
    with_info=True,
    )

#%% datasets examples 보기

tfds.show_examples(train, info)

#%% tensorflow dataset을 dataframe으로 만듦

df = tfds.as_dataframe(train, info)

#%% label count

print(df.label.value_counts())

#%% label name print

print(info.features['label'].names)

#%% 하나씩 뽑아 보기

print(next(iter(train)))

#%% data augmentation을 해 줄 object 생성

rf = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical')
rr = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.2, 0.3))

#%% 원래 image

for i,j in train.take(1):
    pass

plt.imshow(i)

#%% rf로 변형한 image

plt.imshow(rf(i[tf.newaxis]).numpy()[0])

#%% rr로 변형한 image

plt.imshow(rr(i[tf.newaxis]).numpy()[0])

'''
    Data Augmentation
    Theory:
        1. image transformation
        2. deep learning (GAN, VAE, ...)
    
    A에 대한 Tensorflow 구현
        1. tf.keras.preprocessing.image.ImageDataGenerator
        2. tf.keras.layers.experimental.preprocessing ...
            1. 모델의 일부로 사용
            2. map
                2-1. 
                2-2.
    
    A에 대한 전용 라이브러리 활용
'''

#%% Model의 일부로 Data Augment를 사용

# (None, None, 3)으로 이미지 크기 상관없이 input을 받음
input_layer = tf.keras.layers.Input(shape=(None, None, 3))
x = tf.keras.layers.experimental.preprocessing.RandomZoom((0.2, 0.3))(input_layer)
x = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.2, 0.3))(x)
output_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)

model = tf.keras.models.Model(
    inputs=input_layer,
    outputs=output_layer
    )

model(i[tf.newaxis])

model.compile(
    loss='sparse_categorical_crossentropy'
    )

# model.fit의 호출 중에만 Data Augmentation이 실행됨(model.predict 등에서는 실행 x)
# 중간중간 확인이 불가능

#%% map을 사용

