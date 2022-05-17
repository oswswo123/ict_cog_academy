#%% 

import Augmentor
import tensorflow as tf

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

#%% Data Augment를 위한 전용 라이브러리 활용

pipeline = Augmentor.Pipeline(data_dir)

pipeline.flip_left_right(0.1)

t = pipeline.keras_generator(32, scaled=False)

#%% high level 기법

img_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rescale=1/255
    )

image_data = img_data_generator.flow_from_directory(data_dir, target_size=(224, 224))

#%% vgg로 transfer learning

vgg = tf.keras.applications.VGG16(include_top=True)
vgg.trainable = False

myvgg = vgg.layers[:-1]

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.models.Sequential(myvgg)(input_layer)
output_layer = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.models.Model(
    inputs=input_layer,
    outputs=output_layer
    )

model.summary()

#%%

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['acc']
    )

model.fit_generator(image_data, epochs=5)