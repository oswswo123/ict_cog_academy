#%%

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# tf.data.Dataset은 추상클래스
a = tf.data.Dataset.from_tensors(X_train)
b = tf.data.Dataset.from_tensor_slices(X_train)

# a와 b의 attritute는 같다!
print( set(dir(b)) - set(dir(a)) )

#%% Image classification

'''
    기본적인 ML Workflow
    1. 데이터 검사 및 이해하기 (EDA)
    2. input pipeline build
    3. model build
    4. model train
    5. model test
    6. model 개선하고 프로세스들 반복
'''
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

# 내부적으로 dataset을 최적화시킴
# 하나씩 보려면 break 테크닉을 사용하거나, map 형식으로 바꾸어서 next를 통해 볼 수 있음
# 혹은 take를 사용해 무작위로 하나를 뽑을 수도 있음
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )

for i in train_ds.take(1):
    print(type(i))
    break

#%%

# dataset의 pipeline 만들어서 system resource 효율성 증대
# Dataset.cache는 첫 epoch 동안 디스크에서 이미지를 로드한 후 이미지를 메모리에 유지함
# 이렇게 하면 모델을 훈련하는 동안 데이터세트가 병목 상태가 되지 않음
# 데이터세트가 너무 커서 메모리에 맞지 않는 경우,
# 이 메서드를 사용하여 성능이 높은 온디스크 캐시를 생성할 수도 있음
# Dataset.prefetch는 GPU/CPU가 알아서 최적화 시켜주는 함수
#  훈련 중에 데이터 전처리 및 모델 실행 겹침
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%%

input_layer = tf.keras.layers.Input(shape=(180, 180, 3))
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2))(input_layer)
# x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2))(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2))(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_layer = tf.keras.layers.Dense(5)(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

print(model.summary())

#%%

# 0.1 + 0.1 + 0.1은 0.3가 아니라 0.30000000000000004.
# 이는 속도를 위해 numerical stability를 희생시킨 부분
# numerical stability때문에 output에서 softmax를 쓰지않고,
# loss에서 from_logits을 true로 함
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['acc']
    )

history = model.fit(
    train_ds,
    epochs=5,
    shuffle=True,
    )

'''
    Overfitting을 해곃하는 방법
    - 데이터를 더 많이! (Data Augmentation)
    - drop out / regularization
    - early stopping 등
'''