#%% VGGnet

import tensorflow as tf

# include_top : Dense layer까지 같이 가져올 것인가
# weights : 학습된 모델을 가져올 것인가, 구조만 가져올 것인가
VGG16 = tf.keras.applications.VGG16()
print(VGG16.summary())

# 콘솔창에 입력
# from tensorflow.keras.utils import plot_model
# plot_model(VGG16, show_shapes=True)

#%%

# VGGnet의 layer 1 ~ layer 5 가져오기
model1 = tf.keras.models.Sequential(VGG16.layers[1:5])

# VGGnet의 layer 12 추가
model1.add(VGG16.layers[12])

#%% Image Classification

'''
이미지 저장방식
    원본일 경우
    - 디렉토리별
    - 이미지 DB (DB에 이미지 자체를 / 참조 포인트)
    - pandas로 구축 (참조 포인트)
'''

# Image load
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

#%% EDA

for i in data_dir.iterdir():
    print(i)


data_dir.glob('*/*.jpg')    # generator
# print(len(list(data_dir.glob('*/*.jpg'))))            # 총 3670개
# print(len(list(data_dir.glob('daisy/*.jpg'))))        # 633개
# print(len(list(data_dir.glob('dandelion/*.jpg'))))    # 898개
# print(len(list(data_dir.glob('roses/*.jpg'))))        # 641개
# print(len(list(data_dir.glob('sunflowers/*.jpg'))))   # 699개
# print(len(list(data_dir.glob('tulips/*.jpg'))))       # 799개

# for i in data_dir.iterdir():
#     print(str(i).split('\\')[-1])

# class 이름 저장    
class_ = [str(i).split('\\')[-1] for i in data_dir.iterdir() if str(i).split('\\')[-1] != 'LICENSE.txt']

# class별 image 숫자 세기
for i in class_:
    print(f'{i}')
    print(len(list(data_dir.glob(f'{i}/*.jpg'))))

import PIL

# image height, width 측정
daisys = list(data_dir.glob('daisy/*.jpg'))
widths = []
heights = []
for i in data_dir.glob('daisy/*.jpg'):
    im = PIL.Image.open(i)
    widths.append(im.width)
    heights.append(im.height)
    
'''
이미지 크기 통일 필요 !!!
    전략
    - resize(자동) : 이미지 사이즈를 변경 / 이미지가 왜곡됨
    - crop(반자동/수동) : 이미지의 필요한 부분만 잘라서 추출 / object detection 필요

Data Augumentation 필요 !!!
    - CNN은 translation엔 invariance
    - 그러나 rotate나 scale에는 invariance 하지않음
    - 모델의 성능을 향상시키기 위해 Data Augumentation이 필요함
'''

#%% Holdout(train / test Split)

# print(dir(tf.keras.preprocessing))

# directory단위의 image dataset을 처리하는데 효과적
# 자동적으로 image들을 resize 함
# tensor로 data구조를 만들어줌
ims = tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                          subset='training',
                                                          validation_split=(0.2),
                                                          shuffle=True,
                                                          seed=41)
# image 하나만 확인
for i in ims:
    print(i)
    break