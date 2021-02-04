#%% DIP (Digital Image Processing)

# Image Accqurisition
import matplotlib.pyplot as plt

img = plt.imread('IUimage.jpg')     # 3요소 : (경로), 파일명, 확장자
plt.imshow(img)

# 얼굴만 slicing
img2 = img[:650, 100:600, :]
plt.imshow(img2)

# 좌우반전
img3 = img[:, ::-1, :]
plt.imshow(img3)

#%% print 유무의 차이

class X:
    # 그냥 사용시 실행
    def __repr__(self):
        return '1'
    
    # print에서 실행
    def __str__(self):
        return '2'

x = X()
# 콘솔창에 x 와 print(x) 입력해서 차이 찾아보기
# print로 같은 결과를 보고싶다면 from pprint import pprint를 사용하자

#%%
'''
    행렬로 표현된 이미지를 어떻게 볼까?
    - opencv
    - matplotlib (이번엔 이걸 사용)
    
    matplotlib - state machine 방식으로 사용 (절차지향과 유)
               - matplotlib은 변수의 개념이 없다
               - 결과를 빨리 확인하는데 강점을 가짐
'''

import matplotlib.pyplot as plt
import tensorflow as tf

s = tf.keras.datasets.mnist.load_data()[0][0][0]

# matplotlib은 기본적으로 RGB channel을 가정함
# 흑백데이터일 경우 cmap을 설정해줘야함
plt.matshow(s)        # 흑백인 mnist는 이렇게 사용하면 색이 이상하게 변함
plt.matshow(s, cmap='gray')
