# %% list slicing

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(a)

# a[x:y] : 리스트 a의 x부터 y앞까지 추출
b = a[1:5]
print(b)

# a[x:y:z] : 리스트 a의 x부터 y앞까지 z간격으로 추출
c = a[1:7:2]
print(c)

# %% Numpy Indexing

import numpy as np

a = np.random.uniform(0, 10, size=(3, 3)).astype(np.int)
print(a)

# Numpy는 아래와 같은 방식의 Indexing이 가능하다
print(a[0, 0], a[0, 1], a[0, 2])

for row_idx in range(a.shape[0]):
    for col_idx in range(a.shape[1]):
        print(a[row_idx, col_idx])

# %% Numpy slicing

import numpy as np

a = np.random.uniform(0, 10, size=(5, 5)).astype(np.int)
print(a)

# 위의 방식과 합쳐져서 Numpy는 아래와 같은 slicing이 가능하다
print(a[0, 0:3])
print(a[1, 0:4])
print(a[0:2, 0:2])
print(a[0, :])
print(a[:, 0])

# Reverse는 아래와 같은 방식으로 실행
b = a.reshape(-1, )
b = b[::-1]
b = b.reshape(5, 5)
print(b)

# %% 2D Convolution - 실제 image에 filter 씌우기

import matplotlib.pyplot as plt
import numpy as np

# image를 gray로 만드는 함수
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# 이미지 불러오기
dir_path = "C:\\Users\\OhSeungHwan\\.spyder-py3\\ict_cog_academy_project\\Week_02"
file_name = "\\test_image.jpg"

img = plt.imread(dir_path + file_name)
img_gray = rgb2gray(img)

# fig, ax = plt.subplots(figsize=(12, 12))
# ax.imshow(img_gray, 'gray')
# ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
# fig.tight_layout()

print(img_gray.shape)

# 스무디 필터 만들기
blur = 31
filter_ = np.ones(shape=(blur, blur)) / (blur**2)
l_filter = filter_.shape[0]

# 라플라시안 필터 만들기
'''
filter_ = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
l_filter = filter_.shape[0]
'''

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H - l_filter+1, W - l_filter+1))

# convolution 연산
'''
for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):    
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution
'''        

# 얼굴만 스무디
blur = 31
filter_ = np.ones(shape=(blur, blur)) / (blur**2)
l_filter = filter_.shape[0]

img_convolved = img_gray.copy()
for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        if row_idx > 50 and row_idx < 500 and col_idx > 250 and col_idx < 600:
            img_segment = img_gray[row_idx:row_idx+l_filter,
                                   col_idx:col_idx+l_filter]
            convolution = np.sum(img_segment * filter_)
            img_convolved[row_idx, col_idx] = convolution

fig, axes = plt.subplots(1, 2, figsize=(20, 12))

axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')
