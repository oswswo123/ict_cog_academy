#%% opencv

import cv2

print(cv2.__version__)

# cv2.imread(filename, flag=None, /)
# flag는 옵션
img = cv2.imread('IUimage.jpg')

# imread의 flag에 들어갈 수 있는 상수 확인
# 상수의 변수명은 모두 대문자로 작성됨
imread_option = [[i, getattr(cv2, i)] for i in dir(cv2) if 'IMREAD' in i]
#for i in dir(cv2):
#    if 'IMREAD' in i:
#        imread_option.append([i, getattr(cv2, i)])

# opencv에서 BGR채널을 분리
b, g, r = cv2.split(img)

# 분리된 BGR채널을 합침
merge_img = cv2.merge((b, g, r))

# 색 변경(Convert Color)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt

# matplotlib으로 imshow는 가능
# opencv는 BGR순으로, matplotlib은 RGB순으로 읽기 때문에
# 출력 결과가 달라짐
plt.imshow(img)

# numpy에서 array의 순서를 바꾸는 방법은 x[::-1]
# BGR을 RGB로 변경
plt.imshow(img[:, :, ::-1])

# 유의사항!
# JPG같은 압축이미지들은 이미지를 불러들이는 과정에서 손실된 정보를 복원시킴
# Framework별로 이미지를 불러오는 방식에 차이가 있으므로
# 이미지의 data가 다를 수 있다!

#%%

# PIL을 사용해 RGB채널 분리
from PIL import Image

img_pil = Image.open('IUimage.jpg')
r, g, b = img_pil.split()

Image._show(r)

# 분리된 RGB채널을 합침
merge_img = Image.merge('RGB', (r, g, b))
Image._show(merge_img)
