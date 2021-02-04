#%% 배워야 할 Library들
'''
    matlab -> numpy, scipy
    toolbox, toolkit -> scikit-learn, scikit-image
    
    Science, Engineering for DIP
    - numpy, scipy
    - imageio
    
    - scikit-image
    
    Computer Vision
    - opencv (from transitional to deep)
    - imutils
    
    PIL
    = Python Image Library
'''

# pep8에 의하면 모두 대문자인 변수명은 상수처럼 사용한다
from PIL import Image

img = Image.open('IU.png')

print(img.format)
print(img.format_description)
print(img._getexif())

# PIL은 image를 matrix가 아니라 객체로 저장함
# 객체는 matrix와 달리 여러 attribute를 저장할 수 있는 장점이 있음
# ex) image format, extra information ...

#%% imageio

# imageio : numpy를 상속받아서 이미지 처리 기능을 확장시킴

import imageio
import numpy as np

img = imageio.imread('IU.png')

a = np.array([1, 2, 3])

# img의 고유 method 확인
print(set(dir(img)) - set(dir(a)))

#%% Image위에 그림 올리기

from PIL import Image, ImageDraw, ImageFont

# 이미지 불러오기
img = Image.open('IUimage.jpg')

# 그림을 그릴 ImageDraw 인스턴스 생성(Mutable이기에 원본도 수정됨)
draw = ImageDraw.Draw(img)

# 한글을 쓰기 위해 맑은고딕 폰트를 불러옴
f = ImageFont.truetype('malgun.ttf', size=50)

# 네모 그리기
draw.rectangle(((0,0), (100,100)))

# 글씨 쓰기
draw.text((100, 100), text="IU Image")
draw.text((200, 200), text="아이유 사진", font=f)

img.show()

#%% numpy의 함수형 패러다임

import numpy as np

def x(a, b):
    return np.array([a, b])

# composition 기법
a = np.fromfunction(x, (2, 4))
print(a, '\n')

b = np.frompyfunc(oct, 1, 1)(3)
print(b, '\n')

c = np.frompyfunc(oct, 1, 1)([1, 2, 3, 4])
print(c, '\n')

t = np.vectorize(lambda x,y: x + y)
print(t([2], [3]), '\n')

# decorator 방식
@np.vectorize
def tt(x, y):
    return x + y
print(tt([2], [3]), '\n')

# numpy의 ndarray는 mutable이라 함수형 패러다임에 불리함
# 그래서 tensorflow는 immutable data type인 tensor를 사용