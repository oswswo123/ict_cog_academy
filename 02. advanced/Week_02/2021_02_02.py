#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.plot([1, 2, 3], color='pink')


#%%
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# ix_ : vector를 broadcasting 하기 좋게 바꿈
# meshgrid와 broadcasting을 생각하면 유사함
a1, b1 = np.ix_(a, b)     # unpacking
a2, b2 = np.meshgrid(a, b)

# 그 외의 유사한 함수
a3, b3 = np.ogrid[10:40:10, 1:4]
a4, b4 = np.mgrid[10:40:10, 1:4]

# 해당 함수들의 응용사용
x, y = np.ogrid[0:100, 0:100]

# 2x - y만 True가 되는 mask 생성
mask = 2*x - y == 0
arr = np.zeros((100, 100))

# mask가 True인 지점은 255로 set
arr[mask] = 255
plt.imshow(arr, cmap='gray')

#%% numpy operator @

# numpy에서 @는 행렬곱 연산자로 사용됨
# numpy에서 연산자 오버로딩을 통해 정의됨

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[2, 0],
              [0, 2]])

print(a@b)

#%% method방식과 function방식

a = np.arange(100).reshape(25, 4)

np.mean(a)  # 함수 방식
a.mean()    # method 방식 (method는 첫번째 parameter로 self가 자동으로 전달됨)

# dataset split
a, b = np.hsplit(a, (3, ))  # matrix a를 (3, )이므로 3을 기준으로 쪼갬
                            # (x, y, z)라면 x, y, z를 기준으로 4개로 나눔
print(a, b, sep='\n')

# dataset stack
a = np.hstack([a, b])
print(a)