#%% matplotlib의 객체지향 사용방식

import matplotlib.pyplot as plt

fig, axes = plt.subplots()  # fig와 axes의 객체로 unpacking

#%% scikit-learn의 train_test_split

import numpy as np

from sklearn.model_selection import train_test_split

a = np.arange(24).reshape(4, 6)

x_train, x_test = train_test_split(a)
print(x_train, x_test, sep='\n')
print()

# shuffle
x_train, x_test = train_test_split(a, shuffle=True)
print(x_train, x_test, sep='\n')

# 2개를 나누면 보통 x와 y의 train, test로 나눔
# x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=.5)

# stratify : b의 비율에 맞춰서 a를 쪼갬
# x_train, x_test, y_train, y_test = train_test_split(a, b, stratify=b)

#%% numpy 차원확장

import numpy as np

# a는 2차원 데이터
a = np.arange(24).reshape(4, 6)

# np.expand_dims을 사용
a1 = np.expand_dims(a, 0)   # 1 x 4 x 6 으로 차원 확장
print(a1.shape)

a2 = np.expand_dims(a, 1)   # 4 x 1 x 6 으로 차원 확장
print(a2.shape)

a3 = np.expand_dims(a, 2)   # 4 x 6 x 1 으로 차원 확장
print(a3.shape)

# np.newaxis를 사용
a4 = a[:, :, np.newaxis]    # 4 x 6 x 1 으로 차원확장
print(a4.shape)

a5 = a[np.newaxis, ...]     # 1 x 4 x 6 으로 차원확장
print(a5.shape)

#%% numpy 차원 이동

import numpy as np

a = np.arange(24).reshape(4, 6)

# axis 0과 axis 1을 서로 교환
a1 = np.swapaxes(a, 0, 1)
print(a1)

# matrix transpose
a2 = np.transpose(a)
print(a2)

#%% numpy.tile

a = np.arange(6).reshape(2,3)

# numpy.tile(x, (y, z)) : x를 tile로 써서 y x z만큼 채움
a1 = np.tile(a, (2,4))
print(a1)

#%% Einsum
'''
Einsum 표기법
- Domain Specific Language를 이용해, Tensor연산을 표기하는 방법
- 정규표현식(Regular Expression)과 비슷함
- 간결하며, 쉬운 표기법으로 중간 연산없이 계산이 가능함(최적화된 속도)
- matrix처럼, tensor처럼 생각하자
    ex1) dst = np.einsum("ㅁㅁㅁ, ㅁㅁㅁ, ㅁㅁㅁ -> ㅁㅁㅁ", arg1, arg2, arg3)
                          arg1    arg2    arg3     dst
    ex2) dst = np.einsum("ii", x) : x가 i x j 일때 (i, i) 모두 합함(대각 원소의 총합)
    ex3) dst = np.einsum("ij->", x) : x가 i x j 일때 0차원(scalar)으로 만듦(모든 원소를 합함)
    ex4) dst = np.eimsum("ij->i", x) : x의 (i, j)를 i의 차원을 남기고 j차원을 합침
                                        (행을 모두 합해서 dst로 return)
    ex5) dst = np.eimsum("ij->ji", x) : x를 Transpose해서 dst로 return
'''

import numpy as np

a = np.arange(6).reshape(2,3)

dst = np.einsum("ij -> i", a)
print(dst)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# ,는 행렬간 곱을 의미
dst = np.einsum("i, i -> ", a, b) # 같은 자리값을 곱해서 scalar로(모두 합함)
print(dst)

dst = np.einsum("i, j -> ", a, b) # i의 각 원소들을 j와 곱해서 scalar로(모두 합함)
print(dst)

dst = np.einsum("i, j -> i", a, b) # i의 차원의 vector로 결과를 남김
print(dst)

a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)

dst = np.einsum("ij, ij -> ij", a, b) # (i,j) x (i,j)를 수행하여 i x j 차원의 vector로
print(a)
print(b)
print(dst)