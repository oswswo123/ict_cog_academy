# %% Numpy
'''
Numpy는 ndarray라는 객체를 다루는 모듈
Numpy는 행렬연산, 벡터연산에 있어서 매우 편리한 기능을 제공한다
ndarray객체는 pandas, scikit-learn등 여러 모듈에서 사용하게 된다
ndarray의 nd는 n-dimensional를 의미함
그래서 shape가 (3, 1)이면 3x1 matrix(tensor), (3, )이면 3차원 vector를 만든다
'''
import numpy as np

a = [[1, 2], [3, 4]]
b = np.array([[1, 2], [3, 4]])
print("python_list :", a, "\npython_list_type :", type(a))
print("ndarray :", b, "\nndarray_type :", type(b))

# %% ndarray 만들기

import numpy as np

# python list -> ndarray
python_list = [1, 2, 3]
ndarray = np.array(python_list)

# np.zeros()
ndarray2 = np.zeros(shape=(10, ))
print(ndarray2)

# np.ones()
ndarray3 = np.ones(shape=(10, ))
print(ndarray3)

# np.full()
ndarray4 = np.full(shape=(10, ), fill_value=3.14)
print(ndarray4)

# np.full() with np.ones()
ndarray5 = 3.14*np.ones(shape=(10, ))  # 1로 채워진 vector에 스칼라곱을 시행
print(ndarray5)

# np.empty()
# empty로 만들 경우 빈 공간만 잡음
# 그래서 기존 memory buffer에 있는 data가 그대로 들어가게됨
ndarray6 = np.empty(shape=(10, ))
print(ndarray6)

# %% ndarray 만들기 2 - like 사용

import numpy as np

tmp = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
print(tmp)

# like는 parameter로 들어온 ndarray의 shape대로 새로운 ndarray를 만들어줌
ndarray = np.zeros_like(tmp)
print(ndarray)

ndarray1 = np.empty_like(tmp)
print(ndarray1)

# %% Matrix 만들기 - Python list -> ndarray

import numpy as np

python_list = [[1, 2], [3, 4]]
ndarray = np.array(python_list)

print("python_list:", python_list)
print("ndarray: \n", ndarray)

# %% Matrix 만들기 2

import numpy as np

ndarray = np.zeros(shape=(2, 2))
print(ndarray)

ndarray1 = np.empty(shape=(3, 3))
print(ndarray1)

# %% vector vs matrix

import numpy as np

# 이 둘은 절대 같지 않다!!! (하나는 vector, 하나는 matrix(tensor))
ndarray1 = np.ones(shape=(5, ))
ndarray2 = np.ones(shape=(5, 1))

print(ndarray1)
print(ndarray2)

# %% ndarray information

import numpy as np

ndarray = np.full(shape=(2, 2), fill_value=3.14)

print(ndarray)
print("ndarray.shape:", ndarray.shape)       # ndarray의 모양
print("ndarray.dtype:", ndarray.dtype)       # ndarray의 자료형
print("ndarray.size:", ndarray.size)         # ndarray의 element 수
print("ndarray.itemsize:", ndarray.itemsize) # ndarray의 item의 size(단위는 byte)

print("ndarray size:", ndarray.size * ndarray.itemsize)  # ndarray가 차지하는 용량(단위는 byte)

# %% ndarray의 연산

import numpy as np

a = np.array([1, 2, 3])
b = np.array([5, 6, 7])

# 내적 구하기
dot_product = np.sum(a*b)
print(dot_product)

# %% ndarray indexing
'''
data science에서 dataset에 대한 규칙
- row는 sample에 대한 vector
- column은 feature에 대한 vector
'''
import numpy as np

python_list = [[1, 2], [3, 4], [5, 6]]
ndarray = np.array(python_list)
print(ndarray)
print('student 1:', ndarray[0])  # 1번 sample의 data
print('student 2:', ndarray[1])  # 2번 sample의 data
print('student 3:', ndarray[2])  # 3번 sample의 data

# %% 평균 구하기

import numpy as np

n_student = 4
scores = np.random.randint(0, 100, size=(n_student, 3))

# mean 함수 사용! (axis의 0은 열의 평균, axis 1은 행의 평균)
# axis = 2 이후로는 3d 이상의 data에서 사용함
# 그러니까 axis는 첫번째 축, 두번째 축, 세번째 축... 을 정하는 것
# collapsed axis작업임(해당 차원의 data를 없애고 다른 차원의 정보를 남김)
# ex) (50, 3) matrix에서 numpy.mean(axis=0)하면 50이 사라지고 (3, )이 됨
# ex) (50, 3) matrix에서 numpy.mean(axis=1)하면 3이 사라지고 (50, )이 됨
# 단, tensorflow에서는 이 개념이 numpy와 반대

# 각 열을 모두 합해서 평균을 출력
# 학생 차원의 정보를 소멸시키고 과목별 평균 data를 남김
print(np.mean(scores, axis=0)) 

# 각 행을 모두 합해서 평균을 출력
# 과목 차원의 정보를 소멸시키고 학생별 평균 data를 남김
print(np.mean(scores, axis=1))  

# 그 외의 방법
class_sum = np.zeros(shape=(3, ))
for score in scores:
    class_sum += score
class_mean = class_sum / n_student
print(class_mean)

# %% 분산 구하기 (mean 쓰지말래..)

import numpy as np

n_stuednt = 100
scores = np.random.randint(0, 100, size=(n_student, 3))

class_sum = np.zeros(shape=(3, ))
class_squaresum = np.zeros(shape=(3, ))
for score in scores:
    class_sum += score
    class_squaresum += score**2
class_mean = class_sum / n_student
class_var = (class_squaresum / n_student) - (class_mean**2)

# class_var = np.var(scores, axis=0)  # 한방에 구하는 법

print(class_mean)
print(class_var)

# %% MSE (Mean Squared Error) 구하기

import numpy as np

n_point = 100
x = np.random.normal(0, 2, size=(n_point, ))
y = 3*x
predications = 2*x

ms_vector = (y-predications)**2
mse = 0
for ms_item in ms_vector:
    mse += ms_item
mse /= n_point

# mse = np.sum(((y-predications)**2) / n_point)  # 이건 쓰지 말고

print(mse)

# %% reshape(구조 재배열)

import numpy as np

a = np.array([1, 2, 3, 4])
print(a.shape)
print(a, '\n')

a = a.reshape((2, 2))
print(a.shape)
print(a, '\n')

# %% reshape + -1 value

import numpy as np

# reshape method에 parameter로 -1을 제공하면 numpy가 알아서 맞춘다
a = np.random.uniform(0, 20, size=(20, ))
print(a.shape)
a = a.reshape((4, 5))
print(a.shape)
a = a.reshape((4, -1))
print(a.shape)
a = a.reshape((-1, 2))
print(a.shape)

# 이 부분은 엄밀히 말하면 matrix이지만, 수학적으로는 vector로 사용 가능
a = a.reshape((1, -1))  # to row vector
print(a.shape)
a = a.reshape((-1, 1))  # to column vector
print(a.shape)

# %% Broadcasting

import numpy as np

a = np.array([1, 2, 3, 4]).reshape((1, -1))
b = np.array([10, 20, 30]).reshape((-1, 1))
print(a.shape)
print(b.shape)

# 이 수식은 수학적으로는 불가능함 (1 x 4 matrix + 3 x 1 matrix ?)
# 그러나 numpy에서는 알아서 row와 col을 확장시켜서 더하고 3 x 4 matrix로 만듦
c = a + b
print(c.shape)
print(c)

# Broadcastring을 사용할 때는 vector형태로 쓰지말고 matrix로 reshape하여 쓰는것이 좋다
# 그냥 vector로 쓰면 자잘한 오류가 발생하는 경우가 많다
# matrix형태로 axis를 고정시켜서 사용하도록 하자

# %% arange, linsapce

import numpy as np

# np.arange(x, y, z)
# x에서 y까지 z간격으로 ndarray 생성
a = np.arange(5, 100, 2)
print(a)

# np.linspace(x, y, z)
# x에서 y까지 z개로 나누어서 ndarray 생성
b = np.linspace(-10, 10, 21)
print(b)

# %% 다시 MSE 구하기

import numpy as np

n_point = 100
x = np.random.normal(loc=0, scale=2, size=(n_point, ))
y = 3*x
predictions = 2*x

diff_square = (y - predictions)**2
mse = np.mean(diff_square)

print(mse)

# %% 50점 이상은 통과입니다

import numpy as np

n_student = 100
scores = np.random.uniform(0, 100, size=(n_student, ))

# astype은 내가 원하는 type으로 변경해서 전달하는 method
scores = scores.astype(np.int)

student_pass = (scores > 50)  # return값은 bool
pass_percentage = np.sum(student_pass) / n_student
print(student_pass)
print(pass_percentage*100, "% 통과")

# %% k-means를 numpy로 만들기

import numpy as np

n_class, std, n_point = 2, 1, 100

dataset = np.empty(shape=(0, 2))

for class_idx in range(n_class):
    centers = np.random.uniform(-3, 3, size=(2, ))
    
    x_data = np.random.normal(loc=centers[0], scale=std, size=(n_point, 1))
    y_data = np.random.normal(loc=centers[1], scale=std, size=(n_point, 1))
    
    data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, data))
    
dataset = dataset
centroids = np.random.uniform(-5, 5, size=(n_class, 2))

template = "Shape -- dataset:{}\t centroids:{}"
print(template.format(dataset.shape, centroids.shape))

for i in range(9):
    clusters = dict()
    for cluster_idx in range(n_class):
        clusters[cluster_idx] = np.empty(shape=(0, 2))
    
    for data in dataset:
        data = data.reshape(1, -1)
        
        distances = np.sum((data - centroids)**2, axis=1)
        min_idx = np.argmin(distances)
        
        clusters[min_idx] = np.vstack((clusters[min_idx], data))
    
    for cluster_idx in range(n_class):
        cluster = clusters[cluster_idx]
        centroid = np.mean(cluster, axis=0)
        centroids[cluster_idx] = centroid
        