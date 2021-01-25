# %% list sort

a = [50, 30, 10, 60, 20, 70, 15]
print(a)

a.sort()
print(a)

# %% sort기능 없이 sorting 만들기 (python은 buffer없이 값 교체가 가능)

# python은 a, b = b, a 를 통해 값 교체가 가능하다.
a = [50, 30, 10, 60, 20, 70, 15]

for first_index in range(len(a)):
    for second_index in range(first_index+1, len(a)):
        if a[first_index] > a[second_index]:
            a[first_index], a[second_index] = a[second_index], a[first_index]

print(a)
            
# %% 평균은 넘겠지

# 평균을 넘는 학생들의 비율 출력
import numpy as np

scores = np.random.normal(loc=50, scale=10, size=(100,))
score_sum, mean, mean_rate, avg_over = 0, 0, 0, 0

for score in scores:
    score_sum += score
mean = score_sum / len(scores)

for score in scores:
    if score > mean: 
        avg_over += 1
mean_rate = avg_over / len(scores)

print(mean_rate*100, "%", sep='')

# %% Data science의 Vector
'''
수학에서의 Vector
- Vector Space의 원소

Data science의 Vector란?
- Feature의 집합
Data sciende에서의 Vector의 내적(Dot product)
- 두 sample을 내적했을때 값이 클수록 유사한 성향을 가짐
- a dot b = 각 차원들의 곱의 합

행렬의 곱셈
- 행렬의 곱은 가로 x 세로
- 이때 앞 행렬의 열과 뒷 행렬의 행의 정보가 동일하면 의미를 가짐
- (2 x "2") * ("2" x 1) 에서 앞행렬의 열과 뒷행렬 행이 같음
- 행렬의 결과는 앞행렬의 행과 뒷 행렬의 열로 이루어짐
- ex) 초콜릿과 사탕의 수 행렬 x 초콜릿과 사탕의 가격 행렬
        초콜릿  사탕                평시가격  연말가격
     A    2      4         초콜릿     500      1000
     B    5      2     x    사탕      300      600
     C    4      6
     → 초콜릿과 사탕으로 정보가 일치, 결과값은 각 세트의 시기별 가격
     → (3x[2]) * ([2]x2) 였을 때, 대괄호 안의 숫자가 사라지고 (3x2)가 됨
'''
# %% 행렬 계산 (numpy를 사용한)

import numpy as np
A = np.array([[2, 4], [5, 2], [4, 6]])
B = np.array([[500, 1000], [300, 600]])

print(np.matmul(A, B))

# %% Vector Space
'''
벡터 공간(Vector Space)
- 선형대수학의 개념
- 벡터 공간 또는 선형 공간은 원소를 서로 더하거나 주어진 배수로 늘이거나 줄일 수 있는 공간이다.
- 즉, 그 벡터들이 갈 수 있는 모든 공간
- 원소들을 스칼라 곱 시키거나, 원소 벡터간 합을 시켜도 해당 공간을 벗어나지 않음

공학자의 Vector Space
- 어떠한 데이터를 기준으로 스칼라 곱하거나, 데이터간 합했을 때 같은 유형의 데이터인 것
- 특별한 예외들은 예외처리를 해도 됨
- ex) 소리의 파형, 이미지의 RGB
'''
# %% Vector Additions

a = [1, 2, 3]
b = [4, 5, 6]
c = list()

for index in range(len(a)):
    c.append(a[index] + b[index])

print(c)

# %% Hadamard Product

a = [1, 2, 3]
b = [4, 5, 6]
c = list()

for index in range(len(a)):
    c.append(a[index] * b[index])

print(c)

# %% 행렬 곱(matrix multiplication)

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[10, 20], [30, 40], [50, 60]]

n_row, n_col, n_iter = len(A), len(B[0]), len(A[0])
C = list()
buffer_list = list()
product_value = 0

for first_index in range(n_row):
    product_value = 0
    for second_index in range(n_col):
        for repeat in range(n_iter):
            product_value += A[first_index][repeat] * B[repeat][second_index]
        buffer_list.append(product_value)
        product_value = 0
    C.append(buffer_list)
    buffer_list = list()

print(C)

# %% 내적(Dot product)

a = [1, 2, 3]
b = [4, 5, 6]
dot_product = 0;

for iterator in range(len(A)):
    dot_product += a[iterator] * b[iterator]

print(dot_product)

# %% Vector Norm ( ||u|| )
'''
Vector Norm은 벡터의 크기를 뜻한다
즉, 중점에서 해당 벡터까지의 거리이다
'''
import math

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 19]
sq_sum = 0

for a_ele in a:
    sq_sum += a_ele**2
vec_norm = math.sqrt(sq_sum)

print(vec_norm)

# %% 단위 벡터(Unit vector)
'''
벡터들을 1로 나누어 주는 값
각 원소들을 Vector norm으로 나누어 주면 된다.
'''
import math

a = [3, 4, 5]
unit_vec = list()
sq_sum = 0

for a_ele in a:
    sq_sum += a_ele ** 2
vec_norm = math.sqrt(sq_sum)

for a_ele in a:
    unit_vec.append(a_ele / vec_norm)

print(unit_vec)

# %% 단위벡터의 내적

import math

data_set = [[20, 80], [5, 20], [10000, 100]]

# 각 vector 단위벡터 구하기
unit_vecs = list()
for vec in data_set:
    sq_sum = 0
    for ele in vec:
        sq_sum += ele ** 2
    vec_norm = math.sqrt(sq_sum)
    
    unit_vec = list()
    for ele in vec:
        unit_vec.append(ele / vec_norm)
    unit_vecs.append(unit_vec)


# 단위벡터들 내적 구하기
a_b = 0
for iterator in range(2):
    a_b += unit_vecs[0][iterator] * unit_vecs[1][iterator]

a_c = 0
for iterator in range(2):
    a_c += unit_vecs[0][iterator] * unit_vecs[2][iterator]

b_c = 0
for iterator in range(2):
    b_c += unit_vecs[1][iterator] * unit_vecs[2][iterator]
    
print(a_b, a_c, b_c)

# %% Special method

a, b = 10, 20

# integer들이 '+'의 입력을 받으면 method __add__를 호출
# 두 수룰 더함
print(a + b)
print(a.__add__(b))

# list들이 '+'의 입력을 받으면 method __add__를 호출
# 두 리스트들을 concatenate
a, b = [10, 20], [30, 40]
print(a + b)
print(a.__add__(b))
'''
'+'를 입력할 경우 Special method인 __add__가 호출됨
pyhton의 integer객체와 list객체간 __add__의 정의가 다름
__add__와 같은 method들은 Special method라 부른다
python은 다른 언어와 다르게 Special method를 통해 operator의 기능을 변경할 수 있다
'+', '-' 등의 기능을 내가 만든 클래스에서 정의하거나 변경할 수 있다
'''

# %% Special method 변경하기 

class RGB_pixel:
    def __init__(self, rgb):
        self.tmp = rgb
        
    def __add__(self, operand2):
        return "Hello World!"

        
pixel1 = RGB_pixel([100, 150, 20])
pixel2 = RGB_pixel([50, 30, 80])
print(pixel1 + pixel2)

# %% print 함수 사용하기

a, b = 10, 20

# escape character
print(a, '\n', b)
print(a, '\t', b)

# string casting
print(str(a) + ' Hello')

# %% String formatting

templeate = 'format value : {}'.format(10)
print(templeate)
templeate = 'format value : {}'.format("Hello world")
print(templeate)

# %% String formatting 2

# 뒤에 오는 알파벳은 데이터 타입을 정해줌
templeate = 'format value : {:s}'.format("Hello")  # 문자열
print(templeate)

templeate = 'format value : {:d}'.format(20)  # 10진수 정수
print(templeate)

templeate = 'format value : {:f}'.format(20.005)  # 실수
print(templeate)

templeate = 'format value : {:b}'.format(20)  # 2진수
print(templeate)

templeate = 'format value : {:o}'.format(20)  # 8진수
print(templeate)

templeate = 'format value : {:x}'.format(20)  # 16진수
print(templeate)

# %% padding + alignment

# 10칸 미리 확보하고 채워넣기
templeate = 'format value : {:10}'.format(1000)
print(templeate)

templeate = 'format value : {:<10d}'.format(1000)  # 왼쪽 정렬
print(templeate)

templeate = 'format value : {:^10d}'.format(1000)  # 가운데 정렬
print(templeate)

templeate = 'format value : {:>10d}'.format(1000)  # 오른쪽 정렬
print(templeate)

# %% floating point

# 소수점 2째자리까지 출력하는 실수
templeate = 'format value : {:.2f}'.format(10.15641)
print(templeate)

# 소수점 3째자리까지 가운데 정렬로 출력하는 실수
templeate = 'format value : {:^10.3f}'.format(10.156415151654)
print(templeate)

# %% 2개 이상의 값 formatting

templeate = 'A : {}  B : {}  C : {}'.format(10, 20, 30)
print(templeate)

# 중괄호 안에 숫자가 들어갈 경우 해당 순서의 데이터가 들어감
templeate = 'A : {0}  B : {2}  C : {1}'.format(10, 20, 30)
print(templeate)

# key-value를 이용한 formatting
templeate = '{name} : {age}'.format(name="Oh", age=26)
print(templeate)

