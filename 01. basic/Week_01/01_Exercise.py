# 각 cell만 run하는 단축키 : shift + enter
# 모든 cell을 run하는 단축키 : F5

# %% 대입연산자

a = 10
b = 20
c = 10.53
d = 'shin'
e, f = 30, 40
g = 50

# %% 사칙연산자

a = 20
b = 5

print(a + b)  # 덧셈
print(a - b)  # 뺄셈
print(a * b)  # 곱셈
print(a / b)  # 나눗셈
print(a ** b)  # a의 b제곱

print(a // b)  # 몫
print(a % b)   # 나머지

# %%

score1, score2, score3, score4 = 50, 40, 60, 30
mean = (score1 + score2 + score3 + score4)/4

print(mean)

# %% Mean Squared Error 구하기

y1, y2, y3, y4 = 10, 20, 30, 40
p1, p2, p3, p4 = 40, 30, 20, 40

inputnumber = 4

mse_error = ((y1-p1)**2 + (y2-p2)**2 + (y3-p3)**2 + (y4-p4)**2) / inputnumber

print(mse_error)

# %% list 생성시 for문을 간단하게 사용하는 법

test_list = []
for i in range(10):
    test_list.append(2*i)
print(test_list)

# 좀 더 명시적인 list 생성시 for문 활용법
test_list = [2*i for i in range(10)]
print(test_list)

# %% cross-entropy 구현

import math

y = 0.7  # label
p = 0.5  # 예측값

loss = -(y*math.log(p) + (1-y)*math.log(1-p))
print(loss)

p = 0.6
loss = -(y*math.log(p) + (1-y)*math.log(1-p))
print(loss)

# %% List

scores = [30, 40, 20, 10, 60]

print(scores[0])
print(scores[-1])

# %% List pointer의 음수값 활용

file_name = 'n03450230_5037.txt'
file_name_split = file_name.split('.')

print(file_name)
print(file_name_split)

extension = file_name_split[-1]
print(extension)

# %% 빈 리스트를 만드는 방법

test_list = []
print(test_list)

test_list2 = list()  # 얘가 좀 더 명시적으로 빈 리스트를 만드는 방법
print(test_list2)

# %% 값 수정하기

''' mutable object(수정가능 object) '''
scores = [30, 40, 20, 10, 60]
print(scores)
scores[0] = 100
print(scores)

''' immutable object(수정불가능 object) '''
test_tuple = (1, 2, 3, 4, 5)
print(test_tuple[0])
# test_tuple[0] = 100   수정 불가능하므로 값을 수정하면 Error 발생

print(type(scores))
print(type(test_tuple))

# %% for문을 이용한 평균과 분산 측정 (분산 : 제곱의 합의 평균 - 평균의 제곱)

scores = [50, 40, 60, 30]

score_sum = 0;  score_squaresum = 0;
for i in scores:
    score_sum += i
    score_squaresum += i ** 2
    
mean = score_sum / len(scores)
var = (score_squaresum / len(scores)) - mean ** 2
print(mean)
print(var)

# scores의 각 원소에 10을 더하고 다시 평균 분산 측정
score_sum = 0;  score_squaresum = 0;
for i in range(len(scores)):
    scores[i] += 10

for i in scores:
    score_sum += i
    score_squaresum += i ** 2
    
mean = score_sum / len(scores)
var = (score_squaresum / len(scores)) - mean ** 2

print(mean)
print(var)
