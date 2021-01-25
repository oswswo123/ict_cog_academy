# %% 사분면 구하기

point_coordinate = [-2, -1]

if point_coordinate[0] > 0:
    if point_coordinate[1] > 0:
        print("1사분면")
    elif point_coordinate[1] < 0:
        print("4사분면")
    else:
        print("축에 점이 위치")
elif point_coordinate[0] < 0:
    if point_coordinate[1] > 0:
        print("2사분면")
    elif point_coordinate[1] < 0:
        print("3사분면")
    else:
        print("축에 점이 위치")
else:
    print("축에 점이 위치")

# %% 사분면, x축, y축, 원점에 있는지 구하기

point = [0, -2]

if point[0] == 0 and point[1] == 0:
    print("원점")
elif point[0] == 0 and point[1]:
    print("y축")
elif point[1] == 0 and point[0]:
    print("x축")
elif point[0] > 0 and point[1] > 0:
    print("1사분면")
elif point[0] > 0 and point[1] < 0:
    print("4사분면")
elif point[0] < 0 and point[1] > 0:
    print("2사분면")
elif point[0] < 0 and point[1] < 0:
    print("3사분면")

# %% 피보나치 수열

first_number, second_number = 0, 1
number_sum = 0

print(first_number)
print(second_number)
for i in range(10):
    number_sum = first_number + second_number
    first_number = second_number
    second_number = number_sum
    print(number_sum)

# %% advanced 피보나치 수열

first_number, second_number, third_number = 0, 1, 2
number_sum = 0

print(first_number)
print(second_number)
print(third_number)
for i in range(10):
    number_sum = first_number + second_number + third_number
    first_number = second_number
    second_number = third_number
    third_number = number_sum
    print(number_sum)
    
# %% 할인된 총 가격 구하기

prices = [6000, 14000, 17000, 25000, 3000, 300]
discount_prices = list()  # 빈 리스트 만들기
sum_price = 0

for index in range(len(prices)):
    if prices[index] >= 20000:
        discount_prices.append(prices[index] * 0.85)
        sum_price += prices[index] * 0.85
    elif prices[index] >= 15000:
        discount_prices.append(prices[index] * 0.9)
        sum_price += prices[index] * 0.9
    elif prices[index] >= 10000:
        discount_prices.append(prices[index] * 0.95)
        sum_price += prices[index] * 0.95
    else:
        discount_prices.append(prices[index])
        sum_price += prices[index]

print(sum_price)

# %% Numpy로 random 변수 생성하기

import numpy as np

# numpy.random.narmal은 gaussian 분포를 따르는 랜덤변수 생성(정규분포)
# loc은 평균, scale은 분산, size는 크기
test_input = np.random.normal(loc=0, scale=1, size=(100,))

print(type(test_input))

# %% Numpy로 random 변수 생성하기 2 (최대값 추출)

import numpy as np

# numpy.random.uniform을 astype으로 생성...? (조사 필요)
scores = np.random.uniform(0, 100, size=(100, )).astype(np.int32)
maximum_value = scores[0]

for score in scores:
    if maximum_value < score:
        maximum_value = score

print(maximum_value)

# %% NoneType (값이 없는 상태)

# None는 초기값으로 사용하기 매우 좋다
# 예시) if maximum_value == None or maximum_value < score:

a = None

print(a)
print(type(a))

# %% Numpy로 random 변수 생성하기 3 (최소값 추출)

import numpy as np

scores = np.random.uniform(0, 100, size=(100, )).astype(np.int32)
minimum_value = None

for score in scores:
    if minimum_value == None or minimum_value > score :
        minimum_value = score

print(minimum_value)

# %% range의 argument

for i in range(3, 10):
    print(i)   # 초기값이 0이 아니라 3
print("---------")
for i in range(3, 10, 2):
    print(i)   # 초기값은 3으로, 간격은 2로

# %% list comprehension

# in 이후 집합의 원소를 i로 추출
# 2*i를 리스트의 원소로 입력(이를 반복)
test_list = [2*i for i in range(10)]
print(test_list)

# list comprehension에 if문을 추가
# for문을 반복하면서, if의 조건에 맞는 원소를 탑색
# 조건에 맞는 원소를 2*i로 리스트에 입력(이를 반복)
test_list2 = [i*2 for i in range(10) if i % 2 == 0]
print(test_list2)