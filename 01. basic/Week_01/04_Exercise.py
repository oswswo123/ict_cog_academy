# %% numpy를 통한 Random변수 생성 복습 (최댓값, 최솟값, 평균, 분산)

import numpy as np

scores = np.random.uniform(0, 100, size=(100, )).astype(np.int)

maximum_value = None
minimum_value = None
sum_value = 0
sqaredsum_value = 0
mean = 0
var = 0

for index in range(len(scores)):
    if maximum_value == None and minimum_value == None:
        maximum_value = scores[index]
        minimum_value = scores[index]
    elif maximum_value < scores[index]:
        maximum_value = scores[index]
    elif minimum_value > scores[index]:
        minimum_value = scores[index]
    sum_value += scores[index]
    sqaredsum_value += scores[index] ** 2

mean = sum_value / len(scores)
var = (sqaredsum_value / len(scores)) - (mean ** 2)

print(maximum_value, minimum_value, mean, var)

# %% list comprehension

# 3의 배수를 가지는 list 만들기
test_list = [i*3 for i in range(1, 100)]
print(test_list)

# 5의 배수를 가지는 list 만들기
test_list = [i for i in range(1, 100) if i % 5 == 0]
print(test_list)

# %% list comprehension 활용

test_list = [str(i) for i in range(100)]
test_list = [float(i) for i in range(100)]
test_list = [i+1 for i in range(100)]
test_list = [i%4 for i in range(100)]
test_list = [bool(i%2) for i in range(100)]
test_list = [i**2 for i in range(100)]

test_list = [i*j for i in range(10)
                 for j in range(10)
                 if i*j % 2 == 0]

# %% 객체 분석 (python의 핵심)

# python은 모든게 객체!
# 객체란 data + method
# type은 해당 객체의 성분 분석
a = 1
print(type(a))

# dir(a)는 해당 객체의 명령어를 조사
print(dir(a))

# 파이썬은 모든것이 객체이기에 C, Java와는 다른점이 많다!
test_list = [1, 2, 10.55, [1, 2, 3], "Hello World"]
print(test_list)

# %% 객체에 대한 이해 (수학 영어로 점수 list)

scores = [[10, 20], [30, 40], [50, 60]]

# means[0] = 수학평균, means[1] = 영어평균
means = [0, 0]
sums = [0, 0]

for score in scores:
    sums[0] += score[0]
    sums[1] += score[1]

means[0] = sums[0] / len(scores)
means[1] = sums[1] / len(scores)

print(means)

# %% Unpacking

a, b = [1, 2]
print(a, b)

# for score in scores: 를
# for math_score, english_score in scores: 로도 표현가능!

# %% Unpacking 연습 (평균 구하기)

names_scores = [['A', 100],
                ['B', 50],
                ['C', 30]]

sum_score, mean_score = 0, 0

# 일반적으로 python에서는 _ 에 필요없는 값을 저장한다
for _, score in names_scores:
    sum_score += score

mean_score = sum_score / len(names_scores)
print(mean_score)

# %% centroid 구하기

# centroid : 각 점들의 무게중심
coordinates = [[-2, 3], [4, 6], [-10, 30]]
cor_sums = [0, 0]

for loc_x, loc_y in coordinates:
    cor_sums[0] += loc_x
    cor_sums[1] += loc_y

centroid = [(cor_sums[0] / len(coordinates)),
            (cor_sums[1] / len(coordinates))]

print(centroid)

# %% Euclidean distance 구하기

import math

coordinates = [[-2, 3], [4, 6], [-10, 30]]
centroid = [5, -1]
eucl_dis = list()

for loc_x, loc_y in coordinates:
    eucl_dis.append(math.sqrt((loc_x - centroid[0])**2 + 
                              (loc_y - centroid[1])**2))

print(eucl_dis)

# %% Dictionary (Key와 Value로 묶인 값)

test_dict = {1:10, 'b':20, 'c':30}
print(test_dict)
print(test_dict['b'])
print(test_dict[1])

# %% Dictionary에 값 추가

means = dict()
print(means)
means['math'] = 20
print(means)
means['english'] = 30
print(means)
means['physics'] = 40
print(means)

# %% Dictionary 연습 (평균 구하기)

scores = [[10, 20, 30], [30, 40, 10], [50, 60, 50]]
sums = [0, 0, 0]

means_dict = dict()
subjects = ['math', 'english', 'science']

for math_score, eng_score, sci_score in scores:
    sums[0] += math_score
    sums[1] += eng_score
    sums[2] += sci_score

for index in range(len(sums)):
    subject = subjects[index]
    means_dict[subject] = sums[index] / len(sums)

print(means_dict)