# %% binary cross-entropy 구하기

# cross-entropy는 classificaion에서 cost function으로 많이 씀
# 확률의 예측이 어느정도 정확한지를 확인할 때 씀
# 상호 배타적 상황(모든 경우의 확률합이 1)일 경우는 categorical cross-entropy
# ex) Multi-Class Classification에서 사용 : 결과 [1, 0, 0] (이 사진은 개다)
# 일반적으로 softmax와 함께 사용
# 범용형 상황에서는 binary cross-entropy
# ex) Multi-Label Classification에서 사용 : 결과 [1, 0, 1] (이 사진엔 개와 닭이 있다)
# 일반적으로 sigmoid와 함께 사용
# - ( y * log(pred) + (1 - y) * log(1 - pred) )

import math

ground_truth = [0.7, 0.3]
prediction1 = [0.5, 0.5]
prediction2 = [0.9, 0.1]
prediction3 = [0.4, 0.6]

cross_entropy = list()
cross_entropy.append(-(ground_truth[0] * math.log(prediction1[0])
                     + ground_truth[1] * math.log(prediction1[1])))

cross_entropy.append(-(ground_truth[0] * math.log(prediction2[0])
                     + ground_truth[1] * math.log(prediction2[1])))

cross_entropy.append(-(ground_truth[0] * math.log(prediction3[0])
                     + ground_truth[1] * math.log(prediction3[1])))

print(cross_entropy)

# %% binary cross-entropy loss 구하기(평균)

import math

ground_truth = [0.7, 0.3]
prediction1 = [0.5, 0.5]
prediction2 = [0.9, 0.1]
prediction3 = [0.4, 0.6]

cross_entropy = list()
cross_entropy.append(-(ground_truth[0] * math.log(prediction1[0])
                     + ground_truth[1] * math.log(prediction1[1])))

cross_entropy.append(-(ground_truth[0] * math.log(prediction2[0])
                     + ground_truth[1] * math.log(prediction2[1])))

cross_entropy.append(-(ground_truth[0] * math.log(prediction3[0])
                     + ground_truth[1] * math.log(prediction3[1])))

print(cross_entropy)

sum_cros = 0
for cros in cross_entropy:
    sum_cros += cros
cost_func = sum_cros / 3
print(cost_func)

# %% BCE + Numpy

import numpy as np

labels = np.array([0.9, 0.7, 0.4])
predictions = np.array([0.7, 0.5, 0.7])

cross_entropy = -1*np.mean(labels * np.log(predictions)
                           + (1-labels) * np.log(1-predictions))
print(cross_entropy)

# %% One-hot encoding의 BCE 구하기

# one-hot vector : (0, 0, 0, 1, 0, 0) - 정답인 하나만 1이 되는 확률분포 벡터
# 정답을 one-hot vector로 바꾸는 것을 one-hot encoding이라 함

import numpy as np

# 아래 두 예제는 같은 data이다.

# 데이터는 3개, item도 3개
labels = np.array([1, 0, 1])
predictions = np.array([0.9, 0.1, 0.6])


cross_entropy1 = -1 * np.mean(labels * np.log(predictions)
                             + (1-labels) * np.log(1-predictions))
print(cross_entropy1)

# 주의! 데이터는 3개지만, matrix의 item은 6개!
labels = np.array([[0, 1], [1, 0], [0, 1]])
predictions = np.array([[0.1, 0.9],
                        [0.9, 0.1],
                        [0.4, 0.6]])

temp_calcul = labels * np.log(predictions)
temp_calcul = np.sum(temp, axis=1)
cross_entropy2 = -1 * np.mean(temp_calcul)

print(cross_entropy2)

# %% Accuracy 구하기

import numpy as np

labels1 = np.array([1, 0, 1, 0, 1, 0, 1])
predictions1 = np.array([0, 1, 0, 0, 0, 1, 1])

# ndarray끼리 == 연산자를 사용하면
# 일치할 때 True, 다를 때 False를 return한 bool Array를 만듦
print(labels1 == predictions1)

accuracy1 = np.sum(labels1 == predictions1) / labels1.shape[0]
print(accuracy1)

labels2 = np.array([1, 0, 1, 0, 1, 0, 1])
predictions2 = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.6, 0.7])

# prediction2의 element를 반올림(0.5이상은 1, 미만은 0으로)
predictions2 = np.around(predictions2)
# predictions2 = (predictions2 >= 0.5)  을 해도 비슷한 결과를 얻음

accuracy2 = np.mean(labels2 == predictions2)
print(accuracy2)

# %% Accuracy 구하기 2

import numpy as np

label = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

# Numpy.argmax를 이용하여 predictions이 예측한 정답을 추출
predictions = np.argmax(predictions, axis=1)
print(predictions)

# 정확도 연산
accuracy = np.sum(label == predictions).astype(np.int) /label.shape[0]
print(accuracy)

# %% Function

# python은 함수 생성을 def로
# python은 모든 변수가 객체이기에 return 타입을 선언하지 않음
def test_funct(input1, input2):
    result = input1 + input2
    return result

a = test_funct(5, 10)
print(a)

# %% Namespace (c.f. Local space & Global space)

# 이 a는 global variable
a = 10

# locals()는 자신의 Namespace에 있는 정보들을 알려줌
# ex) method, variable, path, log, ...
print(locals())  

def test_function():
    # 이 a는 local variable
    a = 20
    print(a)

def test_function2():
    # 이 a도 local variable
    a = 30
    print(a)

print(a)
test_function()
test_function2()