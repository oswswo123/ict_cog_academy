# 평균, 분산 구하기 (using function)

# 최대점수, 최소점수 index들 구하기

# 유용한 module 소개(termcolor, tqdm)

# MSE 구하기

# 2D Convolution

# BCE (integer format, one-hot-vector format)

# sigmoid, tanh, ReLU

''' 수학 
1. 미분(derivatives)
2. Chain Rule
3. Linear regression
4. Logistic regression
'''

# Linear regression

# Logistic regression

# %% 평균, 분산, 최대값, 최소값 구하기 (using function)

import numpy as np

# function사이는 2줄의 간격을 만드는것이 pep8 conventions
def get_random_scores(n_student):
    '''
    Random한 점수 dataset을 만들어주는 function
    input : student의 수
    return : input으로 받은 숫자만큼의 랜덤한 점수 array
    '''
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, ))
    scores = scores.astype(np.int)
    return scores


def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean


def get_variance(scores):
    squared_sum = 0
    mean = get_mean(scores)
    for cnt, score in enumerate(scores):
        squared_sum += score**2
    variance = squared_sum / (cnt+1) - mean**2
    return variance


def get_max_min(scores, M, m):
    max_score, min_score = None, None
    for idx, score in enumerate(scores):
        if max_score == None or max_score < score:
            max_idx = idx
            max_score = score
        
        if min_score == None or min_score > score:
            min_idx = idx
            min_score = score
    
    if M == True and m == True:
        return max_score, max_idx, min_score, min_idx
    if M == True and m == False:
        return max_score, max_idx
    if M == False and m == True:
        return min_score, min_idx


n_student = 100
scores = get_random_scores(n_student)
    
scores_mean = get_mean(scores)
scores_var = get_variance(scores)
#scores_mean = np.mean(scores)
#scores_var = (np.sum(scores**2) / n_student) - (scores_mean**2)

M, M_idx, m, m_idx = get_max_min(scores, M=True, m=True)
M, M_idx = get_max_min(scores, M=True, m=False)
m, m_idx = get_max_min(scores, M=False, m=True)


# %% 유용한 module 소개 1 (termcolor) - print에 색칠하는 module... (이쁘다)

from termcolor import colored

print(colored('hello world!!', 'red'))
print(colored('hello world!!', 'magenta'))
print(colored('hello world!!', 'red', 'on_white'))
print(colored('hello world!!', 'magenta', 'on_red'))

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Dataset is loading')

# %% 유용한 module 소개 2 (tqdm) - progress bar 표현을 위한 module

from tqdm import tqdm
import time

for idx in tqdm(range(1000)):
    time.sleep(0.01)

# %% MSE 구하기

import numpy as np

def get_mse(ground_truths, predictions):
    mse = np.mean((ground_truths - predictions)**2)
    return mse

n_point = 100
ground_truths = np.random.normal(loc=3, scale=1, size=(n_point, ))
predictions = np.random.normal(loc=3, scale=1, size=(n_point, ))


# %% 2D convolution (using function)

import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    # image를 gray 시키는 함수
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def conv2d(img_gray, filter_):
    # image의 x, y 길이 저장
    height, width = img_gray.shape
    filter_len = filter_.shape[0]
    
    # 2D Conv 연산을 적용시킨 이미지를 찍을 array 생성
    img_2D_conv = np.zeros(shape=(height - filter_len + 1, width - filter_len + 1))
    for h_idx in range(height - filter_len):
        for w_idx in range(width - filter_len):
            convolution_cal = np.sum(img_gray[h_idx:h_idx+filter_len,
                                              w_idx:w_idx+filter_len] * filter_)
            
            img_2D_conv[h_idx][w_idx] = convolution_cal
    
    return img_2D_conv


dir_path = "C:\\Users\\OhSeungHwan\\.spyder-py3\\ict_cog_academy_project\\Week_02"
file_name = "\\test_image.jpg"

img = plt.imread(dir_path + file_name)
img_gray = rgb2gray(img)

# filter의 크기 설정
blur = 21

# 스무디 필터 생성
filter_ = np.ones(shape=(blur, blur))
filter_ /= filter_.size

img_2D_conv = conv2d(img_gray, filter_)

# 시각화 module
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].imshow(img_gray, 'gray')
ax[1].imshow(img_2D_conv, 'gray')

# %% input data processing - 여러 과목 한번에 평균내기

import numpy as np


def get_random_scores(n_student, n_subject):
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, n_subject))
    scores = scores.astype(np.int)
    return scores


def get_mean(scores):
    mean = np.mean(scores, axis=0)
    return mean


n_student = 100
n_subject = 5

scores = get_random_scores(n_student, n_subject)
mean = get_mean(scores)

# %% BCE (integer format, one-hot-vector format)

import numpy as np


def get_bce(labels, predictions):
    if len(labels.shape) == 1:
        # np.eye(x)[y] : one-hot-encoding 하는 함수
        # x는 class의 종류, y는 dataset
        labels = np.eye(2)[labels]
    bce = -1 * labels*np.log(predictions)
    bce = np.sum(bce, axis=1)
    bce = np.mean(bce)
    return bce


labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

bce = get_bce(labels, predictions)
print(bce)

labels = np.array([[0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1]])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

bce = get_bce(labels, predictions)
print(bce)

# %% sigmoid, tanh, ReLU 만들기

import matplotlib.pyplot as plt

# np.exp(x) : 자연로그 e를 사용하는 함수
# e^x값을 return
affine = np.array([-5, 2, 6, 8, 1])

def sigmoid(affine): return 1 / (1 + np.exp(-affine))
def tanh(affine): return (np.exp(affine) - np.exp(-affine)) / (np.exp(affine) +  np.exp(-affine))

# np.maximum(x, y) : x와 y의 각 element를 비교해 더 큰 값을 return하는 함수
def relu(affine): return np.maximum(0, affine)

x_range = np.linspace(-10, 10, 300)
a_sigmoid = sigmoid(x_range)
a_tanh = tanh(x_range)
a_relu = relu(x_range)

plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(x_range, a_sigmoid, label='Sigmoid')
ax.plot(x_range, a_tanh, label='Tanh')
ax.plot(x_range, a_relu, label='ReLU')

ax.tick_params(labelsize=20)
ax.legent(fontsize=40)

# %% 수학
'''  
1. 미분(derivatives)
2. Chain Rule(연쇄법칙)
3. Linear regression
4. Logistic regression

1. 미분(derivatives)
- 현재 값보다 값의 변화율이 중요할때 그 점의 기울기
- 기울기 : y의 변화량 / x의 변화량  →  f(x+h) - f(x) / (x+h) - x
- f(x)의 도함수는 (d / dx) * [f(x)], 그래서 보통 f(x)는 y로 두기에 dy / dx가 됨
- 이 기울기의 h를 0에 수렴하도록 하면 미분
- 이를 간단히 하기위해 미분법이 등장
- y = ln(x)  →  dy/dx = 1/x
- y = e^x  →  dy/dx = e^x
- y = x^n  →  dy/dx = n * x^(n-1) 
- y - log_a(x)  →  dy/dx = 1 / x ln a

- Deep learning에서는 cost function의 최저점을 찾기위해 미분을 사용한다
- 미분계수가 0보다 크면 음의방향에, 0보다 작으면 양의 방향에 최저점이 있다

2. Chain Rule(연쇄법칙)
- y = (x - 1)³ 을 미분하려면?
- Chain Rule에 의해 (d / dx) * [f(g(x))] = f'(g(x))g'(x)
- Chain Rule의 적용을 위해 x - 1을 u로 치환해보자
- (d / du) * [u³] * (d / dx) * [x - 1]
- 지수의 미분법칙에 의해 3u² * (d / dx) * [x - 1]
- u를 모두 x - 1로 바꾸면 3(x - 1)² * (d / dx) * [x - 1]
- 따라서 3(x - 1)²

- 즉, y = f(x), x = g(t)일 때
- d / dt * [f(x)] = d / dx * [f(x)] * d / dt * [g(t)]
  → dy / dt = dy / dx * dx / dt임을 뜻한다

편미분
- 여러 변수에 대해서 각 변수마다 미분을 해주는 것
- 각 변수에 대해서 미분할 때, 다른 변수는 모두 상수 취급한다
- ex) f(a, b) = (y - (ax + b))²을 a와 b에 대해 편미분하면,
    결과값은 (d / da * [f(a, b)], d / db * [f(a, b)])인 2차원 data가 나온다
'''
# %% Linear regression 구현

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n_point = 100

# x : input,  y : label
x = np.random.normal(loc=0, scale=1, size=(n_point, ))
y = 5*x + 0.2*np.random.normal(loc=0, scale=1, size=(n_point, ))

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y)
ax.axvline(x=0, linewidth=3, color='black', alpha=0.5)
ax.axhline(y=0, linewidth=3, color='black', alpha=0.5)

# a : predictor의 parameter
a = -10

learning_rate = 0.01
n_iter = 500
cmap = cm.get_cmap('rainbow', lut=n_iter)

for i in range(n_iter):
    # predictor에 parameter update
    predictions = a*x    
    
    # gradient : Cost function의 미분값 (여기선 MSE 사용)
    # 이 모델은 parameter가 a만 있기에 dL / da 한번으로 가능
    # 하지만 parameter가 추가되어 f(x) = ax + b가 된다면, 편미분을 해야함
    # f(x) = ax1 + bx2 + c가 된다면? dL/da, dL/db, dL/dc를 모두 구해 a, b, c에 update해야함
    gradient = -2 * np.mean(x * (y - predictions))
    
    # gradient update (미분계수가 0보다 크면 음의방향에, 0보다 작으면 양의방향에 최저점이 있다)
    # 혹시 model에 b값이 있다면 dL / db도 구해서 update해야함
    a = a + (-1 * gradient * learning_rate)
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    ax.plot(x_range, y_range, color=cmap(i))

# %% Logistic regression 구현

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n_point = 100
x = np.random.normal(loc=0, scale=1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(loc=0, scale=1, size=(n_point, ))
y = (x_noise >= 0).astype(np.int)

fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(x, y)

# parameter 설정
a, b = np.random.normal(loc=0, scale=1, size=(2, ))
n_iter = 1000
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = list()

for i in range(n_iter):
    # affine function으로 변환
    # z = ax + b
    affine = a*x + b
    
    # affine function에 sigmoid 적용
    # ŷ = 1 / (1 + e^(-z))
    predictions = 1 / (1 + np.exp(-affine))
    
    # BCE를 이용하여 cost function 측정
    # 사실 이 식에선 미분된 값을 써서 필요없다. losses list에 채워넣기 위함일 뿐
    loss = -1 * np.mean(y * np.log(predictions) + 
                        (1-y) * np.log(1-predictions))
    
    # cost function의 미분값을 사용해 gradient 측정
    # Chain rule !
    dl_da = -1 * np.mean(x * (y - predictions))
    dl_db = -1 * np.mean(y - predictions)
    
    # parameter update
    a = a + (-1 * learning_rate * dl_da)
    b = b + (-1 * learning_rate * dl_db)
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1 / (1 + np.exp(-y_range))
    ax.plot(x_range, y_range, color=cmap(i))
    
    losses.append(loss)
    
ax[0].grid()

