# DL이란?
'''
parameterized learning이다.
모든 model은 y = ax + b의 구조
(a와 b엔 온갖 정보가 들어간다.(bias, weight 등))
해당 model에서 parameter인 a, b를 train 하는것이 DL의 핵심!
따라서 model을 남에게 줄때는 a, b만 제공하면 됨
(잘 만든 a와 b만 있으면 어떤 input x에 대해서도 정확한 y가 도출된다!)
'''
# %% dictionary key, value 추출

test_dict = {'a':[1, 2, 3, 4], 'b':2, 'c':3}

# dict에 접근하면 key를 순차적으로 출력한다.
for i in test_dict:
    # 여기서 test_dict[key]를 사용하면 value에 접근 가능
    print(i, test_dict[i], sep=' - ')

# 아니면 이런 방법도 있다
# i에는 key와 value가 같이 묶여서 나옴
for i in test_dict.items():
    print(i)
    key, value = i
    print(key, value)

# %% enumerate

# Q) index를 만들지 않고 비교하는 방법은?

scores = [10, 30, 50, 20, 10]

for idx, score in enumerate(scores):
    # 여기서 idx는 몇 번째 loop인지의 data를 담고있음!
    print(idx, score)

# enumerate는 loop와 collection의 원소를 tuple형태로 반환!
for a in enumerate(scores):
    print(type(a))
 
# %% Numpy
'''
Numpy는 모든 벡터화와 관련된다.
그래서 Numpy는 데이터 처리에 매우 효율적이다.
연산을 위해 사용되는 온갖 for문을 없앨 수 있음!
'''
import numpy as np

# a와 b는 type이 numpy.ndarray (list가 아니다!)
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# a와 b의 concatenate가 아니라 elementwise로 operator가 실행됨
# vector의 계산을 하기위해!
print(a + b)
print(a - b)

# 그래서 Mean squared error를 이렇게 구할 수 있다
# np.sum은 numpy.ndarray들의 각 원소를 합쳐서 numpy.float64로 만듦
diff = a - b
diff_squared = diff**2
mse = np.sum(diff_squared) / len(a)
print(mse)

# a.mean()   : 평균 구하기                               (np.mean(a)와 같다)
# a.std()    : 표준편차 구하기                           (np.std(a)와 같다)
# a.sort()   : 정렬                                     (np.sort(a)와 같다)
# a.max()    : 최대값 구하기                             (np.max(a)와 같다)
# a.min()    : 최소값 구하기                             (np.min(a)와 같다)
# a.argmax() : 최대값의 index 구하기                     (np.argmax(a)와 같다)
# a.argmin() : 최소값의 index 구하기                     (np.argmin(a)와 같다)
# a.shape    : 데이터의 형태 출력 (n x n 인지, 벡터인지)  (np.shape(a)와 같다)
    
# %% k-means 다시!

# matplotlib를 통해 시각화해서 보자!
import numpy as np
import matplotlib.pyplot as plt

n_class, std, n_point = 3, 1 ,100
dataset = np.empty(shape=(0,2))

fig, ax = plt.subplots(figsize=(10, 10))
for idx in range(n_class):
    center = np.random.uniform(-15, 15, size=(2, ))
    
    x_data = np.random.normal(loc=center[0], scale=std, size=(n_point, 1))
    y_data = np.random.normal(loc=center[1], scale=std, size=(n_point, 1))
    
    class_data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, class_data))
    
    ax.scatter(x_data, y_data)

ax.set_xlabel("X data", fontsize=30)
ax.set_ylabel("Y data", fontsize=30)
ax.grid()

# 기존의 centroid 추출 방식은 division by zero error가 자주 일어난다
# why? 초기 centroid가 자신의 cluster element를 하나도 못 가질 수 있음
# centroid = np.random.uniform(-3, 3, size=(n_class, 2)).tolist()
# numpy.random.choice를 통해 dataset에서 무작위 data를 centroid로 지정
centroid_indice = np.random.choice(len(dataset), size=(n_class, ))
centroid = dataset[centroid_indice].tolist()

# 오늘은 dict도 써보자!
cluster_dict = dict()
for class_idx in range(n_class):
    cluster_dict[class_idx] = list()
    
# clustering
for x_loc, y_loc in dataset:
    # 최소 거리와 데이터가 해당하는 cluster를 의미하는 변수 초기화
    min_dist = None
    min_dist_point = 0
    iterator = 0
    # centroid별 data와 거리 측정
    for x_ctd, y_ctd in centroid:
        dist = (x_ctd - x_loc)**2 + (y_ctd - y_loc)**2
        if min_dist == None or min_dist > dist:
            min_dist = dist
            min_dist_point = iterator
        iterator += 1
    # 각 클러스터에 data를 append
    cluster_dict[min_dist_point].append([x_loc, y_loc])

# dict를 이용해 각 cluster들의 중심점 측정
for cluster_idx, cluster in cluster_dict.items():
    x_sum, y_sum, iter_cnt = 0, 0, 0
    for data_point in cluster:
        x_sum += data_point[0]
        y_sum += data_point[1]
        iter_cnt +=1
    x_center = x_sum / iter_cnt
    y_center = y_sum / iter_cnt
    
    if centroid[cluster_idx] == [x_center, y_center]:
        pass # 일치하기에 움직이지 않음
    else:
        centroid[cluster_idx] = [x_center, y_center]

# %% Numpy를 이용한 간결한 K-means

import numpy as np
import matplotlib.pyplot as plt

n_class, std, n_point = 3, 1 ,100
dataset = np.empty(shape=(0,2))

fig, ax = plt.subplots(figsize=(10, 10))
for idx in range(n_class):
    center = np.random.uniform(-15, 15, size=(2, ))
    
    x_data = np.random.normal(loc=center[0], scale=std, size=(n_point, 1))
    y_data = np.random.normal(loc=center[1], scale=std, size=(n_point, 1))
    
    class_data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, class_data))
    
    ax.scatter(x_data, y_data)

ax.set_xlabel("X data", fontsize=30)
ax.set_ylabel("Y data", fontsize=30)
ax.grid()

# 기존의 centroid 추출 방식은 division by zero error가 자주 일어난다
# why? 초기 centroid가 자신의 cluster element를 하나도 못 가질 수 있음
# centroid = np.random.uniform(-3, 3, size=(n_class, 2)).tolist()
# numpy.random.choice를 통해 dataset에서 무작위 data를 centroid로 지정
centroid_indice = np.random.choice(len(dataset), size=(n_class, ))
centroid = dataset[centroid_indice].tolist()

for idx in range(9):
    cluster_dict = dict()
    for class_idx in range(n_class):
        cluster_dict[class_idx] = list()
        
    for data in dataset:
        data = data.reshape(-1, 2)
        # numpy.power를 이용한 거리 계산 (각 원소들의 x, y 좌표차)
        distances = np.power(data - centroid, 2)
        # numpy.sum을 이용한 x, y 좌표차 합하기 (유클리디안 거리로 변경)
        # axis = -1은 마지막 차원을 의미함
        distances = np.sum(distances, axis=-1)  # 사실 axis=0이 맞음(현 코드는 틀림)
        
        # numpy.argmin을 이용한 최소 거리 cluster index 찾기
        m_idx = np.argmin(distances)
        cluster_dict[m_idx].append(data)
    
    # cluster_dict를 조회하여 각 cluster의 center 측정
    for cluster_key, clusters in cluster_dict.items():
        x_mean = np.mean(clusters[0])
        y_mean = np.mean(clusters[1])
        
        # cluster center로 centroid 이동
        centroid[cluster_key] = [x_mean, y_mean]

# %% KNN (K-Nearest Neighbor)
'''
label이 있는 data에 적용 가능
새로운 input이 가장 가까운 기존의 group으로 취급되는 algorithm
핵심 아이디어 : input에서 가장 가까운 label은 어디에 소속되어있는가?
Classification을 하는 기법
한계점 : dataset의 규모(차원, 갯수 등)가 커지면 문제가 생김!
         time complexity + space complexity
'''
