# %% k-means clustering 구현
'''
1. 무작위 지점을 centroid로 지정
2. centroid에 가까운 data들을 cluster로 묶음 (clustering)
3. 각 cluster의 중심점으로 centroid 이동
4. 2~3 반복
5. centroid 이동이 없으면 종료
'''
# np.random.uniform은 전 범위에서 일정하게(마치 주사위 던지듯이) 랜덤값 생성
# np.random.normal은 정규분포(gaussian)을 따르는 랜덤값 생성
import numpy as np

# dataset을 만드는 함수 생성
def make_dataset(n_class, std, n_point):
    # dataset 만들기
    # 0 x 2의 빈 dataset 생성
    dataset = np.empty(shape=(0, 2))
    for idx in range(n_class):
        # size=(2, )는 벡터임을 나타냄. size=(2, 1)이라면 행렬이 된다!
        centers = np.random.uniform(-10, 10, size=(2, ))
        
        # 100 x 1 데이터 벡터를 두개 만듦
        x_data = np.random.normal(loc=centers[0], scale=std, size=(n_point, 1))
        y_data = np.random.normal(loc=centers[1], scale=std, size=(n_point, 1))
        
        # 두 벡터를 합쳐서 100 x 2 데이터 셋을 생성(두 벡터의 열을 합침)
        # horizontal stack (feature를 연결)
        data = np.hstack((x_data, y_data))
        
        # 생성된 cluster data를 dataset에 쌓음
        # vertical stack (sample을 연결)
        dataset = np.vstack((dataset, data))
        
    return dataset

n_class = 5
std = 1
n_point = 100

dataset = make_dataset(n_class, std, n_point)
dataset = dataset.tolist()  # numpy array를 python의 list로 변경
'''
무작위로 생성하니 자꾸 zero division error 발생.
제일 앞에서 class숫자만큼 뽑아 cluster로 지정해야겠다
# 무작위로 centroid 생성
centroid = np.random.uniform(-0.5, 0.5, size=(n_class, 2))
centroid = centroid.tolist()
'''
centroid = list()
for idx in range(n_class):
    centroid.append(dataset[idx])

while True:    
    # 각 dataset를 가장 가까운 centroid에 해당하게 clustering
    # 빈 cluster list 생성
    cluster = list()
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
        # 데이터가 해당하는 cluster를 cluster list에 저장
        cluster.append(min_dist_point)
            
    # 각 cluster의 중심점으로 centroid 이동
    # cluster의 center를 저장할 10칸짜리 list 생성
    cluster_center = [[0, 0] for idx in range(n_class)]
    sums_of_cluster = [[0, 0, 0] for idx in range(n_class)]  # [x, y, repeat]
    for idx in range(n_class * n_point):
        # cluster들의 합 구하기
        sums_of_cluster[cluster[idx]][0] += dataset[idx][0]
        sums_of_cluster[cluster[idx]][1] += dataset[idx][1]
        sums_of_cluster[cluster[idx]][2] += 1  # 해당 cluster의 element 수
    
    # cluster들의 center구하기 (division by zero error 발생)
    # 일부 cluster는 소속된 element를 가지지 못할 경우가 발생
    for idx in range(len(cluster_center)):
        cluster_center[idx][0] = sums_of_cluster[idx][0] / sums_of_cluster[idx][2]
        cluster_center[idx][1] = sums_of_cluster[idx][1] / sums_of_cluster[idx][2]
    
    # cluster_center와 centroid가 다른지 확인
    # 같다면 이동할 수 없으니 종료, 다르다면 cluster_center로 centroid 이동
    if centroid == cluster_center:
        break  # 알고리즘 종료
    else:
        for idx in range(n_class):
            centroid[idx] = cluster_center[idx]
