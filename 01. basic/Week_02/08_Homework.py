# %% cross-correlation 찾기 (유사한 신호 추출)

# 1D convolutional layer와 비슷함
# 내적을 통해 매 시간대별 유사성을 출력하면 될듯?

import numpy as np

signal = np.random.normal(loc=3, scale=1, size=(100, ))
filter_ = np.array([1, 5, 3, 2, 1, 5])
f_size = filter_.shape[0]

# signal과 filter_가 만나기 시작하는 시간은 1
# signal에 filetr_가 sliding하는 표현을 하기 위해 signal에 padding 추가
signal = np.pad(signal, (f_size, f_size), 'constant', constant_values=0)

# signal과 filter를 비교하게 되는 시간의 길이
# padding이 추가된 signal에 대해서 계산해야 함
time_to_compare = signal.shape[0] - filter_.shape[0] + 1

cross_correlation = np.empty(shape=(0, 1))
for time in range(time_to_compare):
    dot_prod = np.sum(signal[time:time+f_size] * filter_)
    cross_correlation = np.vstack((cross_correlation, dot_prod))

