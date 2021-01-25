# 런던 자전거 대여 dataset으로 linear regression 구현

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x_data = list()
y_data = list()
with open('./london_merged.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # 첫 줄은 feature의 설명. 한줄 skip
    next(csvreader)
    
    for line in csvreader:
        _, cnt, t1, t2, hum, *_ = line
        x_data.append([float(t1), float(t2), float(hum)])
        y_data.append(int(cnt))

x_data = np.array(x_data)
y_data = np.array(y_data)

w0, w1, w2, b = np.random.normal(0, 1, size=(4, ))
learning_rate = 0.0001
losses = list()

fig, ax = plt.subplots(figsize=(20, 12))
cmap = cm.get_cmap('rainbow', lut=len(x_data))

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    # forward
    # z : affine function
    # z = w₀x₀ + w₁x₁ + w₂x₂ + b
    z = w0 * x[0] + w1 * x[1] + w2 * x[2] + b
    # loss : MSE
    loss = (y - z)**2
    losses.append(loss)
    
    # backward
    # using Chain Rule
    dl_dz = -2 * ( y - z )
    
    dz_dw0 = x[0]
    dz_dw1 = x[1]
    dz_dw2 = x[2]
    dz_db = 1
    
    dl_dw0 = dl_dz * dz_dw0
    dl_dw1 = dl_dz * dz_dw1
    dl_dw2 = dl_dz * dz_dw2
    dl_db = dl_dz * dz_db
    
    # parameter update
    w0 = w0 - (dl_dw0 * learning_rate)
    w1 = w1 - (dl_dw1 * learning_rate)
    w2 = w2 - (dl_dw2 * learning_rate)
    b = b - (dl_db * learning_rate)
    
ax.plot(losses)