# Logistic Regression 구현

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from logistic_regression_tool import create_dataset
from logistic_regression_tool import set_weight_bias
from logistic_regression_tool import forward_propagation
from logistic_regression_tool import backward_propagation
from logistic_regression_tool import parameter_update

n_point = 1000
noise_coeff = 0.1
cutoff = 1
learning_rate = 0.01
losses = list()

x_data, y_data = create_dataset(n_point, cutoff, noise_coeff)
w, b = set_weight_bias()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x_data, y_data, s=100, alpha=0.3)
cmap = cm.get_cmap('rainbow', lut=len(x_data))

for idx, (x, y) in enumerate(zip(x_data, y_data)):
    # forward
    a, losses = forward_propagation(x, y, w, b, losses)
    
    # backward
    dl_dw, dl_db = backward_propagation(x, y, a)
    
    # parameter update
    w, b = parameter_update(w, b, dl_dw, dl_db, learning_rate)
    
    # drawing graph
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_range = 1 / (1 + np.exp(-1 * (w * x_range + b)))    
    axes[0].plot(x_range, y_range, color=cmap(idx), alpha=0.3)
    
axes[1].plot(losses)
axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)