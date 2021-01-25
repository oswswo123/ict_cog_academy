import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_dataset(n_point, cutoff, noise_coeff):
    # cutoff set
    x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
    x_data_noise = x_data + noise_coeff * np.random.normal(loc=0, scale=1, size=(n_point, ))
    y_data = (x_data_noise > cutoff).astype(np.int)
    return x_data, y_data


def set_weight_bias():
    w, b = np.random.normal(loc=0, scale=1, size=(2, ))
    return w, b


def forward_propagation(x, y, w, b, losses):
    # z : affine function
    z = w * x + b
    # a : sigmoid (predictions)
    a = 1 / (1 + np.exp(-z))
    # loss : binary cross-entropy
    loss = -1 * ( y * np.log(a) + (1 - y) * np.log(1 - a))
    losses.append(loss)
    return a, losses


def backward_propagation(x, y, a):
    # Chain Rule
    # ∂l/∂w = (∂l/∂a) * (∂a/∂z) * (∂z/∂w)
    # ∂l/∂b = (∂l/∂a) * (∂a/∂z) * (∂z/∂b)
    dl_da = (a - y) / (a * (1 - a))
    
    da_dz = a * (1 - a)
    
    dz_dw = x
    dz_db = 1
    
    dl_dw = dl_da * da_dz * dz_dw
    dl_db = dl_da * da_dz * dz_db
    return dl_dw, dl_db


def parameter_update(w, b, dl_dw, dl_db, learning_rate):
    # parameter update
    w = w - (dl_dw * learning_rate)
    b = b - (dl_db * learning_rate)
    return w, b