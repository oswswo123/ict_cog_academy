import numpy as np


def create_dataset(n_point, t_w, t_b, noise):
    # target weight 와 target bias set
    x_data = np.random.normal(loc=0, scale=1, size=(n_point, ))
    y_data = t_w * x_data + t_b + \
                noise*np.random.normal(loc=0, scale=1, size=(n_point, ))
    return x_data, y_data


def set_weight_bias():
    w, b = np.random.normal(loc=0, scale=1, size=(2, ))
    return w, b


def forward_propagation(x, y, w, b, losses):
    # affine function = z
    z = w * x + b
    loss = (y - z)**2
    losses.append(loss)
    return z, losses
    

def backward_propagation(x, y, z):
    # 편미분을 하고, Chain Rule을 이용하여 과정을 조금 더 세분화
    dl_dz = -2 * (y - z)
    
    dz_dw = x
    dz_db = 1
    
    dl_dw = dl_dz * dz_dw
    dl_db = dl_dz * dz_db
    return dl_dw, dl_db


def parameter_update(w, b, dl_dw, dl_db, learning_rate):
    # parameter를 update할때는 반드시 gradient에 -1을 곱해서!
    w = w - ( dl_dw * learning_rate )
    b = b - ( dl_db * learning_rate )
    return w, b