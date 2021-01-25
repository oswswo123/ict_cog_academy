# %% Learning Code

import numpy as np
import basic_nodes as nodes


class AffineFunction:
    def __init__(self, feature_dim, th_list): pass
    def affine_imp(self): pass
    def forward(self, X): pass
    def backward(self, dz2_last, lr): pass
    

class SquareErrorLoss:
    def __init__(self): pass
    def loss_imp(self): pass
    def forward(self, y, pred): pass
    def backward(self): pass