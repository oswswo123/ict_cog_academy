# %% basic_nodes

import numpy as np

class PlusNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
    
    def forward(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x + y
        return self.__z
    
    def backward(self, dz):
        return dz, dz
    

class MinusNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
        
    def forward(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x - y
        return self.__z
    
    def backward(self, dz):
        return dz, -dz
    
    
class MulNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
        
    def forward(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x * y
        return self.__z
    
    def backward(self, dz):
        return self.__y * dz, self.__x * dz
    
class SquareNode:
    def __init__(self):
        self.__x = None
        self.__z = None
    
    def forward(self, x):
        self.__x = x
        self.__z = x**2
        return self.__z
    
    def backward(self, dz):
        return 2 * self.__x * dz


class MeanNode:
    def __init__(self):
        self.__x = None
        self.__z = None
        
    def forward(self, x):
        self.__x = np.array(x)
        self.__z = np.mean(self.__x)
        return self.__z
    
    def backward(self, dz):
        dx = dz * (1 / len(self.__x) * np.ones_like(self.__x))
        return dx