# %% Basic Building Nodes

import numpy as np


class PlusNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
    
    def forward_pass(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x + y
        return self.__z
    
    # dz는 ∂J / ∂z를 뜻함
    # 실제 구현시 도함수의 분모자리에 있는 값만 이름으로 입력
    def backward_pass(self, dz):
        return dz, dz
    
    
class MinusNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
    
    def forward_pass(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x - y
        return self.__z
    
    def backward_pass(self, dz):
        return dz, -1 * dz
    
    
class MulNode:
    def __init__(self):
        self.__x, self.__y = None, None
        self.__z = None
    
    def forward_pass(self, x, y):
        self.__x, self.__y = x, y
        self.__z = x * y
        return self.__z
    
    def backward_pass(self, dz):
        return self.__y * dz, self.__x * dz


class SquareNode:
    def __init__(self):
        self.__x = None
        self.__z = None
    
    def forward_pass(self, x):
        self.__x = x
        self.__z = x ** 2
        return self.__z
    
    def backward_pass(self, dz):
        return 2 * self.__x * dz


class MeanNode:
    def __init__(self):
        self.__x_vector = None
        self.z = None
    
    def forward_pass(self, __x_vector):
        self.__x_vector = np.array(__x_vector)
        self.__z = np.mean(self.__x_vector)
        return self.__z
    
    def backward_pass(self, dz):
        dx = dz * 1 / len(self.__x_vector) * np.ones_like(self.__x_vector)
        return dx
