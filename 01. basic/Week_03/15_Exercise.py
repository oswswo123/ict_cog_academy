# %% Class

import numpy as np

# a와 b는 data는 다르지만 method는 같음
# 이 둘은 numpy.ndarray라는 class에 속함
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(type(a))
print(type(b))

# %% Class 만들기
'''class
- class는 일종의 빵틀과 같다.
- 비슷한 형태의 데이터들과
- 같은 method를 공유하는 객체들을 여러개 만들어야 할때 유용하다.
'''
# create function
def test_function():
    pass

# create class
class TestClass:
    def __init__(self):
        self.x = np.random.uniform(-10, 10, (1, ))
        self.y = np.random.uniform(-10, 10, (1, ))
    
    def add(self):
        print(self.x + self.y)

a = TestClass()
print(a.x, a.y)
a.add()
b = TestClass()
print(b.x, b.y)
a.add()

# %%

# Class명으로 _를 쓰지않는다
# Camel Case로 사용하는 것이 pep8의 Rule
# 각 단어의 첫글자를 대문자로 사용하자
class Point:
    def __init__(self):
        ''' special method
        - __가 앞뒤로 붙은 method는 사람을 위한 method가 아니라
        - 컴퓨터를 위한 method이다
        - special method는 객체를 만들어낼 때 자동적으로 컴퓨터에서 모두 만들어 준다
        '''
        self.x, self.y = np.random.uniform(-10, 10, (2, ))
    
    def get_sum(self):
        return self.x + self.y
    
a = Point()
print(a.x, a.y)
print(a.get_sum())
b = Point()
print(b.x, b.y)
print(b.get_sum())

# %% __init__ method의 parameter

class Point:
    # self는 만들어진 객체 자신을 가리킨다.
    # 즉, 사용되는 인스턴스 자기자신을 가리키는 명령어 (java의 this와 같음)
    # java와 다르게 input parameter가 없더라도 self는 꼭 넣어줘야 함
    def __init__(self, x, y):
        # __init__ method는 객체를 만들 때 자동으로 실행된다 (initializer)
        # 여기에 들어온 parameter를 이용해 객체를 생성할 수 있다
        self.x = x
        self.y = y
        
    # pep8에서는 class내의 method들울 한 줄씩만 띄운다고 말한다
    def get_coordinate(self):
        return [self.x, self.y]
    
    def move_right(self):
        self.x += 1

a = Point(x=10, y=20)
coordinate = a.get_coordinate()
print(coordinate)

a.move_right()
coordinate = a.get_coordinate()
print(coordinate)

# %% Class 실습

import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y, ax):
        self.x = x
        self.y = y
        self.ax = ax
        self.ax.scatter(self.x, self.y)
    
    def get_cor(self):
        return self.x, self.y
    
    def move_up(self):
        self.y += 1
        self.ax.scatter(self.x, self.y)
    
    def move_down(self):
        self.y -= 1
        self.ax.scatter(self.x, self.y)
    
    def move_right(self):
        self.x += 1
        self.ax.scatter(self.x, self.y)
        
    def move_left(self):
        self.x -= 1
        self.ax.scatter(self.x, self.y)


fig, ax = plt.subplots(figsize=(20, 12))

x, y = 10, 10

ex_point = Point(x, y, ax)

ex_point.move_up()
ex_point.move_left()
ex_point.move_down()
ex_point.move_down()
ex_point.move_left()
ex_point.move_down()

# %% Class 실습 2 (numpy 사용 X)
'''
scores = [1, 2, 3, 4, 5]
M, M_idx = scores.get_M()
m, m_idx = scores.get_m()
mean = scores.mean()
variance = scores.get_variance()
'''
class Score:
    def __init__(self, scores):
        self.scores = scores
        
    def get_M(self):
        self.max_value = None
        self.max_idx = None
        for idx, value in enumerate(self.scores):
            if self.max_value == None or self.max_value < value:
                self.max_value = value
                self.max_idx = idx
        return self.max_value, self.max_idx
    
    def get_m(self):
        self.min_value = None
        self.min_idx = None
        for idx, value in enumerate(self.scores):
            if self.min_value == None or self.min_value > value:
                self.min_value = value
                self.min_idx = idx
        return self.min_value, self.min_idx
    
    def get_mean(self):
        self.sum_ = 0
        for value in self.scores:
            self.sum_ += value
        return self.sum_ / len(self.scores)
        
    def get_variance(self):
        self.mean = self.get_mean()
        self.squaredsum = 0
        for value in self.scores:
            self.squaredsum += value**2
        return (self.squaredsum / len(self.scores)) - (self.mean**2)


scores = Score([1, 2, 3, 4, 5])
M, M_idx = scores.get_M()
m, m_idx = scores.get_m()
mean = scores.get_mean()
variance = scores.get_variance()

# %% special method (method overriding)

# special method는 python의 기본적인 동작들
# ex) +, -, * ...

a = [1, 2, 3]
b = [10, 20, 30]

class Vector:
    def __init__(self, input_list):
        self.input_list = input_list
    
    # +는 python이 __add__ method를 call하는 special method
    # 우리는 class에서 이 __add__를 override 할 수 있다
    # = 우리가 직접 + 연산을 정의할 수 있다는 의미
    def __add__(self, v2):
        added_list = list()
        for idx in range(len(self.input_list)):
            added_list.append(self.input_list[idx] + v2.input_list[idx])
        return added_list
    
    # class의 객체의 이름을 call하면 __str__ method가 call됨
    def __str__(self):
        return str(self.input_list)
        
v1 = Vector(a)
print(v1.input_list)

v2 = Vector(b)
print(v2.input_list)

print(v1 + v2)
print(v1)
