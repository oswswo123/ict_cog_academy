# %% Class의 접근제어자

# python은 public, private가 없다
# 대신 _를 하나 붙이면 직접접근 하지말라는 암묵적인 표현
# method나 변수 앞에 __를 붙이면 접근을 회피할 수 있다
# 이를 name mangling(mangling - 뭉개기)이라고 한다
class TestClass:
    def __init__(self):
        # 가능한 직접접근 하지 말라는 뜻
        # 그러나 시스템적으로 가능
        self._a = 10
        # 얘는 __b가 아니라 _TestClass__b로 저장된다
        # 즉, 이름을 바꿔서 숨기는 것. 직접접근 자체를 차단하는 것은 아니다.
        self.__b = 20

test_object = TestClass()
print(test_object.a)
print(test_object.__dict__)

# %% Class의 Setter, Getter

import numpy as np

class SocreAnalyzer:
    def __init__(self, scores):
        self.__scores = scores
    
    def set_scores(self, scores): self.__scores = scores
    def get_scores(self): return self.__secores

# %% Class 실습 ( Setter, Getter 만들기 )

import numpy as np

class Score:
    def __init__(self, scores):
        self.__scores = scores
        
        # cal_M()으로 연산
        self.__max_value = None
        self.__max_idx = None
        
        # cal_m()으로 연산
        self.__min_value = None
        self.__min_idx = None
        
        # cal_mean()으로 연산
        self.__mean = None
        
        # cal_variance()로 연산
        self.__variance = None
        
        # cal_std()로 연산
        self.__std = None
        
        self.cal_all()
        
    def cal_M(self):
        for idx, value in enumerate(self.__scores):
            if self.__max_value == None or self.__max_value < value:
                self.__max_value = value
                self.__max_idx = idx       
    
    def cal_m(self):
        for idx, value in enumerate(self.__scores):
            if self.__min_value == None or self.__min_value > value:
                self.__min_value = value
                self.__min_idx = idx   
    
    def cal_mean(self):
        __sum = 0
        for value in self.__scores:
            __sum += value
        self.__mean = __sum / len(self.__scores)
          
    def cal_variance(self):
        self.cal_mean()
        self.__mean = self.get_mean()
        __squaredsum = 0
        for value in self.__scores:
            __squaredsum += value**2
        self.__variance = (__squaredsum / len(self.__scores)) - (self.__mean**2)
     
    def cal_std(self):
        self.cal_variance()
        self.__variance = self.get_variance()
        self.__std = np.sqrt(self.__variance)
    
    def cal_all(self):
        self.cal_M()
        self.cal_m()
        self.cal_mean()
        self.cal_variance()
        self.cal_std()
   
    def set_scores(self, scores): self.__scores = scores
    def get_scores(self): return self.__scores
    def get_M(self): return self.__max_value, self.__max_idx
    def get_m(self): return self.__min_value, self.__min_idx
    def get_mean(self): return self.__mean
    def get_variance(self): return self.__variance
    def get_std(self): return self.__std


scores = Score([1, 2, 3, 4, 5])
scores.set_scores([3, 4, 5, 6, 7])
print(scores.get_scores())

M, M_idx = scores.get_M()
m, m_idx = scores.get_m()
mean = scores.get_mean()
variance = scores.get_variance()
std = scores.get_std()
