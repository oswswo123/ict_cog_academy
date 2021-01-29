#%% decorator(데코레이터)
'''
    ## closure
    def x(m):
        def y(n):
            return m+n
        return y
    - closure는 함수를 parameter로 받지 않음
    
    decorator
    - 함수를 parameter로 받아 명령을 추가한 뒤 이를 다시 함수의 형태로 반환하는 함수
    - 함수의 내부를 수정하지 않고 기능에 변화를 주고싶을 때 사용
    - 일반적으로 함수의 전처리나 후처리에 대한 필요가 있을 때 사용
    - 생각하기 1) 기존에 있는 함수를 변경하기 위해서 어떻게 decorator를 작성할지
               2) 남이 만든 decorator를 어떻게 사용할지
'''
# decorator (합성함수와 유사함)
def my_decorator(function):
    def inner():
        print('- 전처리 -')
        function()
        print('- 후처리 -')
    return inner

def a():
    print('sun')

my_decorator(a)()

# 원래 만들어둔 함수를
# @my_decorator를 이용하여 내부적으로 변경함
# 이렇게 작성할 경우 함수 a를 실행하면
# a가 parameter로 my_decorator에 들어가서 새로운 기능을 하게 됨
@my_decorator
def a():
    print('sun')

a()

#%% decorator + argument

def my_decorator(fun):
    # decorator에 전달하는 함수의 parameter가 가변적일 경우
    # argument를 통해 인자의 상황과 관계없이 사용할 수 있다
    def inner(*args, **kwargs):
        print('- 전처리 -')
        fun(*args, **kwargs)
        print('- 후처리 -')
    return inner

# print는 parameter가 반드시 1개 필요
my_decorator(print)('abc')

# tt는 parameter가 없음
@my_decorator
def tt():
    print('aaaaa')

tt()
# inner의 parameter로 argument와 keyword argument를 사용함으로써
# 여러 상황에 대처 가능

#%% @classmethod, @staticmethod, @property

# class내에서 자주쓰는 decorator로
# @classmethod, @staticmethod, @property가 있다
class A:
    # decorator로 @classmethod를 붙이면
    # instance method가 아니라 class method가 된다
    @classmethod
    def z(self):
        self.x = 1
        print('class method')
        
class B:
    # decorator로 @staticmethod를 붙이면
    # 이름으로만 접근이 가능하다
    @staticmethod
    def zz(self):
        self.a = 1
        print('aaa')
    
    # 접근을 차단함
    @property   # descriptor
    def zzz(self):
        pass

#%% singledispatch

from functools import singledispatch

@singledispatch
def x(t):
    print('single dispatch')

# decorator가 내부적으로 이름을 다르게 저장해 다른 함수가 되므로
# 작성할때는 함수의 이름이 같아야 함
# method overloading과 유사
@x.register(int)
def _(t):
    print('int')

@x.register(str)
def _(t):
    print('str')

# parameter가 int일때는 int를, str일때는 str을 출력함
# overloading과 유사한 기능을 보여줌
x(3)
x('aaa')

#%% wraps

from functools import wraps

# @wraps는 decorator를 만들기 위한 decorator이다
# @wraps을 주석처리하고, 주석을 없애가면서
# 콘솔창에 y(xxxx)나 xxxx를 입력해 차이를 비교해보자
def y(fun):
    @wraps(fun)
    def z():
        print('- 전처리 -')
        fun()
        print('- 후처리 -')
    return z

@y
def xxxx():
    print('xxxx')

#%% 괄호가 붙은 decorator

# 데코레이터에 괄호가 붙어서 parameter를 받아야한다면
# decorator를 한번 더 감싸서 parameter를 받아야함

from functools import partial

# 인자를 사용하기 위한 3단 구조
def zzz(m):
    def my_decorator(fun):     # decorator
        def inner():
            print(m)
            fun()
            print(m)
        return inner
    return my_decorator

# @zzz에 (parameter)를 붙이지 않으면 error가 발생한다
@zzz(3)
def a():
    print('sun')

a()
print()

# partial이란?
def add(x, y):
    return x + y

# partial을 통해 인자가 없어도 default를 설정할 수 있음
add2 = partial(add, x=1, y=2)
print(add2())           # parameter가 없어도 default로 x는 1, y는 2가 전달
print(add2(x=3, y=5))   # keyword방식을 통해 parameter를 전달하는 것도 가능
print()

# @my_decorator에 partial을 이용하여
# 인자가 없어도 자동으로 입력하도록 한다(default 설정)
# 이를 통해 괄호가 있어도, 없어도 기능하도록 만들 수 있음
def my_decorator(func=None, *, x=2):
    if func == None:
        # partial을 통해
        # my_decorator의 x는 1이 되도록 설정
        return partial(my_decorator, x=1)
    def inner():
        print('-', x, '-')
        func()
    return inner

# 뒤에 괄호를 붙이면 x는 1이, 붙이지 않으면 x는 2가 된다
@my_decorator
def x():
    print('aaa')

x()

#%% type의 활용
'''
    type은 언제쓰는가?
    1. 해당 객체의 type을 확인할 때(type이 출력되면 class, 아니면 instance)
    2. lambda함수처럼 재사용 하지 않을 class를 생성할때 type을 통해 class를 생성 가능
    3. metaclass
'''
#%% meta class
'''
    class B:    # class B:에는 class B(object, metaclass=type):가 숨어있음
        pass
'''
# class의 class를 meta class라 함
# meta class는 class의 행동을 제한할 수 있음
# python은 meta class를 이용하여 추상화(Abstraction)개념을 활용한다

print(type(int))            # int는 class
print(type(int.__class__))  # __class__는 meta class
print()

# metaclass를 생성
class Singleton(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__instance = None
    
    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance

class C(metaclass=Singleton):
    pass

# class A를 3번 인스턴스화 한 것 같지만
# 사실 a, b, c는 같은 memeory를 사용한다
# 이를 singleton이라 함
a = C()
b = C()
c = C()

print(id(a), id(b), id(c))

# 그래서 b의 attribute를 바꾸었는데, c의 attribute가 같이 바뀜
b.att = 123
print(c.att)