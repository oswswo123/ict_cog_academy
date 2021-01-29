#%% 타입 컨버전

# 명시적 방법(연산자)은 없음 > 새로운 객체를 생성
# 묵시적 방법 coerce

#%% Sequence
'''
    Sequence 3가지
    1. list (ex. a = [1, 2, 3])
    2. tuple (ex. b = (1, 2, 3))
    3. range (ex. c = range(10))
    
    Sequence의 특징
    - Indexing과 Slicing이 지원된다
    - 이 기능은 '__getitem__()', '__len__()'을 통해 지원된다
    - '__getitem__()', '__len__()'이 있다면 duck-typing에 의해 Sequence가 된다
'''
#%% 
'''
    함수형 패러다임의 중요한 3가지 package
    - functools
    - itertools
    - operator
'''
#%% composition

# composition 방식 : 다른 객체를 내 class의 정의 안에 포함시키는 것
class A:
    def x(self):
        print('x')
    
    def y(self):
        print('y')

# 상속을 사용        
class B(A):
    def y(self):
        super().y()
        print('B\'s y')

# 상속을 composition 방식으로 대체
# 상속의 문제를 어느정도 해결할 수 있음
class C:
    def __init__(self):
        self.__a = A()
        
    def x(self):
        self.__a.x()
    
    def y(self):
        self.__a.y()
        print('C\'s y')

b = B()
c = C()

b.x()
c.x()
b.y()
c.y()
print()

# '__getattr__()' : Attribute error가 발생하면 실행됨
# Attribute error는 해당 class에 그 attribute가 없으면 발생
# '__getattr__'과 composition방식을 활용해 상속과 유사한 기능을 구현 가능
# 참고로 '__getattr__'과 getattr()은 다르다 (getattr()은 hasattr()과 유사함)
class D:
    def __init__(self):
        self.__a = A()
    
    def __getattr__(self, attr):
        print('error', attr)
        return getattr(self.__a, attr)

d = D()
d.x()
d.y()

#%% property를 활용한 '__get__', '__set__' 변경 (descriptor)

# descriptor : '__get__', '__set__', '__del__'중 하나라도 정의되어 있으면 디스크립터
# 디스크립터는 인스턴스의 속성이 아닌 클래스의 속성으로 정의되어있어야 함

# composition 방식을 이용하여 descriptor를 만들고 get과 set을 customize
class X:
    def __init__(self, x):
       self.x = x
   
    def __get__(self, a, b):
        print('get')
        return self.x
    
    def __set__(self, a, val):
        print('set')
        self.x = val

# descriptor를 만들기 위해 property 객체를 활용
class C:
    def __init__(self, x):
        self._x = x
    
    def getx(self):
        print('get')
        return self._x
    
    def setx(self, value):
        print('set')
        self._x = value
    
    def delx(self):
        del self._x
    
    x = property(getx, setx, delx, "i'm the 'x' property.")

# property를 이용하여 special method '__get__'과 '__set__'을 변경
# 이를 응용하면 맹글링과 다르게 실제로 접근을 제한할 수 있음 (private처럼)
# ex) getx의 return self._x를 삭제하면 get을 차단할 수 있다
c = C(3)
c.x
c.x = 10
print()

# decorator인 property를 활용하여 descriptor 만들기
class D:
    def __init__(self, x):
        self._x = x
    
    # @property를 단독으로 쓰면 아래에 작성된 함수인 x를 괄호없이 쓰게해줌
    @property
    def x(self):
        print('xxxx')
    
    # 위의 property와 함수명이 같아야 함 (여기서는 함수명이 x이므로 x.setter)
    # name.setter, name.getter 등의 decorator를 활용하여 여러 기능을 customize 할 수 있음
    @x.setter           # setter를 customize
    def x(self, a):
        print('set')
        self._x = a
    
    @x.getter           # getter를 customize
    def x(self):
        print('get')
        return self._x

d = D(5)

# 본래는 d.x()로 사용해야함
# decorator인 @property를 붙였더니 괄호를 사용할 수 없음
d.x

# @x.setter를 통해 setter를 customize
d.x = 10

#%% 추상화(abstract)

# abc는 abstract base class의 약자
# 공통된 부분을 구현하지 않고 추상적으로 정의해둠. 구체화가 필요함
from abc import ABCMeta, ABC, abstractclassmethod

# metaclass로 ABCMeta를 지정하는 방식
# 조금 더 유연하게 만들 수 있음
class MyABCMeta(metaclass=ABCMeta):
    pass

# 그냥 class로 ABC를 상속받는 방식
# MyABC의 x는 추상화메소드라 MyABC를 인스턴스화 할 시 error가 발생함
# 한번 더 상속받아서 x를 구현해야 사용 가능
class MyABC(ABC):
    @abstractclassmethod
    def x(self):
        print('aaa')

# a = MyABC()  # error 발생!

class MyABC2(MyABC):
    def x(self):
        print('aaaaa')

x = MyABC2()    # MyABC2로 상속 받아 추상화 메소드를 구현했기에 사용가능
x.x()