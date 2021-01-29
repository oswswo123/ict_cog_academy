#%%
'''
    중급 파이썬 필수적 기법
    1. 함수형 패러다임(프로그래밍)
    2. 객체지향 패러다임
    
    재사용
    - 함수 (행위)
    - 객체 (데이터 + 행위) : 기본적으로 데이터는 접근 가능 → consenting adult(디스크립터)
    
    실제 사용시
    - 함수 : call / 호출
        → 함수 호출 : ()를 붙여서 사용 
    - 클래스 : instance
        → instance로 만들기 : ()를 붙여서 사용, attribute 참조 : . (ex. A.x)
    - 객체 지향의 기본은 인스턴스(객체/object) 사용
    
    PEP8
    - 함수 : snake // ex. moon_beauty
    - 클래스 : camel (capsword = pascal) // ex. MoonBeauty / moonBeauty
    
    객체지향의 4가지 기본 개념
    - Encapsulation
    - Inheritance
    - Abstraction
    - Polymophyism
'''
#%% 용어정리
class A:
    x = 1               # class attribute / variable
    def xx(self):       # instance method
        self.y = 1      # instance attribute / variable
        print('xx')

a = A()             # A를 instance화 시킴
print(type(a))      # a는 instance (type이 아닌 다른값이 print되면 instance)
print(type(A))      # A는 class (class면 type이 print된다)
print(type(int))    # int는 class이지만 기본 내장 데이터타입이므로 camel방식이 아님

#%%
class B:
    x = 1
    
b = B()
c = B()
print(b.x)      # 같은 이름이라면 instance attribute를 먼저 찾고,
                # 없으면 class attribute를 찾음
b.x = 2         # instance attribute 생성
print(B.x)      # B.x = 1
print(b.x)      # b.x = 2
print(vars(b))  # vars : 해당 객체가 가진 instance attribute를 보여줌
print(vars(c))  # c는 현재 instance attribute가 없다
                # instance attribute는 시점에 따라 달라진다!

#%%
class C:
    # '__new__()'는 값을 생성하는 method. return이 있음
    def __new__(cls):
        print('AAA')
        return super().__new__(cls)
    
    # '__init__()'은 값을 초기화하는 method. return이 없음
    def __init__(self):
        print('CCC')
    
    # '__new__()'와 '__init__()'가 함께 생성자의 역할을 수행한다
    
    def xx(self):   # xx는 instance가 사용하면 method, class가 사용하면 function
        print('xx')
        self.b = 2

c = C()

#%% 상속
'''
    상속이 왜 중요한가?
    - 재사용 : 있는 것 그대로 갖다 쓸 수 있다
              있는 것 내맘대로 변화시킬 수 있다 (overriding)
              - 완전히 바꾸거나
              - 일부만 바꾸거나
              없는 것 추가시킬 수 있다.
    - 파이썬의 상속은 복사가 아니라 위임(Delegation)이므로
      부모의 값이 변경되면 자식도 같이 변경될 위험이 있다.
'''
class B:    # 여기에 D(ojbect)가 생략되어 있음
            # 기본적으로 class는 object class를 상속받음
            # python은 다중 상속 언어이므로 2개 이상의 class로부터 상속받을 수 있음
    x = 2
    y = 1
    z = 3
    
    def zz(self):
        print('zz')
        
class D(B): # class D는 class B와 class object를 상속받음
    y = 11  # overriding
    
    def a(self):
        print('a')
    
    def zz(self):
        B.zz(self)
        print('re zz')
    
d = D()
print(dir(d))

#%% 상속 2
class A:
    def __init__(self):
        print('A')
        
class B(A):
    def __init__(self):
        # A.__init__(self)
        super().__init__()  # super()는 instance, A는 class
                            # super(B, self).__init__()와 같은 표현 (좀 더 명확함)
                            # B의 부모클래스의 '__init__()'을 찾아서
                            # instance인 self로 받음
        print('B')

class C(A):
    def __init__(self):
        # A.__init__(self)
        super().__init__()
        print('C')

class D(B, C):
    def __init__(self):
        # B.__init__(self)
        # C.__init__(self)    이렇게 사용하면 A가 두번 초기화됨
        super().__init__()  # 이렇게 사용하면 부모를 잘 찾아가 한번만 초기화시킨다
        print('D')

#%% overloading
'''
    overloading은 method overloading과 연산자 overloading이 있다
    - python은 method overloading을 지원하지 않음
    - 연산자 overloading은 지원
    
    def a():
        pass
    def a(x):
        pass
    
    이 2개를 다른 함수로 간주하는 것이 method overloading 지원
    같은 함수로 간주하는 것을 method overloading 지원 안함
    
    overloading은 보통 generic function의 역할을 한
    
    python의 모든 연산자는 method와 연결되어 있고
    python의 모든 연산자는 function으로 대체 가능하다
     → method와의 연결을 바꿈으로써 연산자의 기능을 바꿀 수 있다(연산자 오버로딩)
'''
# 연산자 오버로딩
class X:
    def __add__(self, a):
        print('연산자의 기능이 바뀌었습니다')
        return 1

x = X()

# + 연산자가 print로 바뀜(연산자의 기능이 재정의됨)
b = x+55
print(b)