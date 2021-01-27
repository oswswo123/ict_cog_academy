#%%
'''
def sun(x):
    pass     일 때 sun이 함수이다(sun(x)은 함수가 아님!)
             map(sun(x), [1, 2, 3, 4])로 쓰면 X
             map(sun, [1, 2, 3, 4])로 사용
'''
#%% callable
'''
    callable : ()를 붙일 수 있는지 없는지 확인하는 method
    ex) class A:
            pass
        callable(A) → True (A() 사용가능)
        
    클래스를 인스턴스화 할 때 '__init__()'가 실행된다
    인스턴스화 된 클래스에 ()를 붙이면 '__call__()'이 실행된다
'''
#%% type hint
'''
def sun(t:int) -> int:
    return 1
    
    sun(t:int) -> int는 type hint.
    t:int는 parameter hint로, 다른걸 넣어도 작동하지만 int를 주는것이 좋다고 알려줌
    -> int는 return hint로, return type이 int라는 것을 알려줌
'''
#%%
'''
    python은 ,로 연결해서 전달하면 tuple형태로 묶음
    ex) 1,2,3,4 → (1, 2, 3, 4)
    
    def moon(*a):  가변 방식
        return a
    
    moon(1, 2, 3, 4)는 (1, 2, 3, 4)를 return하게 됨
    
    
    def star(**a):  가변 키워드 방식
        return a
    
    star(x=1, y=3)은 {'x': 1, 'y': 3}을 return하게 됨
'''
#%%
'''
    First Class Function : 함수를 값처럼 사용가능 (값/데이터)
    
    closure : First class function의 개념을 이용하여
            scope에 묶인 변수를 바인딩하기 위한 기술

    LEGB
    - L : local의 약자로 함수 안을 의미
    - E : enclosing function locals의 약자로 내부함수에서 자신의 외부 함수의 범위를 의미
    - G : global의 약자로 함수 바깥, 즉, 모듈 범위를 의미
    - B : Built-in의 약자로 open, range와 같은 파이썬 내장 함수를 의미
    - L < E < G < B
    
    def sun(x):
        x = 3    # 이 범위가 enclosing
        def moon(y):
            return x+y
        return moon
    
    def sun():
        def moon():
            return 1
        return moon
    이 경우 sun()()은 moon()과 같다 (sun()이 moon이 되므로)
    그러나 moon은 local영역에 있으므로 밖에서 moon()으로 접근은 불가능. sun()()로 사용
    
    a = [len] 이라면
    a[0]([1, 2, 3])와 len([1, 2, 3])은 같다
    
    b = print 이라면
    b('abc')는 abc를 출력한다
    
    반대로 sum = 0와 같은 형태도 가능
    이 경우 built-in 영역에서 만들어진 객체 sum은 global 영역에서 integer 0이 된다
'''
#%% recursion (재귀)

# A(n+1) = A(n) + 2   ex) 1 3 5 7 9 11 
def a(x):
    if x==1:
        return 1
    return a(x-1) + 2

# B(n+1) = 2 * B(n)  ex) 1 2 4 8
def b(x):
    if x==1:
        return 1
    return 2 * b(x-1)