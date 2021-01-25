#%% 
'''
    함수형 프로그래밍(패러다임)
    - 멀티 패러다임 언어 (python) : 함수(수학)를 기본으로 프로그래밍하는 방식
                        → 형식적 증명 가능성, 모듈성, 디버깅과 테스트 용이성, 결합성
                        
    불변성(Immutable) : 명확한 조건 없이 값이 스스로 바뀌지 않음
'''
#%% iterator
'''
container : 여러개의 값을 가지는 객체(list, dictionary 등)
iterable : iterator가 될 수 있는, 순회 가능할 수 있는, 한개씩 뽑을 수 있는 속성
        → container의 element를 하나씩 순서대로 뽑음
iterator : 데이터의 스트림을 표현하는 객체. iterator의 '__next__()'를 반복하여 호출하면
    스트림에 있는 항목들을 차례대로 돌려줌
    lazy기법을 사용(호출할때만 memory에 올라감 → 효율적으로 memory 관리)

dir(x) : 객체의 attribute를 listing하는 명령어
'''

a = [1, 2, 3]
b = iter(a)     # list_iterator
print(next(b))  # iterator는 next를 사용할 수 있음
# dir(a)를 했을 때 '__iter__'이 있으면 iterable함 → 덕타이핑(duck-typing) 특성

#%% comprehension
'''
    Comprehension / Comp : 3가지 종류가 있다.
    - 여러개의 값을 동시에 생성하거나 변환시킬 때 사용
    - 최적화되어 있어서 속도가 빠름
    - 문이 아니라 식으로 사용
    - 식 : 하나의 결과값으로 축약할 수 있음 (ex. 1 + 1 은 2로 축약 가능)
    - 사용방법
        1. [x+1 for x in range(10)]
            → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        2. [x for x in ragne(10) if x%2==0]
            → [0, 2, 4, 6, 8]
        3. [x if x%2==0 else '' for x in range(10)]
            → [0, '', 2, '', 4, '', 6, '', 8, '']
        4. [(x, y) for x in range(10) for y in range(3)]
            → 30개의 element를 가진 list가 생성됨
        5. {x:x+2 for x in range(5)}
            → {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}인 dict
'''
#%%
def y():
    pass
    # function이 pass를 통해 return이 없다면
    # python은 자동적으로 None을 return한다
    # return None이 붙어있는 셈
    
a = y()
print(type(a))

#%%
def z():
    yield 11
    yield 2
    return 3333

b = z()

# yield가 들어가므로 b는 function이 아니라 generator가 된다
print(type(b))

# generator는 iterator처럼 next를 사용하면 yield를 순서대로 하나씩 뽑아온다
# 더이상 뽑아올 수 없다면 error message인 StopIteration으로 return값을 출력한다 
print(next(b))
print(next(b))
print(next(b))

#%% map
'''
    hiegher-order function : 함수를 인자로 받거나 함수를 리턴할 수 있는 함수 (ex. map)
    
    map은 iterable함 (compresion과 유사한 일을 한다)
    map(lambda x:x+1, range(10))으로 사용    
    dataset이 작을때는 compresion이, 클때는 map이 유리하다 (lazy기법때문에)
'''
# map의 첫번째 parameter는 function
# map의 두번째 parameter는 iterable 해야함 (ex. range(10), [1, 2, 3, 4], 'abc')
# 두번째 parameter를 순서대로 하나씩 뽑아서
# 첫번째 parameter인 함수식에 적용해 map 객체를 생성한다
a = map(lambda x : 2*x, range(10))
b = list(a)
print(type(a), type(b))

#%% filter
'''
    predicate function : boolean값을 return하는 함수 (보통 is, able등의 단어가 들어감)
    
    filter : predicate function이 true인 값만 필터링하는 함수
    filter의 첫번째 parameter는 predicate function이어야 한다
    filter의 두번째 parameter는 iterable 해야한다
    filter(labmda x:x>2, [1,2,3,4,5])등의 형태로 사용
'''
a = filter(lambda x : x>2, [1, 2, 3, 4, 5])
b = list(a)
print(type(a), type(b))

#%% reduce

from functools import reduce

# reduce : 하나의 결과값으로 축약하는 함수
a = reduce(lambda x,y : x+y, [1, 2, 3, 4, 5])
print(a)