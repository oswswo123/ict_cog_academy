# %% 0점부터 20점까지 몇명씩 있나?

import numpy as np

score_min, score_max = 10, 20
scores = np.random.uniform(score_min, score_max, size=(300, )).astype(np.int)

score_dict = { i:list() for i in range(score_min, score_max) }
scores_cnt = list()
'''
list만 사용해서 구현
for idx in range(score_range):
    scores_cnt.append(0)
    
for score in scores:
    scores_cnt[score] += 1
'''
for score in scores:
    score_dict[score].append(score)

for dict_idx, _ in score_dict.items():
    scores_cnt.append("{}은 {}번".format(dict_idx, len(score_dict[dict_idx])))

print(scores_cnt)

# %% dict를 이용한 팁 ( dict.get(x, y) )

import numpy as np

score_min, score_max = 10, 20
scores = np.random.uniform(score_min, score_max, size=(300, )).astype(np.int)

score_cnt = dict()

for score in scores:
    # dict.get(x, y) : dict에 해당 키가 있으면 x의 value를, 없으면 y를 return
    score_cnt[score] = score_cnt.get(score, 0) + 1

# string이나 다른 객체에서도 사용 가능
test_string = 'nfeioafjdksljzkfldsjanvklsdjfkls;;;;;;'
chr_cnt = dict()
for charater in test_string:
    chr_cnt[charater] = chr_cnt.get(charater, 0) + 1

print(score_cnt, '\n')
print(chr_cnt)

# %%
''' function
1. 함수(test_function)을 정의하는 순간 test_function이라는 object가 생긴다.
2. object는 data와 method을 가진다.
3. 함수는 data와 method를 가진다
4. data나 method에 접근하는 방법은 object.data, object.method
'''
# %% function에 대한 팁 (default argument)

# constant의 input이 있으면 그 값을 사용
# constant의 input이 없으면 constant로 2를 사용
def multiply_constant(input_list, constant=2):
    # 99%는 constant로 2를 사용하는데, 아주 가끔 다른 constant를 사용할 때
    # 항상 2를 써주기 귀찮음!
    for input_ in input_list:
        print(input_*constant)

score_min, score_max = 10, 20
score = np.random.uniform(score_min, score_max, size=(300, )).astype(np.int)

multiply_constant(score)

# %% function의 argument 정리

# *args, **kwargs는 해당 부분의 input이 여러개라는 것을 암시함
# 이 *에 대한 의미가 python과 C는 명백히 다름!

def test_funct(a, b):
    print(a, b)

test_funct(10, 20)      # key를 안정해주면, 순서대로 들어감
test_funct(a=10, b=20)  # argument를 마치 dictionary처럼 넣어줌 (이게 더 명시적인 방법)

# %% *args

def test_funct(a, b):
    print(a, b)

test_input = [1, 2]
test_funct(*test_input) # 이렇게 넣으면 알아서 unpacking 해줌

# %% **kwagrs

def test_funct(a, b):
    print(a, b)
test_input = {'a':10, 'b':20}
test_funct(*test_input)     # 이렇게 넣으면 key를 unpacking
test_funct(**test_input)    # 이렇게 넣으면 key의 value까지 매칭시켜서 unpacking

# %% *args로 함수를 packing

# parameter에 *를 넣으면 packing
# input data가 여러개라는 것을 암시
# 들어온 input들을 tuple로 묶음
# tuple이니 순서가 필요
# *args는 일반 parameter 뒤에 써야함(python은 어디부터 어디까지가 *args인지 모름)
def test_funct(*a):
    print(type(a))
    print(a)

test_input = [10, 20, 30]

test_funct(10, 20, 30)

test_funct(test_input)
test_funct(*test_input)

# %% **kwargs

# parameter에 **를 넣어도 packing
# input data가 여러개라는 것을 암시
# 단, **kwargs는 input들을 dict로 묶음
# dict이니 순서는 필요 X, 단 key와 value의 set구조
# **kwargs는 제일 뒤에 써야함(**kwagrs가 *args 뒤에 오는것은 python의 syntex)
def test_funct(**input_):
    print(type(input_))
    print(input_)

test_funct(a=10, b=20)

test_input = {'a':10, 'b':20}
test_funct(**test_input)

# *args, **kwagrs는 둘다 만들기 어렵고 헷갈린다...
# 사실 직접 만들어 쓸 일보단 유명한 module의 document를 볼 때 많이 사용

# %% function 내부에서 전역 변수 호출 ( global )

accumulate_a = 0
accumulate_b = 0
accumulate_ab = 0
def get_sum(a, b):
    # global로 지정된 변수들은 전역변수임을 나타낸다
    global accumulate_a, accumulate_b, accumulate_ab
    data_sum = a + b
    accumulate_a += a
    accumulate_b += b
    accumulate_ab += a + b
    print(data_sum, accumulate_a, accumulate_b, accumulate_ab)

get_sum(a=10, b=20)
get_sum(a=20, b=40)
get_sum(a=1000, b=2000)
get_sum(a=10, b=20)
get_sum(a=-10, b=-100)

# %% function 내부에 data를 넣기

def get_sum(a, b):
    get_sum.a_sum += a
    get_sum.b_sum += b
    print(a, b)

get_sum.a_sum = 0
get_sum.b_sum = 0

get_sum(10, 20)
print(get_sum.a_sum, get_sum.b_sum)
get_sum(20, 40)
print(get_sum.a_sum, get_sum.b_sum)
get_sum(a=1000, b=2000)
print(get_sum.a_sum, get_sum.b_sum)
get_sum(a=10, b=20)
print(get_sum.a_sum, get_sum.b_sum)
get_sum(a=-10, b=-100)
print(get_sum.a_sum, get_sum.b_sum)
