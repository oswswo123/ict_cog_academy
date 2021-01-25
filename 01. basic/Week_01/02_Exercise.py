# 변수들이 저장되는 위치

a = 10
b = a

print(id(10))
print(id(a))
print(id(b))

# %% Python은 변수가 할당되는 순간 자료형이 정해짐

a = 10
print(a, type(a))

a = 10.10
print(a, type(a))

a = "Hello World!"
print(a, type(a))

a = [1, 2, 3, 4, 5]
print(a, type(a))

# %% 복습 (평균과 분산 구하기)

scores = [10, 20, 30]
score_sum = 0
score_sqaresum = 0

for score in scores:
    score_sum += score
    score_sqaresum += score ** 2

mean = score_sum / len(scores)
var = (score_sqaresum / len(scores)) - (mean ** 2)

print(mean)
print(var)

# %% 복습2 (math, english 평균 각각 구하기)

math_scores = [10, 20, 30, 40, 50]
math_mean = 0
english_scores = [20, 30, 40, 50, 60]
english_mean = 0

score_sum = 0
for math_score in math_scores:
    score_sum += math_score
math_mean = score_sum / len(math_scores)

score_sum = 0
for english_score in english_scores:
    score_sum += english_score
english_mean = score_sum / len(english_scores)

print(math_mean, '\n', english_mean, sep='')

# %% range() : type은 range이다.

print(range(10))
print(type(range(10)))

# %% print문의 sep='' : searator

print(10, 20, 30, 40, 50)
print(10, 20, 30, 40, 50, sep=' --- ')

# %% 별 찍기 (문자열의 연산)

for i in range(5):
    print(' ' * (4-i) + '*' * (i+1))

# %% list의 연산(Concatenate)과 Vector 연산

test_list1 = [1, 2, 3]
test_list2 = [4, 5, 6]

print(test_list1 + test_list2)

import numpy as np

scores1 = np.array([1, 2, 3])
scores2 = np.array([4, 5, 6])

print(scores1 + scores2)

# %% 비교 연산자 ( <, >, <=, >=, ==, != )

a, b = 10, 20
print(a > b)
print(a < b)
print(a == b)

# %% type casting

a = 10
b = float(a)
c = str(a)
d = bool(a)

print(type(a), type(b), type(c), type(d))

# %% Boolean data type

# int, float -> bool
# 0은 False, 나머지는 모두 다 True
a = 3.0
print(bool(a))

a = 0.0
print(bool(a))

a = -3.0
print(bool(a))

# list -> bool
# list가 비어있으면 False, 나머지는 모두 다 True
a = [1, 2, 3]
print(bool(a))

a = []
print(bool(a))

# %% if문

score = 30

if score > 60:
    print("합격!")
elif score > 40 and score <= 60:
    print("재시험!")
else:
    print("불합격!")

# %% if문 연습 (학점매기기)
'''
학점 매기기
90 ~ 100 : A
70 ~ 89 : B
50 ~ 69 : C
0 ~ 49 : F
'''
score = 89

if score >= 90 and score <= 100:
    print("A")
elif score >= 70 and score < 90:
    print("B")
elif score >= 50 and score < 70:
    print("C")
elif score >= 0 and score < 50:
    print("F")
else:
    raise ValueError
'''
raise는 고의로 예외를 발생시키는 함수
ValueError는 값의 오류를 뜻함
'''

# %% if문 연습2 (학점매기기 2)

score = 89

if score >= 90 and score <= 100:
    if score >= 95 and score <= 100:
        print("A+")
    else:
        print("A0")
elif score >= 70 and score < 90:
    if score >= 80 and score <= 89:
        print("B+")
    else:
        print("B0")
elif score >= 50 and score < 70:
    if score >= 60 and score <= 69:
        print("C+")
    else:
        print("C0")
elif score >= 0 and score <= 49:
    print("F")
else:
    raise ValueError

# %% if문 연습3 (큰 수 출력하기, 같을때는 "같은숫자다")

a, b = 30, 40

if a > b:
    print(a)
elif a == b:
    print('equal')
else:
    print(b)
    
# %% if문 연습4 (양수인지 음수인지 판별하기)

a = 32

if a > 0:
    print('positive')
elif a == 0:
    print('zero')
else:
    print('negative')

# %% if문 연습5 (절댓값 출력하기)

a = -11

if a < 0:
    print(-a)
else:
    print(a)

# %% if문 연습6 (일정금액이상 결제 할인)

price = 15000

if price >= 30000:
    print(price * 0.85)
elif price >= 20000:
    print(price * 0.9)
elif price >= 10000:
    print(price * 0.95)
else:
    print(price)

# %% if문 연습7 (연산 기호에 따라 값 출력하기)

a, b = 10, 20
operator = '+'

if operator == '+':
    print(a+b)
elif operator == '-':
    print(a-b)
elif operator == '*':
    print(a*b)
elif operator == '/':
    print(a/b)
else:
    print('Value Error!')
    raise ValueError
    
# %% if문 연습8 (짝수 홀수 구분하기)
a = 0

if a % 2:
    print('Odd')
else:
    if a:
        print('Even')        
    else:
        print('Zero')
        
# %% if문 연습9 (우수반 구분)
'''
 반 평균이 70점 이상이면 우수반,
 50 ~ 69이면 중급반, 
 아니면 보충반
'''
scores = [10, 30, 60, 80, 20, 50]
score_sum = 0
score_mean = 0

for index in range(len(scores)):
    score_sum += scores[index]
score_mean = score_sum / len(scores)

print(score_mean)
if score_mean >= 70:
    print('우수반')
elif score_mean >= 50 and score_mean <= 69:
    print('중급반')
else:
    print('보충반')
        
# %% if문 연습10 (홀수들의 합, 짝수들의 합 구하기)

odd_sum = 0
even_sum = 0

for i in range(100):
    if i % 2:
        odd_sum += i
    else:
        even_sum += i

