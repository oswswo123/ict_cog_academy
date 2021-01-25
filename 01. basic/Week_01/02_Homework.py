# %% Homework 안내
'''
1. 3의 배수들의 합, 평균
2. 3으로 나누었을 때 0인 값, 1인 값, 2인 값들의 총합과 평균
'''
# %% HW 1

inputdata = 100

three_sum = 0
three_mean = 0
three_count = 0

for count in range(inputdata):
    if count % 3 == 0:
        if count != 0:
            three_sum += count
            three_count += 1

three_mean = three_sum / three_count
print(three_sum)
print(three_mean)

# %% HW 2

inputdata = 10

devide_zero = 0
devide_one = 0
devide_two = 0
zero_number_count = 0
one_number_count = 0
two_number_count = 0

for count in range(inputdata):
    if count % 3 == 0:
        if count != 0:
            devide_zero += count
            zero_number_count += 1
    elif count % 3 == 1:
        devide_one += count
        one_number_count += 1
    elif count % 3 == 2:
        devide_two += count
        two_number_count += 1

print("나머지가 0인 값의 의 총합, 평균 : ", devide_zero, devide_zero/zero_number_count)
print("나머지가 1인 값의 의 총합, 평균 : ", devide_one, devide_one/one_number_count)
print("나머지가 2인 값의 의 총합, 평균 : ", devide_two, devide_two/two_number_count)