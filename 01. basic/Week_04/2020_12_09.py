# 이상적인 loss graph와 실제 graph의 차이

# 실제로는 input의 크기가 커지면 해당 parameter의 민감도가 높아진다
# 즉, θ₁의 계수 x₁ 엄청 크면 다른 parameter θ들은(ex. θ₀는 계수가 1) 학습이 잘 안됨
# why? (y - θx)², θ = (θ₁, θ₀), x = (x₁, 1)을 풀면
# x₁²θ₁² + θ₀² + 2xθ₁θ₀ + ... 의 형태가 됨
# θ₁²의 계수가 x₁²이므로, x₁의 크기에따라 θ₁는 θ₀에 비해 민감해짐 (normalization 필요)
# 2xθ₁θ₀는 계곡선을 회전시킴
# 그래서 실제 data의 그래프는 찌그러진 형태

import numpy as np
import matplotlib.pyplot as plt

th1 = np.linspace(-10, 10, 100)
th0 = np.linspace(-10, 10, 100)

x, y = 2, 3

Th1, Th0 = np.meshgrid(th1, th0)

predictions = Th1*x + Th0
loss = (y - predictions)**2

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(Th1, Th0, loss)
ax.set_xlabel(r'$\theta_{1}$', fontsize=20)
ax.set_ylabel(r'$\theta_{0}$', fontsize=20)
th1 = np.linspace(-10, 10, 100)
th0 = np.linspace(-10, 10, 100)

Th1, Th0 = np.meshgrid(th1, th0)
x = 3

L = (x**2)*(Th1**2) + Th0**2
L = L + 2*x*Th1*Th0

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(Th1, Th0, L,
                  color='r', alpha=0.3)
ax.set_title('x = ' + str(x), fontsize=20)
ax.set_xlabel(r'$\theta_{1}$', fontsize=20)
ax.set_ylabel(r'$\theta_{0}$', fontsize=20)


# %% θ가 3일때 optimal한 loss의 graph

# 실제로는 각 input data마다 loss는 모양이 달라진다
# 모양은 달라도, 가야하는 방향은 일치 (data에 noise가 끼면 일치하지는 않고 근사)
# 단, cost function은 loss들의 평균이므로 모양이 바뀌지 않음
# 그렇기에 cost function만 사용하면 randomness가 낮아지므로 overfitting이 생길 수 있다
# 각 input data별 loss graph 그리기

import numpy as np
import matplotlib.pyplot as plt

n_sample = 20
t_th = 3

x_data = np.random.normal(0, 1, size=(n_sample, ))
y_data = t_th*x_data
th_range = np.linspace(t_th - 6, t_th + 6, 100)

fig, ax = plt.subplots(figsize=(12, 8))

th = np.random.normal(0, 3, size=(1, )).item()
learning_rate = 0.1

for x, y in zip(x_data, y_data):
    prediction = th * x
    dl_dp = -2 * x * (y - prediction)
    
    th = th - (dl_dp * learning_rate)
    
    print(th)
    ax.scatter(th, -2)
    # loss_range = (y - (th_range * x)) ** 2
    loss_range = th_range**2 * (x - y / th_range)**2
    ax.plot(th_range, loss_range)

# %% label에 noise를 추가하면

import numpy as np
import matplotlib.pyplot as plt

n_sample = 10
t_th = 3
noise = 0.2

x_data = np.random.normal(0, 1, size=(n_sample, ))
y_data = t_th*x_data + noise*np.random.normal(0, 1, size=(n_sample, ))
th_range = np.linspace(t_th - 5+0.1, t_th + 5, 100)

x_test = np.linspace(x_data.min(), x_data.max(), 2)
y_test = t_th*x_test

fig, axes = plt.subplots(1, 3, figsize=(16, 7))

axes[0].scatter(x_data, y_data)
axes[0].plot(x_test, y_test, color='r', linestyle=':')
axes[0].axvline(x=0, color='k')
axes[0].axhline(y=0, color='k')
th = 2
th_list = [th]
learning_rate = 0.01
epochs = 20

for epoch in range(epochs):
    for x, y in zip(x_data, y_data):
        prediction = th * x
        dl_dp = -2 *x*(y - prediction)
        
        th = th - (dl_dp * learning_rate)
        th_list.append(th)
    
        loss_range = th_range**2 * (x - y / th_range)**2
        
        if epoch == 0:
            axes[1].plot(th_range, loss_range)

axes[2].plot(th_list)
axes[1].scatter(th_list, np.zeros_like(th_list), 
                s=400, alpha=0.3, color='r')

fig.tight_layout()

