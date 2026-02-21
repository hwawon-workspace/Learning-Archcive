import numpy as np
from numerical_diff import numerical_gradient
#%%
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x) # x에서 기울기 계산
        x -= lr * grad # lr*grad만큼 x를 이동시키기
    return x
#%%
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0]) # 초깃값 (-3, 4)로 설정
# 학습률 변경해가며 보기
gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100) # array([-6.11110793e-10,  8.14814391e-10])
# 학습률이 너무 크면 큰 값으로 발산해버림
gradient_descent(function_2, init_x = init_x, lr = 10.0, step_num = 100) # array([ 2.34235971e+12, -3.96091057e+12])
# 학습률이 너무 작으면 거의 갱신돠지 않고 끝남
gradient_descent(function_2, init_x = init_x, lr = 1e-10, step_num = 100) # array([ 2.34235971e+12, -3.96091057e+12])
# %%