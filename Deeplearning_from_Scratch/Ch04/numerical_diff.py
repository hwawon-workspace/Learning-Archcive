import numpy as np

#%%
# 나쁜 구현 예
def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h # 전방 차분
# x+h와 x 사이의 함수 f의 차분을 구했으나, 진정한 미분은 기울기(접선)임
# 반올림 오차 발생
#%%
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h # 중심 차분/ 중앙 차분
#%%
def function_1(x):
    return 0.01*x**2 + 0.1*x
numerical_diff(function_1, 5)
# %%
# 편미분
def function_2(x): # 변수가 2개 이상
    return x[0]**2 + x[1]**2
#%%
# f(x0, x1) = x0**2 + x1**2
# x0 = 3, x1 = 4
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0
# x0에 대해 편미분
numerical_diff(function_tmp1, 3.0)
# %%
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1
# x1에 대해 편미분
numerical_diff(function_tmp2, 4.0)
# %%
# 기울기 계산
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h # f(x+h)
        fxh1 = f(x)
        
        x[idx] = tmp_val - h # f(x-h)
        fxh2 = f(x)
        
        grad[idx] = (fxh1 -fxh2) / (2*h) # f(x+h) - f(x-h) / 2h
        x[idx] = tmp_val
    return grad
# %%
numerical_gradient(function_2, np.array([3.0, 4.0])) # (3,4)에서의 기울기
numerical_gradient(function_2, np.array([0.0, 2.0])) # (0,2)에서의 기울기
numerical_gradient(function_2, np.array([3.0, 0.0])) # (3,0)에서의 기울기