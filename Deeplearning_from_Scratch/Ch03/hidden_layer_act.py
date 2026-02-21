import numpy as np
import matplotlib.pyplot as plt
#%%
def step_function(x):
	if x > 0:
		return 1
	else:
		return 0

# numpy array형태로 들어갈 때	
def step_function(x):
	y = x > 0 # [True, False, ...] bool형 자료로 나타남
	return y.astype(np.int) # True는 1, False는 0으로 바꿔줌
	
#%%
# step function 시각화 코드
def step_function(x):
	return np.array(x > 0, dtype = np.int)
x = np.arange(-5.0, 5.0, 0.1) # 0.1 간격으로 넘파이 배열 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 출력 경계
plt.show()
#%%
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
# numpy 배열 형식도 input으로 사용 가능
#%%
# 시그모이드 시각화 코드
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
#%%
def relu(x):
	return np.maximum(0, x)