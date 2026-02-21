import numpy as np
#%%
def softmax(a):
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y
#%%
# 오버플로우 문제 해결하기 위해 양 변 C로 나눠줌
def softmax(a):
	c = np.max(a) # 입력 값 중 최대값을 빼줌으로써 오버플로 문제 방지
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y