import numpy as np
#%%
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 원 핫 인코딩 # 정답은 2

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 가장 높다고 예측함
sum_squares_error(np.array(y), np.array(t)) # loss: 0.0975

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # 7일 확률이 가장 높다고 예측함
sum_squares_error(np.array(y), np.array(t)) # loss: 0.5975