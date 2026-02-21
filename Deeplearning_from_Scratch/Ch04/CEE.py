
#%%
def cross_entropy_error(y, t):
    delta = 1e-7 # np.log에 0이 들어가 -inf가 되어 에러가 되지 않도록
    return -np.sum(t*np.log(y + delta))
# %%
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 정답의 출력이 0.6일 때
cross_entropy_error(np.array(y), np.array(t)) # loss: 0.5108

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # 정답의 출력이 0.1일 때
cross_entropy_error(np.array(y), np.array(t)) # loss: 2.3025