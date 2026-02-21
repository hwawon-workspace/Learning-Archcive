import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
#%%
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape, t_train.shape)
#%%
train_size = x_train.shape[0]
batch_size = 10
# np.random.choice(): 지정한 수 중 무작위로 원하는 개수만 꺼낼 수 있음
batch_mask = np.random.choice(train_size, batch_size) 
print(batch_mask) # 무작위 추출할 인덱스, [ 9259 30690 45051 34909 10867 30054 35878 46130 21605 49137]
x_batch = x_train[batch_mask] # 무작위 추출한 x
t_batch = t_train[batch_mask] # 무작위 추출한 x에 해당하는 target
# %%
# np.random.choice()
np.random.choice(60000, 10) # 0~60000 중 10개 무작위 추출
#%%
# data 하나일 때/ 배치로 묶여서 입력될 경우 CEE 
# target이 one-hot encoding일 때
def cross_entropy_error(y, t):
    if y.ndim == 1: # 배열의 차원 수가(배치 수가) 1이면
        t = t.reshape(1, t.size) # (1, size) -> 2차원으로 변환
        y = y.reshape(1, y.size)
    batch_size = y.shape[0] # 배치 사이즈 = 배치 하나 당 data 개수
    # 원핫 인코딩 시 t가 0인 원소는 CEE도 0이므로 계산 무시해도 됨. -> 정답에 해당하는 출력만 CEE 계산 가능 
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

#%%
# target이 one-hot encoding이 아닌 label로 주어졌을 때
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size