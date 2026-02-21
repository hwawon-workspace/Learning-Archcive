import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#%%
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)
    
train_loss_list = [] # loss 기록 리스트

iters_num = 10000 # 반복 수 
train_size = x_train.shape[0] # 학습 데이터 개수
batch_size = 100 # 배치 크기
learning_rate = 0.1 # 학습률

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10) # 신경망

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 미니배치에 사용할 인덱스 생성
    x_batch = x_train[batch_mask] # x 미니배치
    t_batch = t_train[batch_mask] # t 미니배치
    
    grad = network.numerical_gradient(x_batch, t_batch) # 기울기 계산
    
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key] # 학습률 * 기울기만큼 빼주기
    loss = network.loss(x_batch, t_batch) # x 미니배치와 t 미니배치 비교해 loss 계산
    train_loss_list.append(loss) # 기록용 리스트에 추가
# %%
