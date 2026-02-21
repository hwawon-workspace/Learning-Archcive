import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#%%
# 1 에포크 당 train data, test data에 대한 accuracy 기록
# 1 에포크: 학습에서 train data 모두 소진했을 때의 횟수
# ex. train data 10,000개를 100개의 미니 배치로 학습 시 sgd 100회 반복 시 데이터 소진
# -> 1에포크: 100회
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)
    
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10) # 신경망

iters_num = 10000 # 반복 수 
train_size = x_train.shape[0] # 학습 데이터 개수
batch_size = 100 # 배치 크기
learning_rate = 0.1 # 학습률

train_loss_list = [] # loss 기록 리스트
train_acc_list = [] # train acc
test_acc_list = [] # test acc

iter_per_epoch = max(train_size / batch_size, 1) # 1에포크에 필요한 iteration 수
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 미니배치에 사용할 인덱스 생성
    x_batch = x_train[batch_mask] # x 미니배치
    t_batch = t_train[batch_mask] # t 미니배치
    
    grad = network.numerical_gradient(x_batch, t_batch) # 기울기 계산
    
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key] # 학습률 * 기울기만큼 빼주기
    
    loss = network.loss(x_batch, t_batch) # x 미니배치와 t 미니배치 비교해 loss 계산
    train_loss_list.append(loss) # 기록용 리스트에 추가
    
    if i % iter_per_epoch == 0: # 1 에포크를 다 돌면
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | ' + 'f{train_acc:.4f}, {test_acc:.4f}')