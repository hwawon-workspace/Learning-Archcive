import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle
from hidden_layer_act import sigmoid
from softmax import softmax

#%%
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f: # sample_weight.pkl: 학습된 가중치 매개변수, 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있음.
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
#%%
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i]) # 신경망 통해 y값 계산 
    p = np.argmax(y) # 확률이 가장 높은 인덱스
    if p == t[i]: # 추론 값이 타겟 값과 같으면
        accuracy_cnt += 1
print('Accuracy:' + str(float(accuracy_cnt) / len(x))) # 정확도 계산
#%%
# 가중치 형상 출력
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape) # (10000, 784)
print(x[0].shape) # (784, )
print(W1.shape, W2.shape, W3.shape) # (784, 50), (50, 100), (100, 10)
# %%
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size] # x 배치 지정
    y_batch = predict(network, x_batch) # x 배치를 신경망에 통과시킨 y 배치 지정
    p = np.argmax(y_batch, axis = 1) # y_batch 에 가장 높은 확률을 가진 index들만 저장
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # target과 y_batch를 비교해 맞은 것만큼 더해 정확도 측정
print('Accuracy:' + str(float(accuracy_cnt) / len(x))) # 전체 정확도 계산
#%%
# 위 코드에 사용된 문법 설명
# range(start, end, step) 함수 사용법
print(list(range(0, 10)))
print(list(range(0, 10, 3)))

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis = 1) # [1 2 1 0] 가장 큰 값을 가진 index들 저장
print(y)

y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t) # [True True False True]
np.sum(y == t) # 3
# %%
 