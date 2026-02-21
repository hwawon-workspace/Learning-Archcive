#%%
# 6.5 경사하강법 구현

import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.1, .2, .3],
                            [.4, .5, .6],
                            [.7, .8, .9]])

x = torch.rand_like(target)
x.requires_grad = True
print(x)

loss = F.mse_loss(x, target)
loss

threshold = 1e-5
learning_rate = 1.
iter_cnt = 0

while loss > threshold:
  iter_cnt += 1
  loss.backward()

  x = x - learning_rate * x.grad

  x.detach_()
  x.requires_grad_(True)

  loss = F.mse_loss(x, target)

  print('%d-th Loss: %.4e' % (iter_cnt, loss))
  print(x)
#%%
# 6.6 파이토치 오토그래드 소개

x = torch.FloatTensor([[1,2],
                       [3,4]]).requires_grad_(True)

x1 = x + 2
print(x1)

x2 = x - 2
print(x2)

x3 = x1 * x2
print(x3)

y = x3.sum()
print(y)

y.backward()

print(x.grad)

x3.detach_()
# %%
