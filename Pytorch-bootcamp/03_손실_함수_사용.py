#%%
# 5.2 MSE Loss

def mse(x_hat, x):
  y = ((x - x_hat)**2).mean()
  return y

x = torch.FloatTensor([[1,1],
                       [2,2]])
x_hat = torch.FloatTensor([[0,0],
                           [0,0]])
print(mse(x_hat, x))

import torch.nn.functional as F
F.mse_loss(x_hat, x)

F.mse_loss(x_hat, x, reduction = 'sum')

F.mse_loss(x_hat, x, reduction  = 'none')

import torch.nn as nn
mse_loss = nn.MSELoss()
mse_loss(x_hat, x)
#%%
