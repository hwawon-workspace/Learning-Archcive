import numpy as np
#%%
x = np.array([1.0, 2.0, 3.0])
print(x) # [1., 2., 3.,]
type(x) # <class 'numpy.ndarray'>
#%%
A = np.array([[1, 2], [3, 4]])
print(A)
# [[1 2]
#  [3 4]]
print(A.shape) # 행렬의 형상 # (2, 2)
print(A.dtype) # 행렬에 담긴 원소의 자료형 # dtype('int64')