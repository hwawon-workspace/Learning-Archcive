import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.ndim(A) # 배열의 차원 수 2
A.shape # 배열의 형상을 튜플로 반환 (2, 2)
np.dot(A, B) # 행렬 곱