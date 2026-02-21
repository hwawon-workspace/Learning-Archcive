import numpy as np
import matplotlib.pyplot as plt
#%%
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show() # 그래프를 화면에 출력
#%%
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("Tong'spatch.png") # 이미지 읽어오기(이미지 주소 작성)

plt.imshow(img) # 이미지 표시해주는 메서드
plt.show()