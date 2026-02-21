import sys, os
sys.path.append(os.pardir) # 부모 디렉토리의 파일 가져오기
from dataset.mnist import load_mnist # dataset 폴더의 mnist.py에서 정의된 load_mnist 가져오기
import numpy as np
from PIL import Image
#%%
# 최초 실행 시 파일로부터 데이터를 받아오고, 두 번째 실행부터는 로컬에 저장된 pickle 파일을 불러와 읽기 때문에 빠르게 실행됨
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize = False)

print(f'x_train shape: {x_train.shape}, t_train.shape: {t_train.shape}')
print(f'x_test shape: {x_test.shape}, t_test.shape: {t_test.shape}')
#%%
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # numpy array로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()

(x_train, t_train), (x_test, t_test) =  \
    load_mnist(flatten = True, normalize = False)

img, label = x_train[0], t_train[0]
print(label) # 5

print(img.shape) # flatten = True이기 때문에 shape은 784
img_show(img) # flatten되어 784 * 1 형태로 보임

img = img.reshape(28, 28) # 원본 형태 28 * 28로 reshape
print(img.shape) # 28 * 28
img_show(img)