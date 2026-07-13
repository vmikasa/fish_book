# 这下面是书上的源代码

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(train_image,train_labels),(test_image,test_labels)=load_mnist(flatten=True, normalize=False)
# load_mnist的返回值是两个元组，分别是(训练图像, 训练标签), (测试图像, 测试标签)。元组里面套的是numpy的数组

img=train_image[0]
label=train_labels[0]
print(label)
# img_show(img)       # 一维的784，什么破玩意
print(img.shape)
print(img)

# 先将一维的784转为28*28先
img=img.reshape(28,28)      # reshape方法，只接受数组，重新塑形为28*28
img_show(img)

