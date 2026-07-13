# 下面是一个三层神经网络的实现
# 第0层是输入层，然后是第1层，第2层，第3层

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 恒等函数
def identity(x):
    return x
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1=np.dot(X,W1)+B1

print(A1)

Z1=sigmoid(A1)
print(f"Z1是{Z1}")

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)
print(f"Z2：{Z2}")

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3

Y=identity(A3)

# 输出层的激活函数，取决于具体要解决的问题。
# 一般来说，回归问题用恒等函数，二元问题用sigmoid，多元问题用softmax



