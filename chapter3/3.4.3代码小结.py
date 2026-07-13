# 这里小结一下上一节写出来的三层神经网络

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 恒等函数
def identity(x):
    return x


# 这是网络初始化
# 网络层里面是各层神经元的权重和偏置，都是参数
# init_network就是权重和偏置初始化
def init_network():
    network={
        'W1':np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]),
        'b1':np.array([0.1,0.2,0.3]),
        'W2':np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]]),
        'b2':np.array([0.1,0.2]),
        'W3':np.array([[0.1,0.3],[0.2,0.4]]),
        'b3':np.array([0.1,0.2]),
    }  # 创建一个空字典
    return network      # 返回了一个字典

# 定义前向传播，前向传播就是用网络层的参数来计算
def forward(network,x):
    # init_network返回了一个字典，这个字典就是network，所以首先第一步就是解包拿到网络层参数
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=a3

    return y


network=init_network()      # 获得网络层参数
x=np.array([1,0.5])         # 获得输入层。输入层没有权重和偏置。随便算不算神经元，严格来说不算神经元
y=forward(network,x)        # 输入层x输入network，做计算，最后输出y
print(y)


# 有前向传播forward，就必然有反向传播backward
