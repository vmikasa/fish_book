"""
我们来尝试手写实现一下，只用一个numpy
假设这个网络是一个简单的网络，那么就先要权重初始化，然后利用初始化的权重进行预测，再计算损失函数
然后，根据损失函数，来计算出梯度，然后，再按梯度下降方向更新权重W
注意，这里为了演示，只是单层网络，并且没有设置偏置b
"""

import numpy as np

def cross_entropy_error(y,t):
    """这里我需要假设y和t都是批次
        这个函数处理了batch=1和非one-hot标签的bug
    """
    if y.ndim==1:
        y=y.reshape(1,y.size)
        t=t.reshape(1,t.size)

    batch=y.shape[0]
    delta=1e-7

    if y.size==t.size:
        return -np.sum(t*np.log(y+delta))/batch
    else:
        return -np.sum(np.log(y[np.arange(batch),t]+delta))/batch


def softmax_sample(x):
    """
    这个函数有缺陷。只能计算一维数组，假如传入的是batch，即二维数组，那就算不了了
    打个补丁，重新定义一个softmax
    :param x: 传入x一定是一个一维数组
    :return: return是一维数组的概率分布
    """
    c=np.max(x)
    x_exp=np.exp(x-c)
    x_exp_sum=np.sum(x_exp)
    return x_exp/x_exp_sum

def softmax(x):

    # 因为基本都是批处理的情况，所以先看二维数组
    if x.ndim==2:
        c=np.max(x,axis=1,keepdims=True)
        x_exp=np.exp(x-c)
        x_exp_sum=np.sum(x_exp,axis=1,keepdims=True)
        return x_exp/x_exp_sum
    else:
        c=np.max(x)
        x_exp=np.exp(x-c)
        x_exp_sum=np.sum(x_exp)
        return x_exp/x_exp_sum




def numerical_gradient(f,x):
    """
    这个是数值微分的梯度计算函数。返回值是计算好的梯度
    另外，之前写的数值微分的梯度计算代码，只能处理x是一维数组的情况，因为索引有问题
    这次采用np.ndindex()方法，自动获取索引元组，返回值是索引元组的迭代器
    :param f: 待计算梯度的目标函数，这里应该是前向传播后，再经过交叉熵得到的损失函数
    :param x:可以是矩阵
    :return: grad，梯度
    """
    h=1e-4
    x=np.asarray(x,dtype=np.float64)
    grad=np.zeros_like(x,dtype=np.float64)

    for i in np.ndindex(x.shape):
        temp=x[i]

        # 先计算f(x+h)
        x[i]=temp+h
        fh1=f(x)    # 改变x[i]数值，进行一次前向传播，得出fh1

        # 再计算f(x-h)
        x[i]=temp-h
        fh2=f(x)    # 改变x[i]数值，再进行一次前向传播，得到fh2

        # 现在计算第i个维度的梯度
        grad[i]=(fh1-fh2)/(2*h)
        x[i]=temp

    return grad

class SimpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)         # 假设输入的神经元是2个，这里获得初始化W权重

    def forward(self,x):
        z=x@self.W
        y=softmax(z)
        return y

    def loss(self,x,t):
        y=self.forward(x)
        loss=cross_entropy_error(y,t)

        return loss

net=SimpleNet()
print(net.W)
x=np.array([0.6,0.9])
p=net.forward(x)
print(p)
print(np.argmax(p))
t=np.array([0,0,1])
print(net.loss(x,t))
print("-"*30)

x=np.zeros((2,3))
print(x.shape)
index=np.ndindex(x.shape)
print(index)

