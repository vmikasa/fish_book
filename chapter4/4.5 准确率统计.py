import pickle
import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    if x.ndim == 1:
        x=x.reshape(1,x.size)

    c=np.max(x,axis=1,keepdims=True)
    x_exp=np.exp(x-c)
    return x_exp/x_exp.sum(axis=1,keepdims=True)


def forward(x,params):
    """
    前向传播的函数，或者说方法
    :param x:输入的数组，或者说输入的神经元
    :return:返回前向传播y的预测结果
    """
    W1, W2, W3 = params["W1"], params["W2"], params["W3"]
    b1, b2, b3 = params["b1"], params["b2"], params["b3"]

    a1 = x @ W1 + b1
    z1 = sigmoid(a1)

    a2 = z1 @ W2 + b2
    z2 = sigmoid(a2)

    a3 = z2 @ W3 + b3
    z3 = softmax(a3)

    y = z3

    return y

def accuracy(x,t):
    y=forward(x,params)


# 获得权重参数
with open("weight_batch.pkl2","rb") as f:
    params=pickle.load(f)

# 获得测试集
(train_img,train_label),(test_img,test_label)=load_mnist(normalize=True,one_hot_label=True,flatten=True)


# 我管你那多，直接算，反正是batch，不是最终代码

y=forward(test_img,params)
y=np.argmax(y,axis=1,keepdims=True)
t=np.argmax(test_label,axis=1,keepdims=True)
print(100*np.sum(y==t)/y.shape[0])

