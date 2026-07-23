"""
这是3.6手写数字识别的升级版，增加了批处理的功能。同时也稍微复习一下
"""

import pickle
import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def get_data():
    (train_image,train_label),(test_image,test_label)=load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return test_image,test_label

def init_network():
    with open(r"../dataset/sample_weight.pkl",'rb') as f:
        network=pickle.load(f)

    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """
    这个函数有缺陷。只能计算一维数组，假如传入的是batch，即二维数组，那就算不了了
    :param x: 传入x一定是一个一维数组
    :return: return是一维数组的概率分布
    """
    c=np.max(x)
    x_exp=np.exp(x-c)
    x_exp_sum=np.sum(x_exp)
    return x_exp/x_exp_sum

def predict(x,network):
    b1,b2,b3=network["b1"],network["b2"],network["b3"]
    W1,W2,W3=network["W1"],network["W2"],network["W3"]
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    z3=softmax(a3)
    y=z3

    return y

img,label=get_data()
network=init_network()
batch_size=100
accuracy_cnt=0

# range循环遍历，等价于每一次循环结束后再加step，而不是每一次循环一开始就加step
for i in range(0,len(img),batch_size):
    x_batch=img[i:i+batch_size]
    y_batch=predict(x_batch,network)
    """
    np.argmax(axis=1)，axis=1就是消除列这个维度，等价于只看行，返回每一行的最大值的下标，然后组成一维数组。
    不过，更正确的理解是，axis=1，就是在axis=1的这个维度之间比大小，所以是同一行的列与列之间比较，最后得到结果
    """
    p=np.argmax(y_batch,axis=1)

    """注意，np.sum()，括号内可以是布尔值，也就是说可以对布尔值求和"""
    accuracy_cnt+=np.sum(p==label[i:i+batch_size])

print(f"预测准确率是：{100*accuracy_cnt/len(img)}%")

