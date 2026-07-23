# 下面是一些常见的损失函数

import numpy as np


def mean_squared_error(y,t):
    return np.sum(0.5*(y-t)**2)



def cross_entropy_error_test(y,t):
    """
    numpy中，有一些对数函数。比如，np.log()是以e为底的，np.log2()是以2为底的

    :param y: 预测的概率,是一个一维数组
    :param t: 真实的概率，通常为1
    :return: 返回结果是交叉熵，或者说损失函数
    """

    delta=1e-7      # 是为了防止出现log(0)而导致报错的情况
    return -np.sum(t*np.log(y+delta))   # 这里应该是把负号提到log前面，才能表示信息量。但是这里为了美观所以放在sum外面



"""
下面介绍函数np.random.choice()，这个函数是随机抽样函数

np.random.choice(a,size=None,replace=True,p=None)

这是用于随机抽样的函数：
a代表抽样池，只能是一维数组，或者是一个整数;
size是输出形状，默认None代表返回一个元素，也可以返回特定形状比如（1，2）
replace=True，表示默认有放回抽样。如果是replace=False，则表示无放回抽样
p=None，表示默认没有概率分布。如果使用p，则p的长度必须与a相等，表示抽取a的概率权重分布。其中概率分布加起来必须为1，且概率不能有负数
"""

# 下面演示从训练数据中抽取10个作为mini_batch

import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def get_data():
    (train_img,train_label),(test_img,test_label)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    return train_img,train_label

train_img,train_label=get_data()

train_size=train_img.shape[0]
batch_size=10
batch_mark=np.random.choice(train_size,batch_size)

img_batch=train_img[batch_mark]
label_batch=train_label[batch_mark]

print(img_batch.shape)
print(label_batch.shape)

# 下面重新定义一个批次的交叉熵损失函数

def cross_entropy_error(y,t):
    """
    这里输入的y和t都当成批次batch来处理。注意传入的只能是y和t的批次，不能是其他的东西
    :param y: y是输入的预测概率，这里是批次处理，所以y是二维数组。每一行是一个预测输入
    :param t: t是输入的标签，同理，t是跟y一样的批次
    :return:返回值是交叉熵
    """

    # 这里是为了解决y如果是一维数组，则y.shape[0]出问题的bug
    if y.ndim==1:       # y.ndim,这里的n.dim是numbers of dimensions的缩写
        y=y.reshape(1,y.size)       # 这里的y.size，size表示所有元素的总和
        t=t.reshape(1,t.size)

    delta=1e-7
    batch=y.shape[0]        # 有bug，这里y只能是二维数组，如果y是一维数组，那么y.shape[0]返回的就是元素个数，batch会出错
    # return -np.sum(t*np.log(y+delta))/batch     # 还有一个bug，如果不是one-hot标签，那么就会出问题，所以，需要打补丁

    if y.size==t.size:
        return -np.sum(t*np.log(y+delta))/batch
    else:
        return -np.sum(np.log(y[np.arange(batch),t])+delta)/batch

