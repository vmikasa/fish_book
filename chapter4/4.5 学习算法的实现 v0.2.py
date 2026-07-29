"""
这个是v0.2版本，与0.1版本不同的是，该版本使用了batch重构，利用batch并行计算的优势，预计将会大大提升训练速度
"""

import numpy as np



def sigmoid(x):
    """
    需要注意的算是这里传入的x是一个矩阵
    :param x: x是矩阵
    :return: 返回值也是一个通过激活函数的矩阵
    """
    return 1/(1+np.exp(-x))

def softmax(x):
    """
    注意，传入的x也是一个矩阵，是带batch的
    激活函数，softmax，用于概率归一化
    :param x: x是一个矩阵
    :return: 返回值是通过softmax概率归一化的一个矩阵
    """
    if x.ndim==1:
        x=x.reshape(1,x.size)

    c=np.max(x,axis=1,keepdims=True)
    x_exp=np.exp(x-c)
    x_exp_sum=np.sum(x_exp,axis=1,keepdims=True)
    return x_exp/x_exp_sum

def cross_entropy_loss(y,t):
    """
    这里y是预测输出的矩阵，二维；t是正确标签，可能是one-hot，可能是非ont-hot
    用于计算交叉熵损失函数，这里写的是有batch的请看
    :param y: 前向传播预测输出的矩阵，是二维的，如果是一维的做处理变成二维
    :param t: 标签
    :return:返回交叉熵损失函数的计算值，带batch的返回是一个矩阵
    """

    delta=1e-7

    # 先处理一下非二维矩阵的情况，将它们变成二维矩阵
    if y.ndim==1:
        y=y.reshape(1,y.size)
        if t.size==y.size:  # 如果传入的标签t是one-hot，那么需要变成矩阵；如果是非one-hot，那么自然可以不管；如果y.ndim不是1，那么也可以不管
            t=t.reshape(1,t.size)

    batch=y.shape[0]

    if y.size==t.size:
        return -np.sum(t*np.log(y+delta))/batch # 平均损失函数求导，就等于平均梯度相加了。虽然损失函数是标量，但是梯度可以看作是向量

    else:
        return np.sum(-np.log(y[np.arange(batch),t]+delta))/batch


def numerical_gradient(f,x):
    """
    这里是batch实现的数值微分函数。f是目标函数，这里其实就是损失函数，f(x,t；W，b)
    :param f: 目标函数，f一般是固定x和t，微小扰动W或b，前向传播再交叉熵损失函数，从而求得W或b的梯度
    :param x: 这里的x是一个矩阵，必须得可以计算矩阵
    :return: return的是一个数值微分的矩阵
    """

    """
    不需要判断了。因为迭代器np.ndindex天然支持一维、二维，不需要判断数组形状
    # 先判断是否为二维矩阵，如果不是转换为二维矩阵
    if x.ndim==1:
        x=x.reshape(1,x.size)
    """
    # 定义微小扰动h，并且创建grad梯度矩阵，形状与x相同。并且确保x是float
    # 因为整数数组不保存小数，微小扰动h无效，所以要确保x是float
    h=1e-4
    x=np.asarray(x,dtype=np.float64)
    grad=np.zeros_like(x)

    # 取出x的索引，存到index
    index=np.ndindex(x.shape)

    # 下面正式开始梯度计算
    for i in index:
        temp=x[i]   # 将x[i]暂存到temp中

        # 先计算f(x+h)
        x[i]=temp+h
        fh1=f(x)    # 一次前向传播+交叉熵损失函数计算

        # 再计算f(x-h)
        x[i]=temp-h
        fh2=f(x)    # 一次前向传播+交叉熵损失函数计算

        # 下面计算梯度
        grad[i]=(fh1-fh2)/(2*h)

        # 还原x[i]
        x[i]=temp

    return grad






class ThreeLayerNet:
    """
    定义一个三层网络类
    """
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size,weight_init_std=0.01):
        """
        初始化肯定是获得初始权重
        :param input_size: 输入神经元个数。需要注意的是，现在是带批次的了
        :param hidden_size1:隐藏层1神经元个数
        :param hidden_size2: 隐藏层2神经元个数
        :param output_size: 输出神经元个数
        :param weight_init_std: 权重标准差，默认为0.01，用处是防止初始化的权重让sigmoid等激活函数过饱和
        """
        # 用一个字典来init初始权重
        self.params= {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size1),
            "b1": np.zeros(hidden_size1),

            "W2": weight_init_std * np.random.randn(hidden_size1, hidden_size2),
            "b2": np.zeros(hidden_size2),

            "W3": weight_init_std * np.random.randn(hidden_size2, output_size),
            "b3": np.zeros(output_size)
        }

    def forward(self,x):
        """
        前向传播函数。注意这里的x是带batch批次的，不要当成一维数组来处理
        :param x: 输入批次batch
        :return: 返回值输出的概率y
        """

        # 先算出batch，不管有没有用，反正算了再说
        if x.ndim==1:
            x=x.reshape(1,x.size)

        # batch=x.shape[0]  # 好像没什么用，注释掉

        # 获取权重参数
        W1,W2,W3=self.params["W1"],self.params["W2"],self.params["W3"]
        b1,b2,b3=self.params["b1"],self.params["b2"],self.params["b3"]

        # 前向传播计算
        a1=x@W1+b1
        z1=sigmoid(a1)
        a2=z1@W2+b2
        z2=sigmoid(a2)
        a3=z2@W3+b3
        z3=softmax(a3)

        y=z3
        return y


    def loss(self,x,t):
        """
        这是一个方法，用于计算交叉熵损失函数，返回值是交叉熵损失函数的结果
        :param x: x就是输入神经元，带batch版本的
        :param t: t就是标签，带batch版本的
        :return: return计算结果，是平均的交叉熵损失函数值
        """

        # 先获得预测值y
        y=self.forward(x)
        loss=cross_entropy_loss(y,t)
        return loss

    def accuracy(self,x,t):
        """
        用于计算一个batch内的准确率。x是输入神经元，或者说输入矩阵，t是标签
        :param x: 输入的矩阵
        :param t: 标签
        :return: 返回一个0-1的数，表示准确率
        """

        # 先获得y
        y=self.forward(x)

        if y.ndim==1:
            y=y.reshape(1,y.size)

        batch=y.shape[0]

        if y.size==t.size:
            y=np.argmax(y,axis=1)
            t=np.argmax(t,axis=1)
            return np.sum(y==t)/batch
        else:
            y=np.argmax(y,axis=1)
            return np.sum(y==t)/batch


    def numerical_gradient(self,x,t):
        """
        调用外部给numerical_gradient来完成数值微分的梯度计算
        x和t都是固定不变的，数值微分的对象是W和b
        :param x:x是输入的数组，是固定不变的
        :param t:t是正确标签，固定不变的
        :return:返回值是用字典装起来的梯度
        """
        # cross_entropy_loss函数已经将x非batch和t为非one-hot的版本都考虑到了，无需担心
        # 最后得到的是一个交叉熵损失函数的计算值，即固定x和t，改变W和b，得到的计算值
        # 这样，仅改变W和b，然后微小扰动，数值微分，得到的就是W和b的梯度了
        # 另外，外部的numerical_gradient(f,x)函数也封装好了，传入函数关系f，和待计算梯度矩阵x即可
        loss=lambda w:self.loss(x,t)     # 获得函数关系loss

        # 新建梯度
        grad= {
            "W1": numerical_gradient(loss, self.params["W1"]),
            "b1": numerical_gradient(loss, self.params["b1"]),

            "W2": numerical_gradient(loss, self.params["W2"]),
            "b2": numerical_gradient(loss, self.params["b2"]),

            "W3": numerical_gradient(loss, self.params["W3"]),
            "b3": numerical_gradient(loss, self.params["b3"])
        }

        return grad

    def numerical_gradient_descent(self,x,t,lr=0.01):
        grad=self.numerical_gradient(x,t)
        self.params["W1"]-=lr*grad["W1"]
        self.params["b1"]-=lr*grad["b1"]
        self.params["W2"]-=lr*grad["W2"]
        self.params["b2"]-=lr*grad["b2"]
        self.params["W3"]-=lr*grad["W3"]
        self.params["b3"]-=lr*grad["b3"]


# 下面尝试使用这个batch版本的网络

import os,sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# 先获得训练数据
(train_img,train_label),(test_img,test_label)=load_mnist(flatten=True,normalize=True,one_hot_label=True)

# 获得网络层实例
net=ThreeLayerNet(input_size=784,hidden_size1=50,hidden_size2=100,output_size=10)

# 设置超参数相关
train_size=train_img.shape[0]
batch_size=100      # batch大小自己填，想填多少填多少
iter_num=1000       # 无batch版本是训练100000次，那我们这个版本按batch次数迭代，刚好就是1000次迭代次数

# 设置空列表loss_list
loss_list=[]


# 直接利用batch，数值微分梯度下降更新
for i in range(iter_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x=train_img[batch_mask]
    t=train_label[batch_mask]

    net.numerical_gradient_descent(x,t,lr=1)

    # 每个batch完成后，打印一次loss，记录loss
    loss_temp=net.loss(x,t)
    loss_list.append(loss_temp)
    print(f"第{i+1}次训练，当前的loss是{loss_temp}")

# 最终记录，将loss记录为txt文件，将权重记录为pkl文件
import pickle

with open("loss_list_batch.txt2","a",encoding="utf-8") as f:
    for loss in loss_list:
        f.write(f"{loss}\n")

with open ("weight_batch.pkl2","wb") as f:
    pickle.dump(net.params,f)