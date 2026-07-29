"""
这次用反向传播的方法，来实现训练。前面写的Affine层和SoftWithLoss层都是可以正常处理batch的

"""

"""
首先需要学习一个python的语法，有序字典

OrderedDict，有序字典，是python标准库collections模块里面的一个特殊字典
python早期版本，普通字典无法记住存入的键值对的顺序，字典的遍历都是随机的。（python3.7开始，普通字典也支持记忆顺序，但是仍然建议使用OrderDict显示声明）


"""


from collections import OrderedDict
import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def softmax(x):

    if x.ndim == 1:
        x=x.reshape(1, x.size)

    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    exp_sum = np.sum(exp_x, axis=1, keepdims=True)

    return exp_x / exp_sum


def cross_entropy_loss(y,t):

    if y.ndim == 1:
        y=y.reshape(1,y.size)
        if y.size==t.size:
            t=t.reshape(1,t.size)

    delta=1e-7
    batch_size=y.shape[0]

    if y.size==t.size:
        return -np.sum(t*np.log(y+delta))/batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size),t]+delta))/batch_size


def numerical_gradient(f,x):
    """
    数值微分，传入f和x，计算f关于x的梯度。返回值就是梯度
    :param f:
    :param x:
    :return:
    """
    x=np.asarray(x,dtype=np.float64)
    h=1e-4
    index=np.ndindex(x.shape)
    grad=np.zeros_like(x)

    for i in index:
        temp=x[i]

        x[i]=temp+h
        fh1=f(x)

        x[i]=temp-h
        fh2=f(x)

        grad[i]=(fh1-fh2)/(2*h)

        x[i]=temp

    return grad



class SoftmaxWithLoss:

    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None

    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_loss(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        """
        关于为什么要让这里的y-t除以batch_size
        因为有：self.dW=self.x.T@dout
        最后计算出来的梯度dW，是x的转置乘以y-t，也就是说，梯度会把所有图片同一个像素点位置的贡献加起来，所以必然要除以batch_size了
        另一种理解方式，我们使用的损失函数是平均损失函数，任意一张图片的x变化一点点，对损失函数的影响只能是Δx/batch，不能太多，因为是平均值

        :param dout:
        :return:
        """
        batch_size=self.y.shape[0]
        if self.y.size==self.t.size:
            dx=(self.y-self.t)/batch_size

        else:
            dx=self.y.copy()
            dx[np.arange(batch_size),self.t]-=1
            dx=dx/batch_size

        return dx



class Sigmoid:

    def __init__(self):
        self.out=None

    def forward(self,x):
        self.out=1/(1+np.exp(-x))
        return self.out

    def backward(self,dout):
        dx=dout*self.out*(1-self.out)
        return dx


class Affine:

    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.dW=None
        self.db=None
        self.x=None

    def forward(self,x):
        self.x=x
        out=x@self.W+self.b

        return out

    def backward(self,dout):
        self.db=np.sum(dout,axis=0)
        dx=dout@self.W.T
        self.dW=self.x.T@dout

        return dx


class ThreeLayerNet:
    def __init__(self,input_size=784,hidden_size1=50,hidden_size2=100,output_size=10, weight_init_std=0.01):
        """
        先初始化权重，再生成层

        这与数值微分版本的代码不同，数值微分版本的代码是只生成权重，然后直接前向传播算就行了
        而反向传播版本的代码，生成权重后，还需要有层，因为需要保存整个网络结构的每一层的输入数据，这样才能进行反向传播计算梯度
        这听起来有的像是用空间换时间

        这样，在网络类的初始化里面，就可以很自然地改网络层了，自由组合

        :param input_size:
        :param hidden_size1:
        :param hidden_size2:
        :param output_size:
        :param weight_init_std:
        """

        # 初始化权重
        self.params= {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size1),
            "b1": weight_init_std * np.random.randn(hidden_size1),

            "W2": weight_init_std * np.random.randn(hidden_size1, hidden_size2),
            "b2": weight_init_std * np.random.randn(hidden_size2),

            "W3": weight_init_std * np.random.randn(hidden_size2, output_size),
            "b3": weight_init_std * np.random.randn(output_size)
        }


        # 下面初始化网络层
        # 需要用到有序字典。前向传播就是正序，反向传播就是倒序
        self.layers=OrderedDict()
        self.layers["Affine1"]=Affine(self.params["W1"],self.params["b1"])
        self.layers["Sigmoid1"]=Sigmoid()
        self.layers["Affine2"]=Affine(self.params["W2"],self.params["b2"])
        self.layers["Sigmoid2"]=Sigmoid()
        self.layers["Affine3"]=Affine(self.params["W3"],self.params["b3"])    # 最后一层的原始得分

        self.lastLayer=SoftmaxWithLoss()        # 最后一层经过激活函数和损失函数处理

    def forward(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x    # 最后一次x就是最后一个layer的输出了

    def loss(self,x,t):
        y=self.forward(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y=self.forward(x)
        y=np.argmax(y,axis=1)
        if t.ndim!=1:
            t=np.argmax(t,axis=1)
        acc=np.sum(y==t)/x.shape[0]

        return acc

    # 数值微分和反向传播都要有，因为有一个梯度确认，确认反向传播写的对不对

    def numerical_gradient(self,x,t):
        """
        数值微分，用处是验证反向传播计算出来的梯度是否正确
        :param x: 网络输入层的x
        :param t: 正确标签。独热编码和非独热均可。建议使用非独热，节省内存
        :return: 返回值是一个梯度，以字典的形式保存。使用一个变量接收该字典，即可调用出梯度
        """
        loss=lambda w: self.loss(x,t)

        grads= {
            "W1": numerical_gradient(loss, self.params["W1"]),
            "b1": numerical_gradient(loss, self.params["b1"]),
            "W2": numerical_gradient(loss, self.params["W2"]),
            "b2": numerical_gradient(loss, self.params["b2"]),
            "W3": numerical_gradient(loss, self.params["W3"]),
            "b3": numerical_gradient(loss, self.params["b3"])
        }

        return grads

    def gradient(self,x,t):
        """
        反向传播求梯度，输入是网络层输入和正确标签（独热或非独热编码均可。建议非独热，节省内存）
        :param x: 网络输入层的输入x
        :param t: 正确标签
        :return: 返回计算好的W，b的梯度。返回形式是字典对象。用一个变量接收该字典，即可调用出来
        """
        # 梯度下降方法求梯度的思想就是，一次前向传播，前向传播将所有信息都保存到内存里面
        # 然后再来一次反向传播，利用保存到内存里面的信息，一次性把梯度都计算出来

        # 前向传播，保存信息。或者说，获得并且保存当前前向传播的各层的输入
        L=self.loss(x,t)

        # 反向传播，求出梯度
        dout=1
        dout=self.lastLayer.backward(dout)
        layers=list(self.layers.values())
        layers.reverse()    # 反向传播，反转列表
        for layer in layers:
            dout=layer.backward(dout)   # Affine层等会自动保存梯度到字典，循环结束后只需要字典获取梯度即可

        grads= {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
            "W3": self.layers["Affine3"].dW,
            "b3": self.layers["Affine3"].db
        }

        return grads

