"""
实现Affine仿射变换和softmax交叉熵损失函数层
"""

"""
事实上，书上说的那些内容没必要用雅可比矩阵来理解，书上单纯就是，逐元素使用链式法则，再将结果整理成矩阵的形式

我好像知道为什么要用转置了。前向传播的时候，我们关心的是x1和x2一起与权重w作用的情况。
反向传播的时候，我们关心的是梯度，是单个x1或x2作用的情况，所以我们要转置W
值得注意的是，梯度的形状与原矩阵的形状相同。比如，X的形状与X梯度的形状相同，W的形状与W梯度的形状相同
通过形状可以快速确定乘法具体表达式

Y=XW+B，对X和对B求导，有相应的公式。公式为：dX=dL@W.T,dW=X.T@dL
对B求导，Y对B的局部导数就是1

同一个参数B，被同一个批次内的所有图片都使用过，又有广播机制，所以要把一列的L对Y的导数都加起来。（本来是L对Y导数乘以Y对B导数，由于Y对B导数是1，所以就只需要求L对Y导数即可）

"""

import numpy as np

def softmax(x):
    if np.ndim(x)==1:
        x=x.reshape(1,x.size)

    c=np.max(x,axis=1,keepdims=True)
    exp_x=np.exp(x-c)
    exp_sum=np.sum(exp_x,axis=1,keepdims=True)

    return exp_x/exp_sum

def cross_entropy_loss(y,t):

    if np.ndim(y)==1:
        y=y.reshape(1,y.size)
        if t.size==y.size:
            t=t.reshape(1,t.size)

    delta=1e-7
    batch_size=y.shape[0]

    if t.size==y.size:
        return -np.sum(t*np.log(delta+y))/batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size),t]+delta))/batch_size


class Affine:
    def __init__(self,W,b):
        # 注意，这个b也是一个矩阵。只不过写成小b而已
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None


    def forward(self,x):
        """
        四维张量的处理思路也是转换为二维，然后直接用矩阵乘法处理吧？
        假如输入是(N,C,H,W),
        那么可以x.reshape(N,-1)
        不过在工业实践中，应该不会把展平恢复这个功能写进Affine线性处理层，所以这里就认为输入的x都是二维的
        """
        self.x=x
        out=x@self.W+self.b

        return out

    def backward(self,dout):
        dx=dout@self.W.T        # 反向传播返回的梯度
        self.dW=self.x.T@dout
        self.db=np.sum(dout,axis=0)     # 其实dout就是y-t

        return dx


"""
可以通过数学推导，softmax_with_loss求梯度，结果刚好是y-t。也就是说，梯度刚好是预测概率减去真实概率。（感觉这个也可以称之为误差了）
其中，y是softmax的结果。
"""

class SoftmaxWithLoss:

    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None      # 实际训练中我们有时候会打印loss，所以需要存loss

    def forward(self,x,t):
        self.y=softmax(x)
        self.t=t
        self.loss=cross_entropy_loss(self.y,self.t)
        return self.loss

    def backward(self,dout=1):

        batch_size=self.y.shape[0]
        if self.t.size==self.y.size:
            dx=(self.y-self.t)/batch_size

        else:
            dx=self.y.copy()
            dx[np.arange(batch_size),self.t]-=1
            dx=dx/batch_size

        return dx







