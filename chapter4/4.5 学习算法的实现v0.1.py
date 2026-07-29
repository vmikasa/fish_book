"""
终于来到这一步了，让我们亲手实现手写数字识别的模型吧！
先写一个不用batch实现，也不用epoch实现的网络吧
"""

"""
书上是两层网络，但是我想实现一个三层网络，我尝试一下
"""

import numpy as np


def sigmoid(x):
    """
    这里是不带批次的sigmoid函数
    :param x:传入的数组
    :return: y，y是函数返回值
    """
    y=1/(1+np.exp(-x))
    return y

def softmax(x):
    """
    这里暂时是用无batch的输入，后续会升级
    :param x: 输入的一维数组
    :return: 压制后的概率
    """
    c=np.max(x)
    x_exp=np.exp(x-c)
    x_exp_sum=np.sum(x_exp)

    return x_exp/x_exp_sum

def cross_entropy_loss(y,t):
    """
    计算交叉熵损失函数，适用于无batch输入
    :param y: 模型预测输出的概率
    :param t: 正确的标签，可以是one-hot，也可以是非one-hot
    :return: 返回交叉熵损失函数的值
    """

    delta=1e-7

    # 假如是二维的矩阵，因为这里不做batch，所以默认是一维的数组，所以先不做这个判断

    # 假如是一维数组
    # 假如是one-hot标签
    if y.size==t.size:
        return -np.sum(t*np.log(y+delta))
    else:
        return -np.sum(np.log(y[t]+delta))

def numerical_gradient(f,x):
    """
    数值微分梯度计算的函数。需要传入待计算梯度的函数f和当前的值x
    :param f: 待计算梯度的函数，一般是损失函数。这个f一般是通过前向传播+交叉熵损失函数计算得来
    :param x: 这里的x是，一个矩阵
    :return: 返回也是梯度矩阵，形状跟x相同
    """

    h=1e-4      # 先定义一个微小量h，用于中心差分
    grad=np.zeros_like(x,dtype=np.float64)   # 再复制一个跟x一模一样的矩阵
    x=np.asarray(x,dtype=np.float64)

    index=np.ndindex(x.shape)
    for i in index:
        temp=x[i]       # 取temp暂存x[i]
        # 先计算f(x+h)
        x[i]=temp+h
        fh1=f(x)        # 没弄错的话，这里把x带入f(x)，就会进行一次前向传播+损失函数计算吧？那么这个f是什么呢

        # 再计算f(x-h)
        x[i]=temp-h
        fh2=f(x)

        # 再计算索引i的梯度
        grad[i]=(fh1-fh2)/(2*h)

        x[i]=temp

    return grad



class ThreeLayerNet:
    """
    这个类里面，包括了以下内容：
    初始化权重参数，外部可以访问
    前向传播 self.forward 方法
    交叉熵损失函数计算方法 self.loss
    准确率计算方法 self.accuracy
    数值微分计算方法 self.numerical_gradient

    """
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size,weight_init_std=0.01):
        """
        不算第零层，则一共三层神经网络，则有三套W和b
        :param input_size: 显然，这个input_size表示输入的x，这个x可以是带批次的。代表第0层神经元网络的神经元个数，或者说输入的个数。本次演示的是不带批次的x
        :param hidden_size1: 这是第隐藏层1，也就是第一层神经网络的神经元个数
        :param hidden_size2: 这是隐藏层2，也就是第二层神经网络的神经元个数
        :param output_size: 这是输出层，也就是第三层神经网络的神经元个数
        :param weight_init_std: 这是权重标准差，默认是0.01，作用是防止sigmoid等激活函数初始饱和
        """
        # init的第一步先初始化一套权重参数
        self.params= {
            # 创建参数字典，初始化权重
            "W1": weight_init_std * np.random.randn(input_size, hidden_size1),
            "b1": np.zeros(hidden_size1),

            "W2": weight_init_std * np.random.randn(hidden_size1, hidden_size2),
            "b2": np.zeros(hidden_size2),

            "W3": weight_init_std * np.random.randn(hidden_size2, output_size),
            "b3": np.zeros(output_size)
        }

    def forward(self,x):
        """
        前向传播的函数，或者说方法
        :param x:输入的数组，或者说输入的神经元
        :return:返回前向传播y的预测结果
        """
        W1,W2,W3=self.params["W1"],self.params["W2"],self.params["W3"]
        b1,b2,b3=self.params["b1"],self.params["b2"],self.params["b3"]

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
        计算交叉熵损失函数，返回值是交叉熵损失函数的值
        :param x: x是网络输入的数组，当前版本代码是默认无batch输入
        :param t: t是标签，cross_entropy_loss函数已做处理，one-hot或非one-hot标签均可
        :return: 返回值是交叉熵损失函数的值
        """
        y=self.forward(x)
        loss=cross_entropy_loss(y,t)
        return loss

    def accuracy(self,x,t):
        """
        这个函数是用来计算预测准确率的
        :param x: 输入的一维数组，这个版本是没有batch的
        :param t:标签
        :return:由于这个是无batch版本，所以返回值就是bool值0或1
        """
        y=self.forward(x)

        if y.size==t.size:
            y=np.argmax(y)
            t=np.argmax(t)

            return y==t
        else:
            y=np.argmax(y)
            return y==t


    def numerical_gradient(self,x,t):
        """
        这里的数值微分梯度计算，必须要可以计算二维矩阵
        注意，外部已经有一个numerical_gradient梯度计算的函数了，这个内部应该调用这个梯度计算函数

        这里的梯度grad应该也是一个字典。求梯度应该只能一层一层求，一个网络层求一个W和一个b

        我们是用损失函数作为梯度计算的标准的，所以说，对于外部的numerical_gradient()来说，传入的参数应该是损失函数loss和待计算梯度的矩阵

        :param x: 这里的x就是输入的数组x
        :param t:
        :return: 梯度计算结果grad
        """
        loss=lambda w: self.loss(x,t)   # 当前loss函数，假如x和t固定不变，那么显然，loss就是关于W和b的函数了

        grad= {
            "W1": numerical_gradient(loss, self.params["W1"]),
            "b1": numerical_gradient(loss, self.params["b1"]),

            "W2": numerical_gradient(loss, self.params["W2"]),
            "b2": numerical_gradient(loss, self.params["b2"]),

            "W3": numerical_gradient(loss, self.params["W3"]),
            "b3": numerical_gradient(loss, self.params["b3"])
        }

        # 注意，计算出来的grad是根据self.params的字典计算出来的，但是，这里并没有对self.params进行更新
        # 后续应该有对self.params进行更新的操作

        return grad


# 下面开始训练网络

# 先获取数据

import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle

(train_img,train_label),(test_img,test_label)=load_mnist(normalize=True,one_hot_label=True,flatten=True)

# 创建实例对象，获得神经网络
net=ThreeLayerNet(784,50,100,10,0.01)

# 定义学习率
lr=0.01

loss_list=[]

cnt=0
# 重复过程，重复数值微分计算梯度+梯度跟新权重的过程：
for i in np.random.choice(train_img.shape[0],100000):
    # 数值微分计算梯度
    grad = net.numerical_gradient(train_img[i], train_label[i])

    # 梯度更新权重
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        net.params[key] = net.params[key] - lr * grad[key]

    # 每更新一轮权重后，记录一次loss，并且打印一下
    loss_list.append(net.loss(train_img[i], train_label[i]))
    cnt+=1
    print(f"第{cnt}次训练，当前的loss是{loss_list[cnt-1]}")

# 最后应该得到一组训练好的权重参数

# 下面是保存这个权重和loss记录
# 保存loss到txt
with open("loss_record.txt", "w", encoding="utf-8") as f:
    for loss in loss_list:
        f.write(f"{loss}\n")

# 保存权重到pkl
with open("weight","wb") as f:
    pickle.dump(net.params,f)





"""
后记

起初，关于数值微分，我的想法是写三个版本。
首先是用batch=1来训练，训练10万次，mnist数据集有6万张图片，10万次相当于是1.67个epoch。
然后再用batch=100来训练，训练100次，等价为也是1.67个epoch。

在写第一个batch=1版本训练的时候，我发现一个问题，，训练速度着实太慢了。
图片是28*28的像素，一维化后是784个像素点，即需要784个输入神经元。第一层神经网络设计是50个神经元，第二层神经元设计是100个神经元，输出层设计是10个神经元
那么，第一层网络参数是有50*784=39200个W权重参数，50个b参数；
第二层网络参数是100*50=5000个W参数，100个b参数
输出层网络参数是10*100=1000个W参数，10个b参数

那么，所有的网络层总计参数是45360个参数。
而数值微分为了确保精确，一般是采用中心差分的，需要计算f(x+h)和f(x-h)，那么求一个参数的偏导，就需要2次前向传播
进行一次完整的梯度计算，求出这45360个参数的偏导，则需要2*45600=90720次计算
这仅仅是一张图片一次的梯度更新，真正训练，一张图片最少需要10次的梯度更新才可以。
而mnist数据集有6万张图片。这数以亿计的计算量，在当时那个年代几乎是不可能完成的。就连我现在的cpu，计算能力是当时最强的cpu的数万倍，训练1.67个epoch，也要二十多个小时
这个问题先辈们肯定也碰到过(或许这也可能是当时人工智能寒冬的原因之一)，所以想到从论文中回顾历史，看看当时的人们是怎么解决的


然后，我写了第二个batch=100版本来进行训练。
这个版本的训练速度大幅度提升，目测训练速度最少提升了5倍以上。（不过仍然是数值微分）

但是，当1.67个epoch训练完毕后，我发现另外一个问题。
经过前向推理后，我发现预测准确率几乎还是10%，没有变化。
而batch=1版本，预测准确率却是85%。

这很奇怪，同样是1.67个batch，为什么一次看一张图片，计算出来梯度更新，反而不如一次看一百张图片呢？
因为按照我之前的理解，一次看一百张图片，通过计算出来的平均损失函数来进行计算梯度，按理来说，得到的梯度是这一百张图片的平均梯度，梯度指向的方向应该更准确才对，为什么会这么大差别？
一次看一张图片计算出来的梯度，按理来说，这个梯度方向是随机的，方向指向应该有很大误差才对，但是为什么训练出来的结果，准确率更高？

我检查了batch=100版本的权重情况，发现，经过1.67个epoch的训练，权重与高斯初始化的随机权重几乎是一样的。
这说明权重几乎没有更新。

这很奇怪，训练了1.67个epoch了，训练量虽然不多，但是也不至于权重没有更新吧

于是进行了数学层面的检查。发现，由于我的权重初始化的weight_init_std=0.01，用链式法则检查，梯度一层一层传递，传到第一层的时候，几乎已经是0.01^3，也就是10的-6次方这个数量级了
这个数量级很少。而且，sigmoid函数，最大的导数值才0.25，经过三层传递，第一层只有0.25^3约等于0.01，也就是说，梯度传到第一层，数量级约是10的-8次方
虽然batch=1版本和batch=100版本都是等效为epoch=1.67，但是batch=1的版本，梯度是实打实的更新了十万次，参数按梯度下降也更新了10万次
batch=100的版本，虽然按理来说梯度也更新了10万次，但是参数按梯度下降只更新了1000次。
所以batch=100落后于batch=1，似乎是必然的，因为这个梯度更新根本更新不动


由此引发我的第二个问题。这仅仅是三层网络，就出现了如此严重的梯度消失，更复杂的网络，梯度消失问题必然会更严重。

我相信这个梯度消失问题也是前人碰到过的，那历史上的先辈是怎么解决的？

于是抱着这些疑问，以及前人一定碰到过这些问题的确信，我尝试去寻找并且去阅读了当年一些经典论文。第一篇是 efficient backprop

"""

