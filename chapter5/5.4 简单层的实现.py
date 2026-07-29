"""
下面是反向传播的简单层实现，包括乘法层和加法层

乘法层叫 MulLayer，加法层叫 AddLayer

首先，我们需要先声明，为了实现梯度反向传播，整个网络包括前向传播，都是由MulLayer等这样的小单元组成的
然后，以MulLayer为例子，这样的小单元只有两个功能：
一是前向传播forward，输入x和y，并且将当前的x和y存入内存，然后输出乘积结果out
二是反向传播backward，输入下一层反馈的梯度，返回本层的对x和y的梯度（因为这个单元的输入只有x和y，所以要么是对x的梯度，要么是对y的梯度）

forward和backward这两个功能，是相对较为独立的

"""

"""
需要注意的是，在数学上dx表示x的微分，但是在机器学习的代码里面，为了方便，dx表示x的导数（或梯度）
"""

"""
还有就是，一般就是mul和add层写成一个独立的层，其他的log、exp、取倒数什么的，一般都不会写成一个独立的层，而是封装
"""

import numpy as np

class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self,x,y):
        self.x = x
        self.y = y

        out=x*y

        return out

    def backward(self,dout):

        dx=self.y*dout
        dy=self.x*dout

        return dx,dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out=x+y
        return out

    def backward(self,dout):
        """
        加法就好像一个分发器，传过来的梯度误差dout，都一比一地分发给dx和dy
        :param dout:
        :return:
        """
        dx=dout
        dy=dout

        return dx,dy

class ExpLayer:
    def __init__(self):
        self.out=None

    def forward(self,x):
        self.out=np.exp(x)
        return self.out

    def backward(self,dout):
        dx=dout*self.out
        return dx



apple=100
apple_num=2
tax=1.1

# layer
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()

apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)

print(price)

dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
print(dapple,dapple_num,dtax,dprice)

print("--"*10+"分割线"+"--"*10)
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1

# layer
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()

# forward
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange,orange_num)
apple_orange_price=add_apple_orange_layer.forward(apple_price,orange_price)
price=mul_tax_layer.forward(apple_orange_price,tax)

print(price)

# backward
dprice=1
dapple_orange_price,dtax=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange_layer.backward(dapple_orange_price)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)

print(dapple_num,dapple,dorange,dorange_num,dtax)