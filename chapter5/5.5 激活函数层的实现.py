"""
激活函数层的实现
"""

import  numpy as np

class Relu:
    def __init__(self):
        self.mask=None

    def forward(self,x):
        self.mask=(x<=0)    # 这里数组x可以直接和数字比较，并且返回bool值。但是原生的python列表不可以，须知
        out=x.copy()
        out[self.mask]=0
        return out

    def backward(self,dout):
        dx=dout.copy()
        dx[self.mask]=0
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

