# 梯度...全微分...偏导数...

import numpy as np

def numerical_gradient(f,x):
    """

    :param f: 这里的f是一个函数
    :param x: 这里的x是一个数组，比如x[0]，x[1]，x[2]...，是函数f的自变量
    :return: grad，返回的是f的梯度
    """
    h=1e-4
    # 显然第一步是要知道f有多少个自变量，即x的长度
    # 知道x的长度是多少，直接用x.size就可以解决了
    # 然后需要解决的事情是，返回值grad应该也是一个数组
    # 返回值grad的数组形状应该跟x是一样的才对
    # 有一个函数是np.zeros_like()，可以克隆一个形状一模一样的数组出来
    grad=np.zeros_like(x)

    for i in range(x.size):
        # 先将第i个x自变量存起来
        temp=x[i]

        # 先计算f(x+h)
        x[i]=temp+h
        fh1=f(x)

        # 再计算f(x-h)
        x[i]=temp-h
        fh2=f(x)

        # 最后用f(x+h)-f(x-h)，除以2h，得出第i个x变量对应的梯度
        grad[i]=(fh1-fh2)/(2*h)

        # 最后，在退出循环之前，还需要把x[i]恢复原样
        x[i]=temp

    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    """
    这是梯度下降的函数。返回值肯定x，也就是沿梯度下降后的那个点
    :param f: 目标函数，或者说需要梯度下降的函数
    :param init_x: 传入的x，这里叫做初始化的x
    :param lr: 学习率，或者说梯度下降的速度
    :param step_num: 循环的次数，或者说梯度下降的次数
    :return: x，返回值是完成梯度下降后的那个点
    """

    x=init_x    # 注意，x是一个具体的数组，不是变量
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad

    return x
