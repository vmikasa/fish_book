# 偏置b，bias，描述重要程度。公式b+w1*x1+w2*x2<=0，b+w1*x1+w2*x2>0。显然，b越大，那么整个式子越容易大于0，这个神经元就越容易为1
# 下面用这个偏置来实现感知机

import numpy as np

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tem=np.sum(x*w)+b

    if tem<=0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tem=np.sum(x*w)+b

    if tem<=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.3
    tem=np.sum(x*w)+b
    if tem<=0:
        return 0
    else:
        return 1

def NOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(NOR(0,0))
print(NOR(1,1))
print(NOR(0,1))
