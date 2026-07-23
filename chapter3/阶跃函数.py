# 下面让我们来画一画阶跃函数
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    # 关于np.astype(int)，这个方法可以把bool类型转换为0或1
    y = x > 0
    return y.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x=np.arange(-5,5,0.1)
y1=step_function(x)
y2=sigmoid(x)
plt.plot(x,y1,linestyle='--',color='r')
plt.plot(x,y2)

# plt.axhline(y=0,color='r')
# plt.axvline(x=0,color='r')
plt.show()
