# 下面画一画relu函数

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.1)
y=relu(x)
plt.plot(x,y)
plt.show()