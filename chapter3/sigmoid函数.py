# 让我们来画一画sigmoid函数图像
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,5,0.05)
y=1/(1+np.exp(-x))

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(y=0,color='r')
plt.axvline(x=0,color='r')
plt.show()

# 图像画好了。但是有一个看起来很严重的问题。如果输入在-1到1这个范围内，那这个图像就近似为线性，而线性的激活函数就几乎没意义了
# 不巧的是，sigmoid的输出却恰好是0-1，如果是多层sigmoid函数都做激活函数，可想而知，最少网络训练收敛慢，而且有线性风险，表达能力弱


