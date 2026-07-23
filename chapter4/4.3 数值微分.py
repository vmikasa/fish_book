# 数值微分的一些例子

import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/2*h

def f1(x):
    return 0.01*x**2+0.1*x


x=np.arange(0,20,0.01)
y=f1(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

print(numerical_diff(f1,5))
print(numerical_diff(f1,10))