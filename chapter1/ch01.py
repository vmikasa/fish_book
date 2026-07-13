import numpy as np

x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

print(x)
print(x*y)

A=np.array([[1,2],[3,4]])
print(A)

print(A.shape)

print()
print()

# 这里是numpy中的广播机制
A=np.array([[1,2],[3,4]])
B=np.array([10,20])

print(A*B)
for i in range(3):
    print()

X=np.array([[51,55],[14,19],[0,4]])
print(X)

print(X[0])
print(X[0][1])
for i in range(3):
    print()


for row in X:
    print(row)

# np里面的flatten方法，flatten的意思就是压扁

X=X.flatten()

print(X)
print(X>15)
print(X[X>15])      # 倒是一个不错的快速比大小的方法

# 下面来看matplotlib

import matplotlib.pyplot as plt

# plt.figure()的意思是新建画布
x=np.arange(0,6,0.1)    # arange常用于生成等差数列，返回是一个列表
y1=np.sin(x)
y2=np.cos(x)
plt.xlabel("x")
plt.ylabel("y")

plt.plot(x,y1,label="sin(x)")
plt.plot(x,y2,label="cos(x)",linestyle="--",color="red")

plt.legend()
plt.show()


from matplotlib.image import imread

img=imread("../dataset/lena.png")
plt.imshow(img)
plt.show()