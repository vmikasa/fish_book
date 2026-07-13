# 书本3.3节

import numpy as np
import matplotlib.pyplot as plt

A=np.array([1,2,3,4])
print(A)
print(np.ndim(A))
print(np.shape(A))

for i in range(3):
    print("--"*10)

A=np.array([[1,2,3],[4,5,6]])
B=np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(np.shape(B))

C=np.dot(A,B)
print(C)