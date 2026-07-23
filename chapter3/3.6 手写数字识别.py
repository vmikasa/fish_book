"""
终于来到手写数字识别的这个案例了
一般地，对于分类问题来说，输出层的神经元数量就等于要分类的类别
这里是0-9这10个数字，所以输出层需要有10个数字，也就是输出层有10个神经元
这里，我们假设学习（或者说训练）已经结束，直接用学习到的参数，来实现神经网络的推理，也就是前向传播，最后输出预测的结果

关于输入层，因为输入的像素是28*28=784，所以输入层应该有784个神经元，每一个像素点对应一个神经元。也就是说，第一层的权重W1，应该有784个
书中说隐藏层1有50个神经元，也就是说，有50个神经元，每个神经元都会接受输入层的784个像素点，所以都有784个权重和偏置，表示为W1和B1
书中说隐藏层2有100个神经元，也就是说，有100个神经元，每个神经元都会接受上一层50个神经元的输出，所以说有50个权重和偏置，表示为W2和B2
书中说输出层有10个，也就是说，有10个神经元，每个神经元都会接受上一层100个神经元的输出，所以说每个神经元理论上有100个权重和偏置，表示为W3和B3

我们这里的隐藏层1和隐藏层2，以及输出层，都是用softmax激活函数来处理的
对于隐藏层1来说，sigmoid处理的是784个输入a1，然后输出为z1
对于隐藏层2来说，sigmoid处理的是100个a2，然后输出z2
对输出层来说，softmax是处理10个a3，然后输出z3（这里z3就是y）

显然，softmax输出的y，是一个数组。我们需要的是概率最大的那个值，所以需要借助外力

这个外力就是np.argmax()。argmax，arg是argument的缩写，即变量，max就是最大值
np.argmax()，会返回数组最大值所对应的下标。配套的还有一个np.argmin()

输出层的10个神经元，刚好对应数字0到9，哪一个下标对应的概率最大，就可以认为预测的就是这个数。（因为下标也刚好是0到9，与数字是一一对应的）


这里再次说一下。训练就是学习权重参数，推理就是使用学习到的权重参数来进行预测
"""

import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle


def get_data():
    """
    pkl文件，是pickle的缩写，是python专属的序列化二进制缓存文件，作用是把python中的任意对象（比如字典、numpy数组、自定义类、数据集、模型等）打包成二进制，保存到本地磁盘
    序列化就是按照固定的规则，将文件对象转换为二进制
    这里的pkl存的就是模型文件，或者说权重偏置，里面包含了网络层数、网络结构、每层权重偏置和激活函数等参数
    """
    (train_image,train_label),(test_image,test_label)=load_mnist(normalize=True,flatten=True,one_hot_label=False)
    # load_mnist的返回值是(训练图像, 训练标签), (测试图像, 测试标签),是numpy的数组
    return test_image,test_label



def init_network():
    with open(r"../dataset/sample_weight.pkl",'rb') as f:
        network=pickle.load(f)

    return network
    # 这里的network应该是一个字典，因为后文的写法是network['']=xxx

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c=np.max(x)
    x_exp=np.exp(x-c)
    x_exp_sum=np.sum(x_exp)
    return x_exp/x_exp_sum

def predict(x,network):
    """

    :param x: 输入的x，其实是flatten图片数组，是一维的。只能输入一张
    :param network: 这个network其实是一个字典，就是获得的网络层的权重偏置参数
    :return: 返回的是经过softmax处理过的数组，一共有10个数，需要自行处理
    """
    b1,b2,b3=network["b1"],network["b2"],network["b3"]
    W1,W2,W3=network["W1"],network["W2"],network["W3"]
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    z3=softmax(a3)
    y=z3

    return y

img,label=get_data()
network=init_network()
accuracy_cnt=0
for i in range(len(img)):
    temp=predict(img[i],network)
    y=np.argmax(temp)
    if y==label[i]:
        accuracy_cnt+=1

print(f"预测的准确率是：{100*accuracy_cnt/len(img)}%")






