"""
轻舟已过万重山，学了十多天，终于学到这一步了
"""

import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from my_network import ThreeLayerNet,SGD


def gradient_check():
    # 生成小批量随机数据
    x=np.random.randn(5,784)
    t=np.array([1,2,3,4,5])     # 模拟非独热编码

    # 初始化网络实例，默认参数
    net=ThreeLayerNet()
    grad_numerical=net.numerical_gradient(x,t)
    grad_backprop=net.gradient(x,t)

    for key in grad_numerical.keys():
        # 直接打印的话，值就太多了，所以需要求平均值
        diff=np.average(np.abs(grad_numerical[key]-grad_backprop[key]))
        print(f"{key}: {diff}")


# gradient_check()      # 梯度确认。正常训练注释掉

def train():
    # 获取输入数据
    (train_img,train_label),(test_img,test_label)=load_mnist(normalize=True,one_hot_label=False,flatten=True)

    # 获取网络实例，默认参数
    net=ThreeLayerNet()

    # 下面是超参数设置
    iters_num=180000       # 迭代次数一万二，即20个epoch。这里一万二指的是一万二次的梯度更新
    train_size=train_img.shape[0]
    batch_size=100
    optimizer=SGD(lr=0.01)

    train_loss_list=[]
    test_loss_list=[]
    train_acc_list=[]
    test_acc_list=[]

    iter_per_epoch=max(train_img.shape[0]//batch_size,1)

    # 下面是batch实现的训练，每隔一个epoch做一次记录
    cnt = 0     # 作为epoch的计数
    for i in range(iters_num):
        # 获得batch，用掩码
        batch_mask=np.random.choice(train_img.shape[0],batch_size)
        x=train_img[batch_mask]
        t=train_label[batch_mask]

        # 梯度计算
        grads=net.gradient(x,t)

        # 梯度更新
        optimizer.update(net.params,grads)

        # 记录loss
        loss=net.loss(x,t)
        train_loss_list.append(loss)


        if i%iter_per_epoch==0:

            test_loss_list.append(net.loss(test_img,test_label))
            test_acc_list.append(net.accuracy(test_img,test_label))
            train_acc_list.append(net.accuracy(train_img,train_label))
            print(f"第{cnt}个epoch，当前loss是：{net.loss(train_img,train_label)}")
            cnt += 1

        elif i==iters_num-1:
            test_loss_list.append(net.loss(test_img,test_label))
            test_acc_list.append(net.accuracy(test_img,test_label))
            train_acc_list.append(net.accuracy(train_img,train_label))
            print(f"第{cnt}个epoch，当前loss是：{net.loss(train_img,train_label)}")


    with open("train_loss.txt", "w") as f:
        for item in train_loss_list:
            f.write(f"{item}\n")

    with open("test_loss.txt", "w") as f:
        for item in test_loss_list:
            f.write(f"{item}\n")

    with open("train_acc.txt", "w") as f:
        for item in train_acc_list:
            f.write(f"{item}\n")

    with open("test_acc.txt", "w") as f:
        for item in test_acc_list:
            f.write(f"{item}\n")







"""
后记心得

经过实验，我发现同样的梯度更新的次数，batch=1在test上的准确率是85%，而batch=100在test上的准确率是91%
这说明，小批量的batch更能近似代表整个网络，数学是上的大数定律也支持这一论断
而batch=1的，每次更新都是根据当前图片情况来更新，但是对这个分类任务来说，每一张图片的标签可能不同。
比如，第一张图片的标签是1，那么梯度更新后，模型会往预测出1的方向靠近，那么势必会稍微远离预测其他结果的方向
再然后，虽然batch=1和batch=100更新的次数一样，但是batch=100，是实打实的看过的图像是batch=1的一百倍。每次更新也是取batch=100张图片的综合情况（梯度）来更新
还有就是，batch=1，容易引入噪声。如果某一张图片很难认出来，跟噪声一样，那么这张图片容易把权重带歪
"""

"""
实测发现，Relu比Sigmoid强大太多了。Relu在epoch=8的时候，loss就开始明显下降了，而Sigmoid在epoch=50的时候，loss才开始缓慢下降
而且，Relu不仅速度比Sigmoid快，上限也高
速度快应该是因为梯度在传递过程中不会损失，完整传递了，梯度更新快
上限高应该是，Relu的稀疏表达，更容易找出图像的本质特征，更容易让后面的Affine+softmax进行线性分类
神经元就应该是这样的。某一特定的特征，应该是某一特定区域神经元在工作，人脑也应该是这样的

但是Relu的缺点也不可忽视。Relu存在神经元死亡问题。很显然，假如一开始初始化权重的时候，由于Relu是max(0,x)，万一初始化的时候x就小于0，那初始化的时候神经元就死了。
而我们初始化采用的是高斯分布，0为对称轴，偏置b也初始化为0，所以理论上来说，开局就死一半神经元了
"""

"""
我做了四组实验，分别是对b进行高斯随机初始化，b=0，b=0.1，b=0.01，结果如下
b随机，收敛最慢。在epoch=8的时候，才开始有明显变化。推测原因是b随机化，有不少输入是负数，导致开局死更多神经元，导致学到特征需要花更长的时间。最终的test_acc为0.9702左右
b=0，比b随机略好。在epoch=7时候，loss开始有明显变化，最终的test_acc为0.9702左右
b=0.1，比b=0好，在epoch=4的时候，loss就开始有明显变化，最终的test_acc为0.9702左右
b=0.01，在epoch=5时，loss开始有明显变化。虽然前期寻找特征可能花了更多时间，但是最终的test_acc到达了0.9722。有可能是b=0.1破坏了更多的稀疏性，也有可能是单纯的运气
"""
