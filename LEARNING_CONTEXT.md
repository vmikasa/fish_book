# 鱼书学习恢复笔记

更新时间：2026-07-14

这是一份临时上下文笔记，用于在另一台电脑继续《深度学习入门：基于 Python 的理论与实现》（常称“鱼书”）的学习。

## 项目与环境

- 项目名称：`fish_book`。
- Conda 环境：`fish_book`，Python 3.11。
- 已使用的主要库：NumPy、Matplotlib、Pillow。
- PyCharm 项目解释器应指向本机的 `fish_book` Conda 环境；`.idea/` 没有提交，因此每台电脑需要各自设置解释器。
- PowerShell 中 Conda 已可激活。验证环境是否正确：

  ```powershell
  conda activate fish_book
  python --version
  ```

## Git 与跨电脑同步

- GitHub 仓库：<https://github.com/vmikasa/fish_book>
- 默认分支：`main`。
- 笔记本首次使用时，在 PyCharm 选择 `Get from VCS`，克隆上述仓库。
- 日常流程：开始学习前执行 `Git | Update Project`；修改后 `Commit`，再 `Git | Push`。
- `.gitignore` 忽略 PyCharm 配置、Python 缓存及重复的 MNIST 原始压缩包；`dataset/mnist.pkl` 已提交，因此克隆后可离线加载 MNIST。

## MNIST 与 NumPy

- `load_mnist()` 返回嵌套元组：

  ```python
  (x_train, t_train), (x_test, t_test) = load_mnist(...)
  ```

- `x_train` 是训练图像，`t_train` 是对应标签；标签是整数类别 `0` 到 `9`，不是概率排序。
- `flatten=True`：每张图像由 `(28, 28)` 展开为 `(784,)`；全连接网络常使用这种形式。
- `flatten=False`：保留图像空间结构；卷积网络通常需要保留结构。
- `normalize=True`：像素从 `0` 到 `255` 缩放为 `0.0` 到 `1.0`。训练和推理必须使用相同的预处理；显示原始图像时可使用 `normalize=False`。
- 二维数组遵循：`x[行, 列]`，`x[0]` 等价于 `x[0, :]`。
- `axis=1`：在列这一维上归约，得到每一行的结果；MNIST 批量输出形状为 `(batch_size, 10)` 时，用 `np.argmax(y, axis=1)` 得到每张图的预测类别。
- `axis=0`：在行这一维上归约，得到每一列的结果。

## 神经网络学习进度

已讨论的内容：

- 感知机、权重、偏置；偏置可看作常数输入 `1` 乘以权重 `b`。
- 数字电路中的逻辑门分析可迁移到感知机；异或可写为 `AB' + A'B`，单层感知机不能直接表示 XOR。
- 前向传播：`a = XW + b` 后经过激活函数得到输出。
- Sigmoid 输出在 `0` 到 `1`；权重与偏置可以大于 `1` 或为负数。Sigmoid 在饱和区梯度很小，深层网络可能出现梯度消失。
- ReLU 提供非线性，且在正半轴不易出现 Sigmoid 的饱和梯度问题。
- 全连接层：前一层每个单元连接后一层每个单元。
- 卷积层：输入仍是完整图像，但每个输出位置只读取局部区域；padding 是临时补零，不是原始输入像素或可学习神经元。

## 概率论与损失函数进度

- 概率与似然使用相同形式 `P(D | theta)`，区别在于谁被固定：
  - 固定参数 `theta`、考察数据 `D`：概率。
  - 固定已观察数据 `D`、考察参数 `theta`：似然函数 `L(theta; D)`。
- 硬币例子：若只知道 10 次中 7 次正面、3 次反面，似然可写为：

  ```math
  L(theta; D) = C(10, 7) theta^7 (1 - theta)^3
  ```

  若记录了完整的抛掷顺序，则不需要组合数。求最大似然估计时，组合数与 `theta` 无关，可用正比符号省略。
- 最大似然估计：

  ```math
  theta_hat = argmax_theta L(theta; D)
  ```

  找的是使似然最大的参数，不是最大的似然数值。
- 最大似然值是 `max_theta L(theta; D)`；最大似然估计值是 `argmax_theta L(theta; D)`。
- 取对数不改变最大值位置，因为 `log` 在正数范围严格递增；同时能把概率连乘变为对数相加。
- 已开始学习交叉熵：

  ```math
  H(t, y) = -sum_k t_k log(y_k)
  ```

  one-hot 标签下化为 `-log(模型给真实类别的概率)`。下一步应继续从“信息量/惊讶度 `-log q`”和“真实分布上的平均值”两步，慢慢推导交叉熵为何这样定义，以及它为何等价于最小化负对数似然。

## 常用 PyCharm 操作

- `Ctrl + Q`：快速查看函数文档。
- `Ctrl + B`：跳转到定义/源码。
- `Ctrl + Alt + Left`：跳回上一个位置。
- `Ctrl + Shift + I`：快速查看定义。
- `Ctrl + P`：在函数调用括号内查看参数提示。

