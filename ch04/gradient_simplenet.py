# coding: utf-8
# 本代码演示了一个简单的全连接网络，并使用数值微分计算权重的梯度
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    """一个简单的全连接神经网络，只有一层线性变换，用 softmax 与交叉熵误差进行学习。"""
    def __init__(self):
        # 权重初始化为形状 (2,3) 的随机高斯分布值
        self.W = np.random.randn(2,3)

    def predict(self, x):
        """前向传播：计算输入 x 与权重 W 的线性组合。"""
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        计算损失函数值。
        x: 输入数据，形状为 (2,) 的向量
        t: 真实标签，采用 one-hot 编码，形状为 (3,) 的向量
        返回交叉熵误差。
        """
        z = self.predict(x)          # 线性变换后的得分
        y = softmax(z)               # 通过 softmax 转为概率分布
        loss = cross_entropy_error(y, t) # 计算交叉熵误差

        return loss

# 输入样本：两个特征
x = np.array([0.6, 0.9])
# 正确标签：第三类为 1，其余为 0 的 one-hot 向量
t = np.array([0, 0, 1])

# 实例化网络
net = simpleNet()

# 定义以权重 W 为自变量的损失函数，用于计算梯度
f = lambda w: net.loss(x, t)
# 使用数值微分求损失函数关于权重的梯度
dW = numerical_gradient(f, net.W)

# 输出梯度矩阵
print(dW)