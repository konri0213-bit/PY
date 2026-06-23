# coding: utf-8
# 导入系统和操作系统模块，用于路径操作
import sys, os
# 将父目录加入系统路径，方便导入 common 目录下的模块
sys.path.append(os.pardir)
# 导入激活函数、损失函数等常用函数，例如 sigmoid, softmax, cross_entropy_error 等
from common.functions import *
# 导入数值微分计算梯度的函数
from common.gradient import numerical_gradient


class TwoLayerNet:
    """两层的全连接神经网络（输入层-隐藏层-输出层），用于分类任务"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化网络的权重和偏置参数
        :param input_size:   输入层神经元个数
        :param hidden_size:  隐藏层神经元个数
        :param output_size:  输出层神经元个数（类别数）
        :param weight_init_std: 权重初始化的标准差，用于高斯分布随机初始化
        """
        # 用字典存放所有可学习参数
        self.params = {}
        # 第一层权重 W1，形状为 (input_size, hidden_size)，用高斯分布初始化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层偏置 b1，形状为 (hidden_size,)，初始化为 0
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层权重 W2，形状为 (hidden_size, output_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层偏置 b2，形状为 (output_size,)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        进行前向传播，根据输入数据 x 计算各类别的预测概率
        :param x: 输入数据，形状为 (batch_size, input_size)
        :return:  输出层的 softmax 概率，形状为 (batch_size, output_size)
        """
        # 取出第一层和第二层的权重与偏置
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 第一层线性变换
        a1 = np.dot(x, W1) + b1
        # 第一层激活函数（sigmoid）
        z1 = sigmoid(a1)
        # 第二层线性变换
        a2 = np.dot(z1, W2) + b2
        # 输出层激活函数（softmax），得到分类概率
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """
        计算当前批次数据的交叉熵损失
        :param x: 输入数据，形状为 (batch_size, input_size)
        :param t: 监督标签（one-hot 编码），形状为 (batch_size, output_size)
        :return:  平均交叉熵误差（标量值）
        """
        # 先通过前向传播得到预测概率
        y = self.predict(x)
        # 使用交叉熵误差函数计算损失
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算当前批次数据的分类准确率
        :param x: 输入数据
        :param t: 监督标签（one-hot 编码）
        :return:  分类正确的样本比例（0~1 之间的浮点数）
        """
        # 前向传播得到预测概率
        y = self.predict(x)
        # 取出预测概率最大的类别索引（沿 axis=1 即类别方向取 argmax）
        y = np.argmax(y, axis=1)
        # 取出真实标签的类别索引（one-hot 转为整数类别）
        t = np.argmax(t, axis=1)

        # 比较预测与真实标签，求正确率
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        使用数值微分方法计算所有参数的梯度（速度较慢，通常用于梯度校验）
        :param x: 输入数据
        :param t: 监督标签
        :return:  字典，键为参数名，值为对应参数的数值梯度
        """
        # 定义一个以 W 为变量的损失函数闭包，用于 numerical_gradient 内部计算
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # 分别计算 W1, b1, W2, b2 的数值梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """
        使用误差反向传播算法高速计算梯度
        :param x: 输入数据，形状为 (batch_size, input_size)
        :param t: 监督标签（one-hot 编码），形状为 (batch_size, output_size)
        :return:  字典，键为参数名，值为对应参数的梯度
        """
        # 取出各层参数
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        # 当前批次样本数量
        batch_num = x.shape[0]

        # ---------- 前向传播 ----------
        # 第一层加权和
        a1 = np.dot(x, W1) + b1
        # 第一层激活值
        z1 = sigmoid(a1)
        # 第二层加权和
        a2 = np.dot(z1, W2) + b2
        # 输出概率
        y = softmax(a2)

        # ---------- 反向传播 ----------
        # 输出层误差信号：交叉熵 + Softmax 的简化反转误差，除以 batch_num 求平均
        dy = (y - t) / batch_num
        # W2 的梯度 = 隐藏层激活 z1 的转置 * 输出误差
        grads['W2'] = np.dot(z1.T, dy)
        # b2 的梯度 = 输出误差按列求和（对每个输出神经元加总所有样本的误差）
        grads['b2'] = np.sum(dy, axis=0)

        # 误差反向传播至隐藏层之前的线性部分 a1
        da1 = np.dot(dy, W2.T)
        # 乘以 sigmoid 函数的导数，得到关于 a1 的梯度
        dz1 = sigmoid_grad(a1) * da1
        # W1 的梯度 = 输入 x 的转置 * dz1
        grads['W1'] = np.dot(x.T, dz1)
        # b1 的梯度 = dz1 按列求和
        grads['b1'] = np.sum(dz1, axis=0)

        return grads