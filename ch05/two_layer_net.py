# coding: utf-8
# 导入系统与操作系统接口模块，用于文件路径操作
import sys, os
# 将父目录添加到模块搜索路径，以便于导入 common 目录下的自定义模块
sys.path.append(os.pardir)
import numpy as np
from common.layers import *  # 导入所有自定义层实现（如 Affine, Relu, SoftmaxWithLoss 等）
from common.gradient import numerical_gradient  # 导入数值梯度计算函数
from collections import OrderedDict  # 有序字典，用于保持层的顺序


class TwoLayerNet:
    """
    两层全连接神经网络（输入层 -> 隐藏层 -> 输出层）
    隐藏层使用 ReLU 激活函数，输出层为 softmax 配合交叉熵损失
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化网络结构与参数
        :param input_size: 输入层神经元数量（特征维度）
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出层神经元数量（分类类别数）
        :param weight_init_std: 权重初始化时的标准差（默认0.01，采用小随机数初始化）
        """
        # ---------- 初始化权重参数 ----------
        # 参数字典，存放网络所有的可训练参数（W1, b1, W2, b2）
        self.params = {}
        # 第一层权重：形状为 (input_size, hidden_size)，服从正态分布，标准差为 weight_init_std
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层偏置：形状为 (hidden_size,)，初始化为全0
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层权重：形状为 (hidden_size, output_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层偏置：形状为 (output_size,)
        self.params['b2'] = np.zeros(output_size)

        # ---------- 生成网络层的实例 ----------
        # 使用有序字典存储各层，保证前向/反向传播时顺序正确
        self.layers = OrderedDict()
        # 第一层仿射变换（全连接）：y = x * W1 + b1
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 激活函数层：ReLU
        self.layers['Relu1'] = Relu()
        # 第二层仿射变换：y = x * W2 + b2
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 最后一层为 Softmax 与交叉熵损失的综合层，用于训练时计算损失值
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        对输入数据进行前向传播，得到预测分数（未经 softmax）
        :param x: 输入数据，形状为 (batch_size, input_size)
        :return: 输出分数，形状为 (batch_size, output_size)
        """
        # 按照 layers 中定义的顺序依次执行每一层的前向计算
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        计算网络的损失函数值
        :param x: 输入数据
        :param t: 监督数据（真实标签，one-hot 编码或类别索引形式均可，由 SoftmaxWithLoss 内部处理）
        :return: 损失值（标量）
        """
        # 先通过 predict 得到 softmax 层的输入分数 y
        y = self.predict(x)
        # 将分数与标签送入最后一层，计算并返回交叉熵损失
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        计算模型在给定数据上的分类准确率
        :param x: 输入数据
        :param t: 监督数据（one-hot 编码或类别索引）
        :return: 准确率（0~1 之间的浮点数）
        """
        # 获取预测分数并进行前向传播
        y = self.predict(x)
        # 取每行中分数最高的位置作为预测类别
        y = np.argmax(y, axis=1)
        # 如果监督数据 t 是 one-hot 编码（二维），则也转换为一维类别索引
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        # 计算预测正确的样本数占比
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        使用数值微分（中心差分）计算各参数的梯度
        该方法通常用于梯度检查（验证解析梯度的正确性），速度较慢
        :param x: 输入数据
        :param t: 监督数据
        :return: 包含各参数梯度的字典：grads['W1'], grads['b1'], grads['W2'], grads['b2']
        """
        # 定义一个以参数为变量的损失函数闭包，固定数据 x 和 t
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # 对第一层权重 W1 进行数值梯度计算
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 第一层偏置 b1
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第二层权重 W2
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 第二层偏置 b2
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """
        使用误差反向传播法（解析梯度）计算各参数的梯度
        比数值微分计算速度快，是实际训练中使用的梯度求法
        :param x: 输入数据
        :param t: 监督数据
        :return: 包含各参数梯度的字典
        """
        # ---------- 前向传播 ----------
        # 调用 loss 方法执行完整前向计算，并更新各层内部状态（如中间变量缓存）
        self.loss(x, t)

        # ---------- 反向传播 ----------
        # 反向传播的起始梯度信号，通常设为 1（因为损失函数关于自身的导数为 1）
        dout = 1
        # 首先通过最后一层（SoftmaxWithLoss）反向传播
        dout = self.lastLayer.backward(dout)

        # 按照前向传播相反的顺序获取各层，并执行反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # ---------- 提取各参数的梯度 ----------
        grads = {}
        # 从 Affine1 层中取出当前批次下计算好的 dW 与 db
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # 从 Affine2 层中取出 dW 与 db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads