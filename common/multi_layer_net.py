# coding: utf-8
import sys, os
# 为了导入父目录中的 common 模块，将父目录加入系统路径
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict
# 导入所有定义好的层类，如 Affine, Relu, Sigmoid, SoftmaxWithLoss 等
from common.layers import *
# 导入数值梯度计算函数
from common.gradient import numerical_gradient


class MultiLayerNet:
    """全连接的多层神经网络

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度，用于控制正则化项
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        # 记录网络结构参数
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)      # 隐藏层的数量
        self.weight_decay_lambda = weight_decay_lambda     # L2 正则化系数
        self.params = {}                                    # 保存权重和偏置参数的字典

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        # 激活函数映射表，根据字符串选择对应的激活层类
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        # 使用 OrderedDict 按顺序保存网络各层
        self.layers = OrderedDict()
        # 逐层添加隐藏层的仿射变换和激活函数
        for idx in range(1, self.hidden_layer_num+1):
            # 仿射层（全连接层），使用对应的权重和偏置
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 激活函数层
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        # 输出层的仿射变换（不接非线性激活函数，直接输出得分）
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        # 最后一层为 SoftmaxWithLoss，用于计算损失和进行反向传播
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        根据激活函数类型选择 He 或 Xavier 初始化，并从正态分布中采样权重，
        偏置统一初始化为零。

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        # 所有层的神经元数量列表，包括输入层、各隐藏层和输出层
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # 对每一层权重进行初始化
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # 根据字符串选择初始化时的缩放因子
            if str(weight_init_std).lower() in ('relu', 'he'):
                # He 初始化，适用于 ReLU 激活函数
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                # Xavier 初始化，适用于 sigmoid 或 tanh 激活函数
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            # 权重矩阵形状为 (前一层神经元数, 当前层神经元数)
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # 偏置向量初始化为 0
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        """前向传播，根据输入数据计算输出得分

        Parameters
        ----------
        x : 输入数据，形状为 (batch_size, input_size)

        Returns
        -------
        输出得分，形状为 (batch_size, output_size)
        """
        # 依次通过每一层进行前向计算
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数（包含 L2 权重衰减正则化项）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签（one-hot 编码或类别索引）

        Returns
        -------
        损失函数的值（标量），等于交叉熵损失加上权重衰减项
        """
        # 先得到预测输出
        y = self.predict(x)

        # 计算 L2 正则化损失： 0.5 * λ * Σ(W^2) ，对所有层的权重平方求和
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 总损失 = Softmax 交叉熵损失 + 权重衰减项
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        """计算分类准确率

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        准确率（0.0 ~ 1.0）
        """
        y = self.predict(x)
        # 找到预测得分最高的类别索引
        y = np.argmax(y, axis=1)
        # 如果教师标签为 one-hot 形式，转换为类别索引
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        # 计算正确预测的比例
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        利用数值微分法对每一个参数计算损失函数的梯度，
        通常用于梯度检查，速度较慢。

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # 定义用于梯度计算的损失函数，将参数作为函数输入
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # 对所有参数逐个计算数值梯度
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        采用反向传播算法高效地计算所有参数的梯度，
        同时将权重衰减正则化项的梯度一并加入。

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward: 执行前向传播并计算损失（包含正则化项），各层保存中间数据供反向传播使用
        self.loss(x, t)

        # backward: 从最后一层开始反向传播梯度
        dout = 1  # 损失关于自身的梯度为 1
        # SoftmaxWithLoss 层的反向传播
        dout = self.last_layer.backward(dout)

        # 将所有层按顺序反转，从输出层向输入层逐层反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定各参数梯度，并加上 L2 权重衰减对应的梯度 (λ * W)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # 仿射层中 dW 是反向传播得到的交叉熵损失梯度，加上正则化梯度
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            # 偏置没有正则化项，直接使用反向传播得到的 db
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads