# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """简单的ConvNet

    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        # 从卷积参数字典中提取各项配置
        filter_num = conv_param['filter_num']         # 卷积核个数（输出通道数）
        filter_size = conv_param['filter_size']       # 卷积核的空间尺寸（正方形，边长为filter_size）
        filter_pad = conv_param['pad']                # 填充宽度
        filter_stride = conv_param['stride']          # 卷积步长
        input_size = input_dim[1]                     # 输入图像的空间尺寸（假设高宽相等，例如MNIST为28）
        # 卷积层输出的空间尺寸计算公式：(输入尺寸 - 卷积核尺寸 + 2*填充) / 步长 + 1
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 池化层（2x2，步长2）进一步将空间尺寸减半，最终展平为一维向量的长度
        # 计算公式：卷积核个数 * (池化后高度) * (池化后宽度)
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 初始化权重
        self.params = {}
        # 第1层权重：卷积核，形状为 (filter_num, 输入通道数, filter_size, filter_size)
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)  # 卷积层的偏置，每个卷积核对应一个常数偏置

        # 第2层权重：全连接层，将池化后的特征图展平后映射到隐藏层节点
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        # 第3层权重：输出层，将隐藏层映射到最终分类数
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        # 卷积层
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()                                 # 激活函数ReLU
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2) # 2×2最大值池化，步长为2，尺寸减半

        # 第一个全连接层（Affine），由池化输出映射到隐藏层
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()                                 # 再次使用ReLU激活
        # 第二个全连接层，由隐藏层映射到输出层（未加激活，直接送入SoftmaxWithLoss）
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # 最后输出的Softmax和交叉熵损失合并层，用于计算损失和反向传播的起始梯度
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 按顺序依次通过各层前向传播，得到最终输出得分（未经过softmax的logits）
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        # 前向传播获取预测输出
        y = self.predict(x)
        # 由SoftmaxWithLoss层计算交叉熵损失
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        # 如果教师标签是one-hot形式（二维），转换为类别索引形式（一维）
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        # 按批次计算准确率，避免一次性处理过多数据
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]  # 提取当前批次的输入
            tt = t[i*batch_size:(i+1)*batch_size]  # 提取当前批次的标签
            y = self.predict(tx)                    # 得到预测得分
            y = np.argmax(y, axis=1)                # 得分最大的类别作为预测结果
            acc += np.sum(y == tt)                  # 统计预测正确的样本数
        
        # 返回整体准确率
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

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
        # 定义以当前参数为变量的损失函数（闭包），便于数值微分逐个计算
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        # 对三层权重和偏置分别做数值微分，计算损失对每个参数的梯度
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

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
        # forward: 执行一次前向传播并计算损失，同时保存各层计算图所需的中间变量
        self.loss(x, t)

        # backward: 从损失层开始反向传播梯度
        dout = 1  # 损失关于自身的梯度初始值为1
        dout = self.last_layer.backward(dout)  # 通过SoftmaxWithLoss层反向传播

        # 按相反顺序遍历所有隐含层进行反向传播
        layers = list(self.layers.values())
        layers.reverse()  # 逆序，从最后一层（Affine2）往前传播至第一层（Conv1）
        for layer in layers:
            dout = layer.backward(dout)

        # 设定：从各层对象中取出计算好的权重梯度和偏置梯度
        grads = {}
        # 卷积层的权重梯度与偏置梯度（在反向传播时已累积在层的dW,db属性中）
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        # 第一个全连接层
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # 第二个全连接层
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        """将模型参数保存到Pickle文件中"""
        # 将参数字典复制一份（防止外部引用修改内部状态）
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """从Pickle文件中加载模型参数，并同步更新到各层"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        # 恢复参数字典
        for key, val in params.items():
            self.params[key] = val

        # 将加载的参数显式写入对应层中，以保证层对象与参数字典的一致性
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]