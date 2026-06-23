# coding: utf-8
import sys, os
# 将父目录加入系统路径，以便导入父目录中的模块（如 common 包）
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    """识别率为99%以上的高精度的ConvNet

    网络结构如下所示
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        # ==================== 初始化权重 ====================
        # pre_node_nums：各权重层输入侧每个神经元的连接数（即前一层的通道数*感受野大小）
        # 用于 He 初始化时计算缩放因子，保证各层输出的方差稳定
        # 各元素依次对应：卷积层1~6、全连接层7、全连接层8的输入连接数
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        # 使用 ReLU 时推荐的 He 初始值：标准差为 sqrt(2/n) ，n 为前一层神经元的平均连接数
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)
        
        self.params = {}
        # pre_channel_num 记录当前层的输入通道数，初始为输入图像的通道数
        pre_channel_num = input_dim[0]
        # 循环创建6个卷积层的权重和偏置
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            # 权重形状：(filter_num, 输入通道数, filter_size, filter_size)
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            # 偏置初始化为0
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            # 更新下一层的输入通道数为当前层的滤波器个数
            pre_channel_num = conv_param['filter_num']
        # 全连接层1（第7层）：输入为最后一个池化层的展开尺寸 (64*4*4)，输出为 hidden_size
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        # 全连接层2（第8层）：输入为 hidden_size，输出为 output_size（分类数）
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # ==================== 构建网络层（有序列表） ====================
        self.layers = []
        # 第1个卷积块：卷积1 + ReLU
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        # 第1个卷积块：卷积2 + ReLU + 池化
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 第2个卷积块：卷积3 + ReLU
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        # 第2个卷积块：卷积4 + ReLU + 池化（注意 conv_param_4 的填充为2，输出尺寸不变可能会变大）
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 第3个卷积块：卷积5 + ReLU
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        # 第3个卷积块：卷积6 + ReLU + 池化
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 全连接部分：第7层仿射 + ReLU + Dropout(训练时丢弃50%)
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        # 第8层仿射 + Dropout(训练时丢弃50%)，最后配合 SoftmaxWithLoss 输出类别概率
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        
        # 最终的 softmax 与交叉熵损失层，不在 layers 列表中单独管理，用于计算损失和反向传播的起点
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        """前向传播：逐层计算，Dropout 层需要传入训练标志以决定是否随机丢弃神经元"""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                # Dropout 层的 forward 需要 train_flg 参数，训练时随机失活，推理时不失活
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算损失：先进行完整的前向传播（训练模式），再通过 SoftmaxWithLoss 层计算交叉熵损失"""
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算模型在给定数据上的准确率，支持分批处理以适应内存或 GPU 限制"""
        # 如果标签为 one-hot 编码，则转换为类别索引
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        # 按 batch 分批预测，累加正确个数
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)  # 推理模式
            y = np.argmax(y, axis=1)  # 取最大概率对应的类别
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """计算网络中各带参数层的梯度，返回与 self.params 对应的梯度字典"""
        # 1. 前向传播，同时计算损失（内部会调用 loss 方法完成一次完整前向）
        self.loss(x, t)

        # 2. 反向传播，从最后一层 softmax 开始
        dout = 1
        dout = self.last_layer.backward(dout)

        # 将网络层顺序反转，从输出层向输入层传播梯度
        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 3. 提取各带参数层的梯度（参数层在 self.layers 中的索引事先固定）
        grads = {}
        # (0, 2, 5, 7, 10, 12, 15, 18) 分别对应：
        # Conv1, Conv2, Conv3, Conv4, Conv5, Conv6, Affine1, Affine2
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        """将当前模型参数（权重、偏置）保存到 pickle 文件"""
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """从 pickle 文件加载模型参数，并同步更新到对应的层中"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        # 更新参数字典
        for key, val in params.items():
            self.params[key] = val

        # 将参数重新赋值给相应的层对象，以保证网络层内部使用的参数与 self.params 一致
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]