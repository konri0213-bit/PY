# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """扩展版的全连接的多层神经网络
    
    具有Weiht Decay、Dropout、Batch Normalization的功能

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    use_dropout: 是否使用Dropout
    dropout_ration : Dropout的比例
    use_batchNorm: 是否使用Batch Normalization
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        # 保存网络结构参数
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)          # 隐藏层的层数
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda        # L2正则化强度
        self.use_batchnorm = use_batchnorm
        self.params = {}                                      # 参数字典

        # 初始化权重（根据激活函数选用He或Xavier初始值）
        self.__init_weight(weight_init_std)

        # 生成层（按照顺序构成网络）
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu} # 激活函数字典
        self.layers = OrderedDict()                           # 用有序字典按顺序保存层
        for idx in range(1, self.hidden_layer_num+1):
            # 添加仿射层（全连接层）
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 若使用Batch Normalization，则初始化缩放参数gamma和偏移参数beta，并添加BN层
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])   # 初始化为1
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])   # 初始化为0
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            # 添加激活函数层
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            # 若使用Dropout，则在激活函数之后添加Dropout层
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        # 最后的输出层（仿射层，不添加激活函数和BN、Dropout）
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        # 最后的损失函数层：SoftmaxWithLoss
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        # 所有层的尺寸列表，从输入层到输出层
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # 根据激活函数类型选择相应的初始化缩放因子
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            # 初始化权重矩阵（正态分布）和偏置（全零）
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        """前向传播，根据train_flg控制Dropout和BatchNorm的行为"""
        for key, layer in self.layers.items():
            # Dropout和BatchNorm层在训练和测试时行为不同，需要传入train_flg标志
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """求损失函数
        参数x是输入数据，t是教师标签
        """
        # 前向传播得到预测值
        y = self.predict(x, train_flg)

        # 计算L2正则化项（权重衰减）的损失值：0.5 * λ * Σ(W^2)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):  # 遍历所有层的权重（包括输出层）
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        # 总损失 = softmax交叉熵损失 + 正则化项
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        """计算分类准确率"""
        # 预测（测试模式，不使用Dropout）
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)          # 获取预测类别
        if T.ndim != 1 : T = np.argmax(T, axis=1)  # 若教师标签为one-hot，则转换为类别索引

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """求梯度（数值微分）

        Parameters
        ----------
        X : 输入数据
        T : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # 定义以损失函数为目标的回调函数，训练模式下计算损失
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        # 对所有层的权重和偏置进行数值微分
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            # 如果使用了Batch Normalization且不是输出层（输出层没有BN），则计算gamma和beta的梯度
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        """误差反向传播法求梯度（同时包含了Weight Decay的导数）"""
        # forward（训练模式，启用Dropout和BatchNorm的训练行为）
        self.loss(x, t, train_flg=True)

        # backward：从最后一层（SoftmaxWithLoss）开始反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        # 将层列表反转，按反向顺序处理
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 收集各参数的梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # 权重梯度除了反向传播得到的dW，还需加上L2正则化项的导数 λ * W
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # 如果使用了Batch Normalization且不是输出层，则保存gamma和beta的梯度
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads