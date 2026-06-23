# coding: utf-8
import sys, os
# 为了导入父目录的文件而进行的设定，将父目录追加到系统路径中
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist  # 导入 MNIST 数据集加载函数
from common.multi_layer_net_extend import MultiLayerNetExtend  # 导入带有 BatchNorm 等扩展功能的多层网络类

# 读入数据
# 参数含义：normalize=True 将像素值归一化至0-1，one_hot_label=True 将标签转换为 one-hot 向量
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 构建网络：输入层784个神经元，两个隐层各100个神经元，输出层10个类别，启用Batch Normalization
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10,
                              use_batchnorm=True)

# 取出训练数据的第一个样本作为小批量，用于梯度检查
x_batch = x_train[:1]
t_batch = t_train[:1]

# 利用反向传播计算网络各参数的梯度
grad_backprop = network.gradient(x_batch, t_batch)
# 利用数值微分计算网络各参数的梯度（用于验证反向传播梯度是否正确）
grad_numerical = network.numerical_gradient(x_batch, t_batch)

# 遍历所有参数键，比较反向传播梯度与数值微分梯度之间的误差
for key in grad_numerical.keys():
    # 计算该参数梯度的平均绝对误差（MAE）
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))  # 输出参数名及其对应的梯度误差