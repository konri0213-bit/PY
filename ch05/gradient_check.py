# coding: utf-8
# 梯度检查脚本：通过比较数值微分与误差反向传播法得到的梯度，验证两层神经网络的实现正确性
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist  # 加载MNIST数据集
from two_layer_net import TwoLayerNet  # 导入自定义的两层神经网络类

# 读入数据
# normalize=True 将图像像素值归一化到0~1之间，one_hot_label=True 将标签转换为one-hot编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化两层神经网络：输入层大小784（28x28图像），隐藏层50个神经元，输出层10个数字类别
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 选取训练数据的前3个样本构成小批量，用于梯度检验
x_batch = x_train[:3]
t_batch = t_train[:3]

# 通过数值微分计算梯度，作为“正确”的梯度参考（耗时较长但实现简单可靠）
grad_numerical = network.numerical_gradient(x_batch, t_batch)
# 通过误差反向传播法计算梯度，速度快，是实际训练中使用的方式
grad_backprop = network.gradient(x_batch, t_batch)

# 遍历网络所有参数的梯度，比较两种方法得到的梯度值之间的差异
for key in grad_numerical.keys():
    # 计算反向传播梯度与数值微分梯度之间各元素的绝对差的平均值
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    # 输出当前参数的名称及平均差异值；差异极小（如1e-8量级）表明反向传播实现正确
    print(key + ":" + str(diff))