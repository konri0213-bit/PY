# coding: utf-8
# 导入系统与路径相关模块，用于将父目录加入搜索路径
import sys, os
# 将父目录添加到系统路径，使得后续能正确导入父目录下的包（如 deep_convnet, dataset）
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
# 导入自定义的深度卷积网络类
from deep_convnet import DeepConvNet
# 导入 MNIST 数据集加载函数
from dataset.mnist import load_mnist

# 加载 MNIST 数据集，设置 flatten=False 以保留图像的二维形状 (1, 28, 28)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 实例化深度卷积网络
network = DeepConvNet()
# 载入预训练好的参数
network.load_params("deep_convnet_params.pkl")

# 为加快测试速度，只选取前 10000 个样本进行评估
sampled = 10000  # 取样数量，用于实现高速化
x_test = x_test[:sampled]
t_test = t_test[:sampled]

# 使用默认的 float64 数据类型计算并输出测试精度
print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

# 将测试数据转换为 float16 类型，降低数值精度以测试模型在低精度下的表现
x_test = x_test.astype(np.float16)
# 遍历网络的所有参数，将其全部转换为 float16 类型
for param in network.params.values():
    # 使用 param[...] 保持原对象引用，原地修改数据类型
    param[...] = param.astype(np.float16)

# 使用 float16 数据类型再次计算并输出测试精度，对比高精度与低精度的差异
print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))