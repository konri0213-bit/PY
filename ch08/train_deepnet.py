# coding: utf-8
# 导入标准库 sys 和 os，用于系统路径设置等操作
import sys, os
# 将当前目录的父目录添加到 sys.path 中，以便可以从父目录导入自定义模块（如 dataset、deep_convnet 等）
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
# 导入 numpy，用于高效的数值计算
import numpy as np
# 导入 matplotlib.pyplot，用于绘图和可视化
import matplotlib.pyplot as plt
# 从 dataset.mnist 模块导入 load_mnist 函数，用于加载 MNIST 数据集
from dataset.mnist import load_mnist
# 从 deep_convnet 模块导入 DeepConvNet 类，这是一个定义好的深度卷积神经网络模型
from deep_convnet import DeepConvNet
# 从 common.trainer 模块导入 Trainer 类，用于封装网络的训练过程（包括小批量训练、评估等）
from common.trainer import Trainer

# 加载 MNIST 数据集，设置 flatten=False 表示保留图像的二维形状（适合卷积网络输入）
# (x_train, t_train) 为训练图像和训练标签，(x_test, t_test) 为测试图像和测试标签
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 实例化深度卷积网络对象，准备开始训练
network = DeepConvNet()  
# 创建训练器，传入网络模型、训练数据、测试数据以及各项训练超参数
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20,  # 训练的总轮数（遍历所有训练数据的次数）
                  mini_batch_size=100,  # 每次迭代使用的小批量样本数量
                  optimizer='Adam',  # 优化器选择 Adam，自适应学习率优化算法
                  optimizer_param={'lr':0.001},  # 优化器的参数，这里设置学习率为 0.001
                  evaluate_sample_num_per_epoch=1000)  # 每个 epoch 结束后，用于评估性能的验证集样本数量
# 启动训练过程
trainer.train()

# 训练完成后，将学习到的网络参数保存到文件中，以便后续直接加载使用
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")