# coding: utf-8
import sys, os
# 将父目录添加到系统模块搜索路径中，以便能够顺利导入位于父目录中的自定义模块
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
# 从 dataset.mnist 模块导入 load_mnist 函数，用于加载 MNIST 手写数字数据集
from dataset.mnist import load_mnist
# 导入简单卷积神经网络模型定义
from simple_convnet import SimpleConvNet
# 导入训练器类，用于管理整个训练循环（包含前向传播、反向传播、优化器更新和评估）
from common.trainer import Trainer

# ---------------------------------------------
# 数据准备
# ---------------------------------------------
# 读入数据：返回训练数据和测试数据，flatten=False 保持图像原始形状 (1, 28, 28) 以适配卷积层
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 处理花费时间较长的情况下减少数据 
# 如果训练时间过长，可以取消以下两行注释以仅使用部分数据快速实验
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

# 设定最大训练轮数（遍历整个训练集的次数）
max_epochs = 20

# ---------------------------------------------
# 构建卷积神经网络
# ---------------------------------------------
# 实例化一个简单的卷积神经网络
# input_dim: 输入数据的维度 (通道数, 高, 宽)，符合 MNIST 单通道灰度图
# conv_param: 卷积层参数字典，指定滤波器数量、尺寸、填充和步幅
# hidden_size: 全连接隐藏层的神经元数量
# output_size: 输出层的神经元数量，对应于 10 个数字类别
# weight_init_std: 权重初始化的标准差，用于控制初始权重的分布范围
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# ---------------------------------------------
# 设置训练器并执行训练
# ---------------------------------------------
# 创建训练器对象，封装了网络、数据、超参数以及评估策略
# mini_batch_size: 每次迭代使用的批量样本数
# optimizer: 优化器选择 'Adam'
# optimizer_param: 优化器的参数字典，这里设置学习率 lr 为 0.001
# evaluate_sample_num_per_epoch: 每个 epoch 用于评估准确率时抽取的样本数量，加快评估速度
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
# 启动训练过程，Trainer 内部会执行前向传播、反向传播、参数更新，并记录每个 epoch 的损失与准确率
trainer.train()

# ---------------------------------------------
# 保存训练好的模型参数
# ---------------------------------------------
# 将网络各层学习到的权重和偏置参数持久化到 "params.pkl" 文件中，方便后续加载与推理
network.save_params("params.pkl")
print("Saved Network Parameters!")

# ---------------------------------------------
# 绘制训练过程中的准确率变化曲线
# ---------------------------------------------
# 定义训练和测试曲线的标记样式，o 表示圆形，s 表示方形
markers = {'train': 'o', 'test': 's'}
# 生成横坐标值，表示 epoch 序号（0 到 max_epochs-1）
x = np.arange(max_epochs)
# 绘制训练准确率曲线，markevery=2 表示每隔两个点绘制一个标记，避免图形过于拥挤
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
# 绘制测试准确率曲线
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
# 添加坐标轴标签和图例
plt.xlabel("epochs")
plt.ylabel("accuracy")
# 设置纵轴范围为 0 到 1.0，因为准确率取值在 0%~100% 之间
plt.ylim(0, 1.0)
# 将图例放置在右下角，避免遮挡曲线
plt.legend(loc='lower right')
# 显示绘制的图形
plt.show()