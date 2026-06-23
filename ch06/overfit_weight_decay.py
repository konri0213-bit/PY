# coding: utf-8
# 导入必要的系统模块，用于路径操作
import os
import sys

# 将父目录加入到系统路径中，以便导入父目录下的模块（common, dataset等）
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
# 从dataset.mnist导入MNIST数据加载函数
from dataset.mnist import load_mnist
# 导入我们自定义的多层神经网络类
from common.multi_layer_net import MultiLayerNet
# 导入SGD优化器
from common.optimizer import SGD

# 加载MNIST数据集，设置normalize=True将输入图像像素值归一化到0.0~1.0
# 返回训练数据和测试数据，每个数据都包含输入图像(x)和对应的标签(t)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了人为制造过拟合，只选取前300个训练样本
# 这样模型会在少量数据上过度学习，便于观察权值衰减的效果
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（权值衰减）的设定 =======================
# 权值衰减正则化系数 λ，用于抑制过拟合。
# weight_decay_lambda = 0 # 不使用权值衰减的情况（注释掉）
weight_decay_lambda = 0.1
# ====================================================

# 创建多层神经网络实例
# 输入层大小784（MNIST图像28×28）
# 隐藏层为6个全连接层，每层100个神经元
# 输出层大小10（对应0-9数字类别）
# weight_decay_lambda 传递权值衰减系数，用于计算损失时加入L2正则项
network = MultiLayerNet(input_size=784,
                        hidden_size_list=[100, 100, 100, 100, 100, 100],
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
# 初始化随机梯度下降优化器，学习率为0.01
optimizer = SGD(lr=0.01)

# 训练总轮数（epoch数）
max_epochs = 201
# 当前训练集样本数量
train_size = x_train.shape[0]
# 每个批次的大小（mini-batch size）
batch_size = 100

# 用于记录每个epoch的训练损失（此处仅初始化，后续未实际记录）
train_loss_list = []
# 记录每个epoch的训练准确率
train_acc_list = []
# 记录每个epoch的测试准确率
test_acc_list = []

# 计算每个epoch对应的迭代次数，至少为1
iter_per_epoch = max(train_size / batch_size, 1)
# 当前已完成的epoch计数
epoch_cnt = 0

# 主训练循环，设置一个极大的上限（实际由max_epochs控制退出）
for i in range(1000000000):
    # 从训练集中随机抽取batch_size个样本的索引，构成一个mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    # 取出对应的输入数据
    x_batch = x_train[batch_mask]
    # 取出对应的监督标签
    t_batch = t_train[batch_mask]

    # 计算当前批次数据的梯度（包含权值衰减产生的正则化梯度部分）
    grads = network.gradient(x_batch, t_batch)
    # 使用优化器更新网络参数
    optimizer.update(network.params, grads)

    # 判断是否已经完整遍历一个epoch（即迭代次数达到iter_per_epoch的整数倍）
    if i % iter_per_epoch == 0:
        # 计算当前模型在整个训练集上的准确率
        train_acc = network.accuracy(x_train, t_train)
        # 计算当前模型在测试集上的准确率
        test_acc = network.accuracy(x_test, t_test)
        # 保存当前epoch的准确率数据，便于后续绘图
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 打印当前epoch的训练和测试准确率
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        # 完成一个epoch，计数器加1
        epoch_cnt += 1
        # 如果达到预设的最大epoch数，则终止训练
        if epoch_cnt >= max_epochs:
            break

# 3. 绘制训练和测试准确率曲线 ==========
# 定义不同数据的标记样式
markers = {'train': 'o', 'test': 's'}
# 生成epoch的横坐标序列，从0到max_epochs-1
x = np.arange(max_epochs)
# 绘制训练准确率曲线，使用圆形标记，每10个epoch显示一个标记点以防过密
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
# 绘制测试准确率曲线，使用方形标记，同样每10个epoch显示一个点
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
# 设置X轴标签
plt.xlabel("epochs")
# 设置Y轴标签
plt.ylabel("accuracy")
# 设置Y轴范围为0到1.0，因为准确率取值在0~1之间
plt.ylim(0, 1.0)
# 在右下角显示图例
plt.legend(loc='lower right')
# 显示图形
plt.show()