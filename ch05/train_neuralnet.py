# coding: utf-8
import sys, os
# 将父目录添加到系统路径，以便导入自定义模块
sys.path.append(os.pardir)

import numpy as np
# 从 dataset 包中导入用于加载 MNIST 数据集的函数
from dataset.mnist import load_mnist
# 导入自定义的两层神经网络类
from two_layer_net import TwoLayerNet

# 读入数据
# 加载 MNIST 数据，进行归一化处理并将标签转换为 one-hot 编码
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 实例化两层神经网络，指定输入层大小（28*28=784）、隐藏层神经元数量、输出层大小（10个类别）
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设定训练的超参数
iters_num = 10000               # 总迭代（梯度下降）次数
train_size = x_train.shape[0]  # 训练样本总数
batch_size = 100               # 每次迭代使用的批量大小
learning_rate = 0.1            # 学习率，控制参数更新步长

# 用于记录训练过程中的指标
train_loss_list = []  # 存储每次迭代的训练损失
train_acc_list = []   # 存储每个 epoch 时的训练集准确率
test_acc_list = []    # 存储每个 epoch 时的测试集准确率

# 计算一个 epoch 对应的迭代次数（至少为1，避免除零错误）
# 一个 epoch 定义为遍历整个训练集一次
iter_per_epoch = max(train_size / batch_size, 1)

# 开始迭代训练
for i in range(iters_num):
    # 从训练集中随机选取 batch_size 个样本的索引
    batch_mask = np.random.choice(train_size, batch_size)
    # 根据索引获取输入批量数据
    x_batch = x_train[batch_mask]
    # 根据索引获取监督标签批量数据
    t_batch = t_train[batch_mask]
    
    # 梯度
    # 使用数值微分法计算梯度（已注释，改用误差反向传播法）
    #grad = network.numerical_gradient(x_batch, t_batch)
    # 使用误差反向传播法高效计算当前批量的梯度
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    # 遍历网络的所有可训练参数（权重和偏置），根据计算的梯度进行梯度下降更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 计算当前批量的损失值，并记录到列表中用于后续观察
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 每经过一个 epoch（即完整遍历一次训练集）时，评估模型性能
    if i % iter_per_epoch == 0:
        # 计算当前模型在整个训练集上的准确率
        train_acc = network.accuracy(x_train, t_train)
        # 计算当前模型在测试集上的准确率，用于观察泛化能力
        test_acc = network.accuracy(x_test, t_test)
        # 记录准确率
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 打印当前 epoch 的训练准确率和测试准确率
        print(train_acc, test_acc)