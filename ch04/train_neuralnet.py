# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 实例化两层神经网络：输入大小784（28×28图像展平），隐藏层50个神经元，输出层10个类别（数字0-9）
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# ---------- 超参数设定 ----------
iters_num = 10000  # 适当设定循环的次数（随机梯度下降的总迭代步数）
train_size = x_train.shape[0]  # 训练样本总数
batch_size = 100  # 每次迭代使用的小批量大小
learning_rate = 0.1  # 参数更新的学习率

# 用于记录训练过程中的指标变化
train_loss_list = []  # 每个迭代步的损失值
train_acc_list = []   # 每个epoch结束时的训练精度
test_acc_list = []    # 每个epoch结束时的测试精度

# 计算一个epoch约等于多少次迭代（至少为1），用于判断何时评估精度
iter_per_epoch = max(train_size / batch_size, 1)

# ---------- 训练循环 ----------
for i in range(iters_num):
    # 从训练集中随机抽取batch_size个样本的索引，用作本次小批量
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]   # 小批量的输入数据
    t_batch = t_train[batch_mask]   # 小批量的监督标签（one-hot向量）

    # 计算梯度（使用误差反向传播法，实现高速计算）
    # grad = network.numerical_gradient(x_batch, t_batch)  # 数值微分版本（较慢，已注释）
    grad = network.gradient(x_batch, t_batch)

    # 更新参数（对所有层的权重和偏置执行SGD更新）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算本次小批量上的损失值，并记录到列表中
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每一个epoch结束时，计算全部训练数据和测试数据的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# ---------- 绘制精度变化曲线 ----------
markers = {'train': 'o', 'test': 's'}  # 预设的绘图标记样式（train用圆点，test用方块；此处未在plot中显式使用）
x = np.arange(len(train_acc_list))  # 横轴：epoch数（与精度记录数组等长）
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")                # 横轴标签：epoch数
plt.ylabel("accuracy")             # 纵轴标签：识别精度
plt.ylim(0, 1.0)                    # 纵轴范围设为0~1，与精度取值一致
plt.legend(loc='lower right')      # 在右下角显示图例
plt.show()