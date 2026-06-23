# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

# 加载经过正规化（normalize）处理后的 MNIST 数据集，返回训练数据与测试数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了再现过拟合，故意减少训练数据的数量，仅使用前300个样本
x_train = x_train[:300]
t_train = t_train[:300]

# 设定是否使用 Dropout 以及 Dropout 的比例 ========================
# use_dropout 为 True 表示使用 Dropout 方法进行正则化
use_dropout = True  # 不使用Dropout的情况下为False
# dropout_ratio 指定每个 Dropout 层随机丢弃神经元的比例
dropout_ratio = 0.2
# ====================================================

# 构建扩展的多层神经网络
# 输入层大小为 784（28x28 像素），隐藏层共有6层，每层100个神经元，输出层10个神经元（对应0~9数字）
# 根据 use_dropout 和 dropout_ration 参数决定是否在隐藏层后插入 Dropout 层
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
# 创建训练器，设定训练参数：训练周期数、小批量大小、优化器类型及学习率等
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
# 开始执行训练过程
trainer.train()

# 从训练器中获取训练过程和测试过程中每个 epoch 的识别精度列表
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 绘制训练集和测试集准确率变化的图形==========
# 定义不同数据集的绘图标记样式
markers = {'train': 'o', 'test': 's'}
# 生成 epoch 的横坐标数值
x = np.arange(len(train_acc_list))
# 绘制训练准确率曲线，以圆圈标记，每10个 epoch 显示一个标记点
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
# 绘制测试准确率曲线，以方块标记，每10个 epoch 显示一个标记点
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")                      # 横坐标标签
plt.ylabel("accuracy")                    # 纵坐标标签
plt.ylim(0, 1.0)                         # 设置纵轴范围，准确率在0到1之间
plt.legend(loc='lower right')            # 在右下角显示图例
plt.show()                               # 显示绘制的图形