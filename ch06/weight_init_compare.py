# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np  # 导入数值计算库，用于矩阵运算和随机数生成
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化训练损失曲线
from dataset.mnist import load_mnist  # 从 dataset 包中导入 MNIST 数据加载函数
from common.util import smooth_curve  # 导入曲线平滑工具，使损失曲线更美观
from common.multi_layer_net import MultiLayerNet  # 导入多层神经网络类
from common.optimizer import SGD  # 导入随机梯度下降优化器


# 0:读入MNIST数据==========
# 加载 MNIST 数据集，设置 normalize=True 表示将像素值归一化到 [0,1] 区间
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]  # 训练样本总数
batch_size = 128               # 每次迭代使用的批量大小
max_iterations = 2000          # 训练的总迭代次数


# 1:进行实验的设置==========
# 定义三种不同的权重初始化方式及其参数：
# 'std=0.01'：用标准差 0.01 的高斯分布直接初始化，不进行针对性缩放
# 'Xavier'：Xavier 初始化，适合激活函数为 sigmoid 的场景，用字符串 'sigmoid' 表示
# 'He'：He 初始化，适合 ReLU 激活函数，用 'relu' 表示
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
# 使用学习率为 0.01 的 SGD 优化器
optimizer = SGD(lr=0.01)

# 保存不同初始化方式对应的网络对象
networks = {}
# 保存不同初始化方式在每个迭代记录的损失值
train_loss = {}
for key, weight_type in weight_init_types.items():
    # 创建具有 4 个隐藏层（每层 100 个神经元）的多层网络
    # input_size=784 (28x28 图像展平), output_size=10 (数字类别)
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []  # 每个初始化方式对应一个空列表，用于记录损失


# 2:开始训练==========
for i in range(max_iterations):
    # 从训练集中随机抽取 batch_size 个样本构成当前批次
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 对每一种权重初始化方式，分别计算梯度并更新参数
    for key in weight_init_types.keys():
        # 计算当前批次上的梯度
        grads = networks[key].gradient(x_batch, t_batch)
        # 使用 SGD 优化器更新网络参数
        optimizer.update(networks[key].params, grads)
    
        # 计算更新后的网络在当前批次上的损失
        loss = networks[key].loss(x_batch, t_batch)
        # 将损失值记录下来，用于后续绘图
        train_loss[key].append(loss)
    
    # 每 100 次迭代打印一次当前损失，便于观察训练进展
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.绘制图形==========
# 不同初始化方式的损失曲线使用不同标记，便于区分
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)  # 横轴为迭代次数
for key in weight_init_types.keys():
    # 绘制平滑后的损失曲线，每隔 100 次迭代显示一次标记点
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")  # 横轴标签
plt.ylabel("loss")        # 纵轴标签
plt.ylim(0, 2.5)          # 固定纵轴范围，便于比较不同初始化下的损失尺度
plt.legend()              # 显示图例
plt.show()                # 展示绘制的图形