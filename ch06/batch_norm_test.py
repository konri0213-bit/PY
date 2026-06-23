# coding: utf-8
# 导入系统与路径相关模块
import sys, os
# 将父目录加入搜索路径，以便导入 common 与 dataset 中的模块
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
# 从 dataset 子包中导入 MNIST 数据加载函数
from dataset.mnist import load_mnist
# 导入扩展版多层网络（支持批标准化等）
from common.multi_layer_net_extend import MultiLayerNetExtend
# 导入优化器类，本脚本主要使用 SGD，Adam 虽导入但未使用
from common.optimizer import SGD, Adam

# 加载 MNIST 数据集，设定 normalize=True 将像素值归一化到 [0,1]
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 减少学习数据，仅使用前 1000 条样本以加快实验速度
x_train = x_train[:1000]
t_train = t_train[:1000]

# 训练超参数设置
max_epochs = 20                       # 总训练轮数
train_size = x_train.shape[0]         # 当前训练样本总数
batch_size = 100                      # 小批量大小
learning_rate = 0.01                  # 学习率


def __train(weight_init_std):
    """
    使用指定权重初始化标准差训练两个网络并记录精度变化。
    
    参数 weight_init_std : float
        网络中权重初始化的标准差（高斯分布的标准差）。
        该参数用于对比不同初始化下，批标准化（Batch Normalization）带来的效果差异。
    
    返回值:
        train_acc_list : list
            普通网络（无批标准化）每个 epoch 在训练集上的识别精度列表。
        bn_train_acc_list : list
            带批标准化网络每个 epoch 在训练集上的识别精度列表。
    """
    # 构建带批标准化的多层网络，隐藏层均为 100 个神经元，共 5 层，输出 10 类
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    # 构建不带批标准化的普通多层网络，结构相同，仅 use_batchnorm 默认为 False
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    # 使用随机梯度下降优化器，学习率已由全局变量设置
    optimizer = SGD(lr=learning_rate)
    
    # 用于记录每个 epoch 精度的列表
    train_acc_list = []
    bn_train_acc_list = []
    
    # 每个 epoch 内需要执行的迭代次数，至少为 1 次
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0  # 当前已完成的 epoch 计数
    
    # 无限循环，通过内部判断 epoch_cnt 达到 max_epochs 后跳出
    for i in range(1000000000):
        # 随机抽取小批量索引，生成当前批次的输入和标签
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        # 两个网络分别使用同一个批次计算梯度并更新参数
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)   # 误差反向传播求梯度
            optimizer.update(_network.params, grads)      # 使用 SGD 更新权重与偏置
    
        # 每完成一个 epoch 对应的迭代次数时，计算并记录当前精度
        if i % iter_per_epoch == 0:
            # 计算普通网络在当前训练集上的准确率
            train_acc = network.accuracy(x_train, t_train)
            # 计算带批标准化网络在当前训练集上的准确率
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            # 打印当前 epoch 信息
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            # 达到预设的最大 epoch 数后终止训练
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 3.绘制图形==========
# 生成 16 个对数均匀分布的权重初始化标准差，从 10^0 到 10^{-4}
weight_scale_list = np.logspace(0, -4, num=16)
# x 轴为 epoch 编号，用于绘图
x = np.arange(max_epochs)

# 对每个初始化标准差，训练并绘制子图
for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    # 调用训练函数，获取两种网络在各 epoch 的精度
    train_acc_list, bn_train_acc_list = __train(w)
    
    # 在 4 行 4 列的子图中定位
    plt.subplot(4,4,i+1)
    # 设置子图标题，显示当前使用的权重标准差
    plt.title("W:" + str(w))
    # 仅在最后一个子图（i==15）显示图例标签，避免其他子图重复显示图例
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    # 设置纵坐标范围为 [0, 1]，对应准确率区间
    plt.ylim(0, 1.0)
    # 为便于观察，仅最左侧一列显示 y 轴刻度标签，其余列隐藏
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    # 仅最后一行显示 x 轴标签，其余行隐藏
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    # 将图例放置在右下角
    plt.legend(loc='lower right')
    
# 显示所有子图
plt.show()