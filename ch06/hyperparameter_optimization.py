# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# 加载 MNIST 数据集，并将像素值归一化到 [0,1] 区间
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了实现高速化，减少训练数据
x_train = x_train[:500]
t_train = t_train[:500]

# 分割验证数据
validation_rate = 0.20  # 验证集占据整体训练数据的比例
validation_num = int(x_train.shape[0] * validation_rate)  # 验证集样本数
x_train, t_train = shuffle_dataset(x_train, t_train)  # 打乱训练数据，避免原始顺序带来的偏差
x_val = x_train[:validation_num]  # 切分验证输入数据
t_val = t_train[:validation_num]  # 切分验证标签数据
x_train = x_train[validation_num:]  # 剩余部分作为新的训练输入数据
t_train = t_train[validation_num:]  # 剩余部分作为新的训练标签数据


def __train(lr, weight_decay, epocs=50):
    """使用给定的学习率和权值衰减强度进行模型训练

    Args:
        lr: 学习率 (learning rate)
        weight_decay: 权值衰减强度（L2 正则化系数 lambda）
        epocs: 训练的轮数，默认 50

    Returns:
        trainer.test_acc_list: 每个 epoch 结束后的验证准确率列表
        trainer.train_acc_list: 每个 epoch 结束后的训练准确率列表
    """
    # 构建多层神经网络，输入层 784 个神经元（28x28 图像展平），
    # 隐藏层均为 100 个神经元共 6 层，输出层 10 个神经元（对应数字 0-9）
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    # 初始化 Trainer，负责执行训练循环
    # 使用 SGD 优化器，mini_batch_size=100，不输出每个 epoch 的详细日志
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    # 返回训练过程中记录的验证准确率与训练准确率列表
    return trainer.test_acc_list, trainer.train_acc_list


# 超参数的随机搜索======================================
optimization_trial = 100  # 随机搜索的试验次数
results_val = {}  # 保存每次试验的验证准确率变化曲线，键为超参数字符串
results_train = {}  # 保存每次试验的训练准确率变化曲线
for _ in range(optimization_trial):
    # 指定搜索的超参数的范围===============
    # 权值衰减系数在对数空间中均匀随机采样：10^{-8} ~ 10^{-4}
    weight_decay = 10 ** np.random.uniform(-8, -4)
    # 学习率在对数空间中均匀随机采样：10^{-6} ~ 10^{-2}
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    # 执行一次训练，并获取验证与训练准确率的历史记录
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    # 输出当前试验的最终验证准确率和对应的超参数
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    # 将超参数组合字符串作为键，方便后续排序与展示
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 绘制图形========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20  # 绘制前 20 个最佳试验的结果曲线
col_num = 5  # 子图列数
row_num = int(np.ceil(graph_draw_num / col_num))  # 根据列数计算所需行数
i = 0

# 按最终验证准确率降序排列，显示效果最好的超参数组合
for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    # 打印排名、验证准确率及对应的超参数组合
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    # 在子图中绘制该超参数组合的训练曲线
    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))  # 子图标题表示排名
    plt.ylim(0.0, 1.0)  # 准确率纵轴范围为 0~1
    if i % 5: plt.yticks([])  # 非第一列的子图不显示 y 轴刻度，保持图像简洁
    plt.xticks([])  # 所有子图不显示 x 轴刻度
    x = np.arange(len(val_acc_list))  # x 轴为 epoch 索引
    plt.plot(x, val_acc_list)  # 实线绘制验证准确率变化
    plt.plot(x, results_train[key], "--")  # 虚线绘制训练准确率变化
    i += 1

    # 当剩余试验不足 20 个时，停止绘图
    if i >= graph_draw_num:
        break

plt.show()