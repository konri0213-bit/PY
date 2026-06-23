# coding: utf-8
# 导入必要的系统与文件路径处理模块
import os
import sys
# 将父目录加入系统路径，以便导入上级目录中的自定义模块
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# 导入绘图库
import matplotlib.pyplot as plt
# 从自定义dataset包中导入MNIST数据加载函数
from dataset.mnist import load_mnist
# 从common.util中导入曲线平滑函数，用于使损失曲线更光滑
from common.util import smooth_curve
# 导入多层神经网络类
from common.multi_layer_net import MultiLayerNet
# 导入所有优化器类（SGD, Momentum, AdaGrad, Adam 等）
from common.optimizer import *


# 0:读入MNIST数据==========
# 载入MNIST数据集，并将像素值归一化到0~1之间
# 返回 (训练图像, 训练标签), (测试图像, 测试标签)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 训练样本总数
train_size = x_train.shape[0]
# 每次迭代使用的批量大小
batch_size = 128
# 总迭代次数（随机梯度下降的更新步数）
max_iterations = 2000


# 1:进行实验的设置==========
# 定义不同优化器的字典，键为优化器名称，值为对应的优化器实例
optimizers = {}
optimizers['SGD'] = SGD()          # 随机梯度下降
optimizers['Momentum'] = Momentum()  # 动量法
optimizers['AdaGrad'] = AdaGrad()    # AdaGrad自适应学习率
optimizers['Adam'] = Adam()          # Adam自适应矩估计
# 若需要也可以加入 RMSprop 优化器，此处暂时注释掉
#optimizers['RMSprop'] = RMSprop()

# 存储每个优化器对应的神经网络模型
networks = {}
# 存储每个优化器在训练过程中的损失值记录列表
train_loss = {}
# 遍历所有优化器，分别为它们创建结构相同的多层神经网络，并初始化损失记录列表
for key in optimizers.keys():
    # 创建具有四个隐藏层（每层100个神经元）的多层神经网络
    # 输入尺寸为784（28x28图像展平），输出尺寸为10（0~9数字）
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    # 初始化该优化器对应的损失值列表，用于后续记录每次迭代的损失
    train_loss[key] = []    


# 2:开始训练==========
# 进行 max_iterations 次迭代更新
for i in range(max_iterations):
    # 从训练数据中随机抽取 batch_size 个样本的索引（采样掩码）
    batch_mask = np.random.choice(train_size, batch_size)
    # 根据掩码选取当前批量输入图像数据和对应标签
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 对每个优化器分别使用同一批数据进行一次参数更新
    for key in optimizers.keys():
        # 计算当前网络在批量数据上的梯度信息
        grads = networks[key].gradient(x_batch, t_batch)
        # 利用对应优化器根据计算出的梯度更新网络参数
        optimizers[key].update(networks[key].params, grads)
    
        # 计算更新后当前网络在此批量数据上的损失值
        loss = networks[key].loss(x_batch, t_batch)
        # 将本次损失值追加记录到列表中，供后续绘图使用
        train_loss[key].append(loss)
    
    # 每100次迭代输出一次当前各优化器下的损失，便于观察训练进展
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.绘制图形==========
# 为不同优化器定义不同的标记样式，用于绘图区分
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
# 生成横轴迭代次数（0 到 max_iterations-1）
x = np.arange(max_iterations)
# 对每个优化器，绘制经平滑处理后的训练损失曲线
for key in optimizers.keys():
    # smooth_curve 函数会对损失序列进行指数移动平均或其他平滑处理，减少噪声
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
# 设置坐标轴标签
plt.xlabel("iterations")
plt.ylabel("loss")
# 设置纵轴范围便于观察对比
plt.ylim(0, 1)
# 显示图例
plt.legend()
# 显示绘制的图像
plt.show()