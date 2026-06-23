# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from matplotlib.image import imread
from common.layers import Convolution

def filter_show(filters, nx=4, show_num=16):
    """
    可视化展示卷积层滤波器（卷积核）的权重图像。
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    :param filters: 卷积核权重数组，形状为(FN, C, FH, FW)，分别表示滤波器数量、通道数、滤波器高、滤波器宽。
    :param nx: 子图网格的列数，即每行显示的滤波器数量，默认为4。
    :param show_num: 需要显示的滤波器总数，默认为16个，要求 <= FN。
    """
    FN, C, FH, FW = filters.shape  # 获取滤波器数量的静态信息
    ny = int(np.ceil(show_num / nx))  # 根据列数计算总共需要的行数，向上取整

    fig = plt.figure()  # 创建一个新的绘图窗口
    # 调整子图之间的边距，使图像紧凑排列
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        # 按4行4列布局添加子图（i+1为子图序号），隐藏坐标轴刻度
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        # 显示第i个滤波器的第一个通道的权重，使用灰度反转色图并最近邻插值以避免平滑
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


# 构建一个简单的卷积神经网络实例
# 输入维度为单通道28x28的图像（如MNIST手写数字）
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},  # 卷积层：30个5x5滤波器，不填充，步长为1
                        hidden_size=100, output_size=10, weight_init_std=0.01)  # 全连接隐藏层100个神经元，输出10类，权重初始化标准差0.01

# 学习后的权重
network.load_params("params.pkl")  # 从文件加载已经训练好的网络参数（权重和偏置）

filter_show(network.params['W1'], 16)  # 可视化展示第一层卷积的16个滤波器权重

# 读取灰度测试图像（如经典的lena图）
img = imread('../dataset/lena_gray.png')
# 将图像重塑为网络需要的四维输入形状：(样本数N=1, 通道数C=1, 高H, 宽W)
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()  # 创建一个新的绘图窗口用于展示卷积后的特征图

w_idx = 1  # 指定使用的卷积核权重索引，此处未实际使用，仅作占位

for i in range(16):
    w = network.params['W1'][i]  # 取出第i个滤波器的权重，形状为(C, FH, FW)
    b = 0  # network.params['b1'][i]   # 偏置项暂时设为0，注释掉的代码表示可以从参数中读取每个滤波器的偏置

    # 将权重调整为四维形状以适配卷积层输入要求：(滤波器数量FN=1, 通道C, 高FH, 宽FW)
    w = w.reshape(1, *w.shape)
    #b = b.reshape(1, *b.shape)  # 偏置同样可调整形状（本处未使用）
    # 实例化一个卷积层，使用当前滤波器的权重和偏置
    conv_layer = Convolution(w, b) 
    out = conv_layer.forward(img)  # 对输入图像进行单滤波器卷积操作，得到输出特征图
    # 卷积输出形状为(1, 1, H', W')，去除批次和通道维度，得到二维特征图
    out = out.reshape(out.shape[2], out.shape[3])
    
    # 以4x4子图形式显示每个滤波器的输出特征图
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()  # 显示所有绘制的图像窗口