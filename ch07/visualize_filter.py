# coding: utf-8
# 导入科学计算库，用于处理多维数组
import numpy as np
# 导入绘图库，用于可视化滤波器
import matplotlib.pyplot as plt
# 从自定义模块导入简单卷积神经网络类
from simple_convnet import SimpleConvNet

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    可视化卷积核（滤波器）。
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    参数：
        filters : 四维权重数组，形状为 (滤波器数量, 通道数, 高度, 宽度)
        nx      : 每行显示的滤波器个数，默认为8
        margin  : 子图之间的边距（当前实现中未显式使用）
        scale   : 显示缩放比例（当前实现中未显式使用）
    """
    # 获取滤波器数组的形状信息
    # FN: 滤波器总数, C: 输入通道数, FH: 滤波器高度, FW: 滤波器宽度
    FN, C, FH, FW = filters.shape
    # 根据滤波器总数和每行最大显示数，计算需要的子图行数（向上取整）
    ny = int(np.ceil(FN / nx))

    # 创建一个新的图形窗口
    fig = plt.figure()
    # 调整子图布局，使子图紧密排列，减少边框和间距
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # 遍历每一个滤波器，将其绘制到对应的子图中
    for i in range(FN):
        # 添加子图，位置索引从1开始；隐藏坐标轴刻度
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        # 显示滤波器第一个通道的权重，使用反向灰度颜色映射，最近邻插值保持像素块状显示
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    # 展示所有子图
    plt.show()


# 实例化一个简单的卷积神经网络（此时权重被随机初始化）
network = SimpleConvNet()
# 随机进行初始化后的权重
# 可视化第一层卷积的初始随机权重
filter_show(network.params['W1'])

# 学习后的权重
# 从文件中加载预训练好的模型参数
network.load_params("params.pkl")
# 可视化学习后的第一层卷积权重，观察其学到的特征模式
filter_show(network.params['W1'])