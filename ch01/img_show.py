# coding: utf-8
# 导入 matplotlib 的 pyplot 模块，用于图像的展示与绘制
import matplotlib.pyplot as plt
# 从 matplotlib.image 中导入 imread 函数，用于读取图像文件为数组
from matplotlib.image import imread

# 使用 imread 读取指定路径下的图像文件，将图像数据存储为 NumPy 数组
# 变量 img 保存了图像的像素矩阵，通常形状为 (高度, 宽度, 颜色通道数)
img = imread('../dataset/lena.png') #读入图像
# 调用 imshow 函数将图像数组可视化显示在画布上
plt.imshow(img)

# 展示所有已绘制的图形窗口，并进入事件监听状态，直到窗口被关闭
plt.show()