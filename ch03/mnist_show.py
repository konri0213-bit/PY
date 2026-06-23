# coding: utf-8
# 导入系统与路径处理模块，用于设置父目录的引用路径
import sys, os
# 将父目录添加到模块搜索路径中，以便后续导入位于父目录中的自定义包（如 dataset）
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
# 导入 MNIST 数据集加载函数
from dataset.mnist import load_mnist
# 导入 PIL 库的 Image 模块，用于图像显示
from PIL import Image

# ----------------------------------------------------------------
# 函数: img_show
# 作用: 将 NumPy 数组格式的图像数据转换为 PIL 图像对象并直接显示
# 参数:
#   img : 形状为 (H, W) 的二维数组，表示灰度图像；
#         像素值可以是浮点数或整数，内部会统一转为 uint8 类型
# 说明:
#   该函数不进行图像保存，直接调用系统默认的图像查看器显示图像
# ----------------------------------------------------------------
def img_show(img):
    # 使用 np.uint8 确保像素值类型与 PIL 要求的 8 位无符号整型一致
    pil_img = Image.fromarray(np.uint8(img))
    # 调用系统默认应用打开并显示图像
    pil_img.show()

# ----------------------------------------------------------------
# 加载 MNIST 数据集
# flatten=True  : 将每张 28×28 的图像展平为 784 个元素的一维数组
# normalize=False: 保持像素原始值（0~255 整数），不做归一化处理
# 返回值:
#   (x_train, t_train): 训练图像与对应的标签
#   (x_test, t_test)  : 测试图像与对应的标签
# ----------------------------------------------------------------
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 取出训练集中的第一张图像及其标签
img = x_train[0]
label = t_train[0]
# 打印标签，期望输出为 5
print(label)  # 5

# 打印当前图像数组的形状，这里因为展平，应为 (784,)
print(img.shape)  # (784,)
# 将一维数组重新变形为原始的 28×28 二维图像矩阵，以便查看图像内容
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
# 打印变形后的形状，应为 (28, 28)
print(img.shape)  # (28, 28)

# 调用 img_show 将这幅图像显示出来
img_show(img)