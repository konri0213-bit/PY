# coding: utf-8
import numpy as np  # 导入NumPy库，用于高效的数值计算与数组操作
import matplotlib.pylab as plt  # 导入matplotlib的pyplot模块，用于绘制图形


def relu(x):
    """ReLU（Rectified Linear Unit，修正线性单元）激活函数。
    对于输入数组x中的每个元素，若大于0则保留原值，否则输出0。
    """
    return np.maximum(0, x)  # 逐元素比较，返回x与0之间的最大值，实现ReLU功能

# 生成一个从-5.0到5.0（含起始值，不含终止值）的一维数组，步长为0.1
x = np.arange(-5.0, 5.0, 0.1)
# 计算x数组对应的ReLU函数值，得到输出数组y
y = relu(x)
# 绘制ReLU函数曲线，以x为横坐标，y为纵坐标
plt.plot(x, y)
# 设置y轴的显示范围，下界为-1.0，上界为5.5，更清晰地展示ReLU的非负特性
plt.ylim(-1.0, 5.5)
# 显示绘制的图形
plt.show()