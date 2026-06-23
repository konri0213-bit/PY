# coding: utf-8
# 指定源文件编码为 UTF-8，以支持中文等字符
import numpy as np
# 导入数值计算库 NumPy，用于高效处理多维数组和数学运算
import matplotlib.pylab as plt
# 导入 matplotlib 的 pylab 模块作为绘图接口，提供类似 MATLAB 的绘图功能


def sigmoid(x):
    """
    计算 Sigmoid 激活函数的值。
    
    Sigmoid 函数公式为：1 / (1 + exp(-x))
    它将任意实数压缩到 (0, 1) 区间内，常用于二分类的概率输出。
    
    参数：
        x : 数值或 NumPy 数组
    返回：
        与 x 同型的数值或数组，元素值为对应的 Sigmoid 输出
    """
    return 1 / (1 + np.exp(-x))
    # np.exp(-x) 计算 e 的 -x 次方，利用 NumPy 的向量化特性可对整个数组逐元素计算


# 生成输入数据：从 -5.0 到 5.0，步长为 0.1 的等差数组
X = np.arange(-5.0, 5.0, 0.1)
# 计算整个输入序列对应的 Sigmoid 输出值
Y = sigmoid(X)

# 绘制 Sigmoid 函数曲线，X 为横坐标，Y 为纵坐标
plt.plot(X, Y)
# 设置 y 轴的显示范围，稍微超出 [0, 1] 以便完整清晰地观察曲线的饱和趋势
plt.ylim(-0.1, 1.1)
# 展示绘制的图形窗口
plt.show()