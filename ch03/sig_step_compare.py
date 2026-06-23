# coding: utf-8
# 导入科学计算库 numpy 以及 matplotlib 的 pylab 绘图接口
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    """
    Sigmoid 激活函数。
    将任意实数输入映射到 (0, 1) 区间，常用于神经网络的激活函数。
    公式: 1 / (1 + exp(-x))
    
    参数 x : 标量或 numpy 数组
    返回   : 与 x 形状相同的 numpy 数组，元素值为 sigmoid 输出
    """
    return 1 / (1 + np.exp(-x))    


def step_function(x):
    """
    阶跃函数（阈值 0）。
    输入大于 0 时输出 1，否则输出 0，是感知机中常用的激活函数。
    
    参数 x : 标量或 numpy 数组
    返回   : 整数类型 numpy 数组，元素为 0 或 1
    """
    # 利用 numpy 的布尔运算生成 True/False 数组，再转换为整型（True->1, False->0）
    return np.array(x > 0, dtype=np.int)


# 在 -5.0 到 5.0 之间以 0.1 为步长生成等差数列，作为 x 轴的采样点
x = np.arange(-5.0, 5.0, 0.1)
# 计算每个采样点对应的 sigmoid 函数值
y1 = sigmoid(x)
# 计算每个采样点对应的阶跃函数值
y2 = step_function(x)

# 绘制 sigmoid 曲线，默认蓝色实线
plt.plot(x, y1)
# 绘制阶跃函数曲线，'k--' 表示黑色（black）虚线
plt.plot(x, y2, 'k--')
# 设置 y 轴的显示范围，留出少量边距使图像更清晰
plt.ylim(-0.1, 1.1)  # 指定图中绘制的 y 轴的范围
# 展示绘制结果
plt.show()