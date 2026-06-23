# coding: utf-8
import numpy as np              # 导入 NumPy 库，提供高效的数组运算和数学函数
import matplotlib.pyplot as plt # 导入 Matplotlib 的 pyplot 模块，用于数据可视化

# 生成数据
# 创建自变量 x：从 0 开始，到 6 结束（不包含 6），步长为 0.1
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
# 计算因变量 y：对数组 x 中的每个元素求正弦值，结果仍为 NumPy 数组
y = np.sin(x)

# 绘制图形
# 在默认的坐标轴上绘制 x 和 y 的折线图（即正弦曲线）
plt.plot(x, y)
# 显示当前所有已绘制的图形窗口，程序会在此处阻塞直至窗口关闭
plt.show()