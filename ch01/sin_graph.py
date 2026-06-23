# coding: utf-8
import numpy as np              # 导入NumPy库，用于高效的数值计算
import matplotlib.pyplot as plt # 导入Matplotlib的pyplot模块，用于数据可视化

# 生成数据
# 使用NumPy生成从0到6（不包含6）步长为0.1的等差数列，作为自变量x
x = np.arange(0, 6, 0.1)
# 计算x中每个元素的正弦值，得到因变量y，形成正弦波数据
y = np.sin(x)

# 绘制图形
# 使用pyplot的plot函数绘制x和y的折线图
plt.plot(x, y)
# 显示绘制的图形窗口
plt.show()