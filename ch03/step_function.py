# coding: utf-8
# 导入科学计算库 NumPy，用于高效的数组和数学运算
import numpy as np
# 导入 matplotlib 的 pyplot 模块，提供类似 MATLAB 的绘图接口
import matplotlib.pylab as plt


# 定义阶跃函数（单位阶跃函数）
# 该函数会将输入 x 中大于 0 的元素置为 1，其余置为 0
def step_function(x):
    # x > 0 返回一个布尔数组，dtype=np.int 将 True 转换为 1，False 转换为 0
    return np.array(x > 0, dtype=np.int)

# 生成从 -5.0 到 5.0，步长为 0.1 的等差数列，作为图形的 x 轴数据
X = np.arange(-5.0, 5.0, 0.1)
# 对 X 中的每个元素计算阶跃函数的输出，得到对应的 y 轴数据
Y = step_function(X)
# 绘制阶跃函数的图像，默认会以折线连接各数据点
plt.plot(X, Y)
# 设置 y 轴的显示范围，使图形在垂直方向上留出一些边距，避免线条紧贴边框
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
# 显示绘制好的图形
plt.show()