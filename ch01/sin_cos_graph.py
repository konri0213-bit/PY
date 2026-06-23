# coding: utf-8
# 导入必要的科学计算与绘图库
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
# 以0.1为步长，生成从0到6的等差数列，作为横轴坐标
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
# 计算 x 对应的正弦值，作为第一条曲线的纵轴数据
y1 = np.sin(x)
# 计算 x 对应的余弦值，作为第二条曲线的纵轴数据
y2 = np.cos(x)

# 绘制图形
# 绘制正弦曲线，实线，图例标签为 "sin"
plt.plot(x, y1, label="sin")
# 绘制余弦曲线，虚线，图例标签为 "cos"
plt.plot(x, y2, linestyle = "--", label="cos")
# 设置 x 轴标签
plt.xlabel("x") # x轴的标签
# 设置 y 轴标签
plt.ylabel("y") # y轴的标签
# 设置图表标题
plt.title('sin & cos')
# 显示图例，自动识别已设置的 label
plt.legend()
# 将绘制的图形在窗口中显示出来
plt.show()