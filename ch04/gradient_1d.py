# coding: utf-8
import numpy as np                # 导入NumPy库，用于高效的数值计算
import matplotlib.pylab as plt    # 导入matplotlib的pylab模块，用于绘图

def numerical_diff(f, x):
    """
    使用中心差分法计算函数 f 在点 x 处的数值导数
    中心差分相较前向/后向差分具有更高的精度
    参数:
        f : 函数对象
        x : 求导点
    返回:
        f 在 x 处的导数值
    """
    h = 1e-4 # 定义微小步长 0.0001，用于近似极限过程
    return (f(x+h) - f(x-h)) / (2*h)  # 中心差分公式: (f(x+h)-f(x-h)) / 2h

def function_1(x):
    """
    示例函数：f(x) = 0.01 * x^2 + 0.1 * x
    用于后续求导和切线计算
    """
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    """
    计算函数 f 在点 x 处的切线方程，并以闭包形式返回切线函数
    步骤：
        1. 计算该点处的斜率 d （数值导数）
        2. 计算切线的截距（当 x=0 时的 y 值）：y = f(x) - d * x
        3. 返回切线函数：t -> d * t + y
    参数:
        f : 原函数
        x : 切点横坐标
    返回:
        一个 lambda 函数，输入 t 返回切线上对应 t 的纵坐标
    """
    d = numerical_diff(f, x)          # 求函数在 x 点的导数（切线斜率）
    print(d)                          # 打印斜率值，便于观察导数结果
    y = f(x) - d*x                    # 计算切线在 x=0 处的截距（直线方程: y = d*x + intercept）
    return lambda t: d*t + y          # 返回切线方程对应的匿名函数

# 主程序开始
x = np.arange(0.0, 20.0, 0.1)        # 生成从 0 到 20 步长为 0.1 的数组，作为 x 轴数据
y = function_1(x)                     # 计算原函数在 x 各点上的函数值
plt.xlabel("x")                       # 设置 x 轴标签为 "x"
plt.ylabel("f(x)")                    # 设置 y 轴标签为 "f(x)"

tf = tangent_line(function_1, 5)      # 获取 function_1 在 x=5 处的切线函数
y2 = tf(x)                            # 根据切线函数，计算该切线上与 x 数据点对应的 y 值

plt.plot(x, y)                        # 绘制原函数 f(x) 的曲线
plt.plot(x, y2)                       # 绘制 x=5 处的切线
plt.show()                            # 显示绘制的图形