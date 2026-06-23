# coding: utf-8
# 导入科学计算库 numpy，用于数组和数学运算
import numpy as np
# 导入 matplotlib 的 pyplot 模块，用于数据可视化
import matplotlib.pylab as plt
# 从 gradient_2d 模块导入数值梯度计算函数 numerical_gradient
# 该函数使用中心差分法近似计算函数 f 在点 x 处的梯度向量
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降法实现

    参数：
        f:         目标函数，接受一个 numpy 数组 x 作为输入
        init_x:    初始点，numpy 数组
        lr:        学习率 (learning rate)，控制每次参数更新的步长
        step_num:  梯度下降的迭代步数

    返回：
        x:         最终收敛到的点（numpy 数组）
        x_history: 记录每一次迭代后参数位置的数组，形状为 (step_num, len(init_x))
    """
    # 当前解，初始化为给定的起始点
    x = init_x
    # 用于存储每一步参数位置的历史记录列表
    x_history = []

    # 执行指定次数的迭代更新
    for i in range(step_num):
        # 保存当前位置的副本到历史记录中（copy 避免后续修改影响记录）
        x_history.append(x.copy())

        # 计算目标函数在当前点 x 处的梯度向量
        grad = numerical_gradient(f, x)
        # 沿着梯度反方向（即函数值下降最快的方向）更新参数
        # 更新步长由学习率 lr 决定
        x -= lr * grad

    # 返回最终解以及整个优化路径的历史
    return x, np.array(x_history)


def function_2(x):
    """
    示例二维目标函数 f(x0, x1) = x0^2 + x1^2
    该函数在原点 (0,0) 取得全局最小值 0，形状是一个开口向上的抛物面
    """
    return x[0]**2 + x[1]**2


# 设置梯度下降的初始点，位于 (-3.0, 4.0)
init_x = np.array([-3.0, 4.0])

# 学习率设定为 0.1，稍大一些以在较少步数内观察到明显收敛
lr = 0.1
# 迭代步数设定为 20 步
step_num = 20
# 调用梯度下降法，获得最终点 x 和优化路径 x_history
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

# ---- 以下为可视化部分 ----

# 绘制水平虚线 y = 0，即 x0 轴，颜色为蓝色虚线
plt.plot([-5, 5], [0, 0], '--b')
# 绘制垂直虚线 x = 0，即 x1 轴，颜色为蓝色虚线
plt.plot([0, 0], [-5, 5], '--b')

# 绘制优化路径上的点：横坐标为 x_history 的第一列（x0），纵坐标为第二列（x1）
# 使用圆形标记 'o' 将路径上的每一步标示出来
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

# 设置 x0 轴的显示范围为 -3.5 到 3.5
plt.xlim(-3.5, 3.5)
# 设置 x1 轴的显示范围为 -4.5 到 4.5
plt.ylim(-4.5, 4.5)
# 设置 x0 轴的标签
plt.xlabel("X0")
# 设置 x1 轴的标签
plt.ylabel("X1")
# 显示绘制的图形
plt.show()