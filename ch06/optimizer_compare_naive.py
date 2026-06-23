# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np  # 导入NumPy，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib，用于绘图
from collections import OrderedDict  # 导入OrderedDict，用于保持优化器顺序
from common.optimizer import *  # 导入自定义的优化器类（SGD、Momentum、AdaGrad、Adam等）


def f(x, y):
    """目标函数：一个扁平的二维二次函数，x方向变化平缓，y方向变化陡峭"""
    return x**2 / 20.0 + y**2


def df(x, y):
    """目标函数的梯度：分别对x和y求偏导数"""
    return x / 10.0, 2.0*y  # df/dx = x/10, df/dy = 2y

init_pos = (-7.0, 2.0)  # 参数的初始位置
params = {}  # 参数字典，用于存储当前的x和y值
params['x'], params['y'] = init_pos[0], init_pos[1]  # 初始化参数
grads = {}  # 梯度字典，用于存储当前梯度值
grads['x'], grads['y'] = 0, 0  # 初始化梯度为0


optimizers = OrderedDict()  # 有序字典，按添加顺序保存不同优化器
optimizers["SGD"] = SGD(lr=0.95)         # 随机梯度下降，学习率0.95
optimizers["Momentum"] = Momentum(lr=0.1) # 动量法，学习率0.1
optimizers["AdaGrad"] = AdaGrad(lr=1.5)   # AdaGrad，学习率1.5
optimizers["Adam"] = Adam(lr=0.3)         # Adam，学习率0.3

idx = 1  # 子图索引，用于2x2布局

for key in optimizers:  # 依次遍历每种优化器
    optimizer = optimizers[key]  # 获取当前优化器实例
    x_history = []  # 记录参数x的历史路径
    y_history = []  # 记录参数y的历史路径
    params['x'], params['y'] = init_pos[0], init_pos[1]  # 每种优化器都从同一个初始点开始
    
    for i in range(30):  # 进行30次参数更新
        x_history.append(params['x'])  # 保存当前x坐标
        y_history.append(params['y'])  # 保存当前y坐标
        
        grads['x'], grads['y'] = df(params['x'], params['y'])  # 计算当前位置的梯度
        optimizer.update(params, grads)  # 使用当前优化器更新参数
    

    # 生成网格点，用于绘制等高线
    x = np.arange(-10, 10, 0.01)  # x轴范围[-10, 10]，步长0.01
    y = np.arange(-5, 5, 0.01)    # y轴范围[-5, 5]，步长0.01
    
    X, Y = np.meshgrid(x, y)  # 生成网格坐标矩阵
    Z = f(X, Y)  # 计算每个网格点上的函数值
    
    # 为了简化等高线图，将函数值大于7的部分置为0（使等高线更清晰）
    mask = Z > 7
    Z[mask] = 0
    
    # 绘制当前优化器的优化路径
    plt.subplot(2, 2, idx)  # 创建2x2的子图，当前位置为idx
    idx += 1  # 索引递增
    plt.plot(x_history, y_history, 'o-', color="red")  # 红色圆圈连线绘制参数变化轨迹
    plt.contour(X, Y, Z)  # 绘制目标函数的等高线
    plt.ylim(-10, 10)  # 设置y轴显示范围
    plt.xlim(-10, 10)  # 设置x轴显示范围
    plt.plot(0, 0, '+')  # 在原点(0,0)即最小值位置画一个'+'号
    #colorbar()  # 可显示颜色条（已被注释）
    #spring()    # 可设置颜色映射（已被注释）
    plt.title(key)  # 设置子图标题为优化器名称
    plt.xlabel("x")  # x轴标签
    plt.ylabel("y")  # y轴标签
    
plt.show()  # 显示所有子图