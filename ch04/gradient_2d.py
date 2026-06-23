# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    """使用中心差分对单个点 x 计算函数 f 的数值梯度（非批量版本）。
    参数 f 是接收一维数组并返回标量的函数，x 是一维 numpy 数组。
    返回与 x 形状相同的梯度数组。
    """
    # 微小扰动步长，用于逼近导数
    h = 1e-4 # 0.0001
    # 初始化梯度数组，与输入 x 形状相同
    grad = np.zeros_like(x)
    
    # 对每个维度独立计算偏导数
    for idx in range(x.size):
        # 暂存当前维度的原始值，以便计算后还原
        tmp_val = x[idx]
        
        # 计算 f(x + h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        # 计算 f(x - h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        
        # 中心差分公式：(f(x+h) - f(x-h)) / (2h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        # 将 x 在该维度的值还原，避免副作用
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    """数值梯度计算，支持单点与批量输入。
    如果 X 是一维数组（单个点），直接调用 _numerical_gradient_no_batch；
    如果 X 是二维数组（多个点按行排列），则对每一行分别计算梯度，返回所有梯度的二维数组。
    """
    # 一维数组：单点情况
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        # 初始化二维梯度数组，与 X 形状相同
        grad = np.zeros_like(X)
        
        # 对 X 中的每一个点（每一行）分别计算梯度
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    """示例多元函数：f(x0, x1) = x0**2 + x1**2（平方和）。
    输入 x 可以是单个点（一维数组）或一批点（二维数组）。
    返回函数值或每个样本的函数值所组成的一维数组。
    """
    # 单个点：直接返回所有维度的平方和（标量）
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        # 批量点（二维数组，每行是一个点）：沿 axis=1 计算每行的平方和，返回一维数组
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    """根据函数 f 在点 x 处的梯度，返回该点处切线（在二维情况下为切平面）的线性近似函数。
    返回一个 lambda 函数，其参数 t 为自变量，值为 f(x) + 梯度·(t - x) 的线性展开结果。
    这里为简单可视化，使用 y = f(x) - d*x 形式，保证在 t = x 处与 f(x) 一致，斜率由梯度 d 给出。
    """
    # 计算 f 在点 x 处的数值梯度 d
    d = numerical_gradient(f, x)
    # 打印梯度值供调试
    print(d)
    # 截距项，使得 t = x 时切线函数值与 f(x) 相等（详细推导：y = f(x) - d * x）
    y = f(x) - d*x
    # 返回线性函数：t -> d*t + y
    return lambda t: d*t + y
     
if __name__ == '__main__':
    # 生成沿两个坐标轴的一维点序列，作为网格点的坐标
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    # 根据 x0, x1 生成二维网格坐标矩阵（但后续会扁平化）
    X, Y = np.meshgrid(x0, x1)
    
    # 将网格坐标矩阵扁平化为一维数组，方便批量计算
    X = X.flatten()
    Y = Y.flatten()
    
    # 计算函数 function_2 在所有网格点处的数值梯度
    # np.array([X, Y]) 的形状为 (2, N)，其中 N 为点数，代表批量输入
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    # 创建图形
    plt.figure()
    # 绘制梯度向量场：在每个 (X, Y) 点绘制一个箭头，方向为负梯度方向（梯度下降方向）
    # -grad[0], -grad[1] 表示梯度的相反方向（指向函数值下降最快的方向）
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    # 设置坐标轴范围
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    # 坐标轴标签
    plt.xlabel('x0')
    plt.ylabel('x1')
    # 显示网格
    plt.grid()
    # 图例（本例中未指定标签，但保留调用）
    plt.legend()
    # 绘制图形
    plt.draw()
    # 显示图形窗口
    plt.show()