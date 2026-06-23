# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    # 使用中心差分法计算一维数组 x 在函数 f 下的数值梯度
    # 参数 f: 接收一维数组并返回标量的目标函数
    # 参数 x: 一维 numpy 数组，表示求梯度的当前点
    # 返回: 与 x 形状相同的梯度数组
    h = 1e-4 # 0.0001  微小扰动量，用于逼近导数
    grad = np.zeros_like(x)  # 初始化梯度数组，数据类型与形状与 x 一致
    
    for idx in range(x.size):   # 遍历数组中的每一个变量分量
        tmp_val = x[idx]        # 暂存当前分量的原始值，便于后续还原
        # 计算 f(x + h)，即正向扰动后的函数值
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        # 计算 f(x - h)，即负向扰动后的函数值
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        # 利用中心差分公式计算该分量的偏导数（梯度）
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值，避免对其他分量的计算产生副作用
        
    return grad


def numerical_gradient_2d(f, X):
    # 针对二维数组（或者一维数组）计算批量数值梯度
    # 若 X 是一维数组，直接视为单个样本调用一维梯度计算
    # 若 X 是二维数组，则认为每一行是一个独立的输入样本，逐行计算梯度
    # 参数 f: 接收一维数组并返回标量的目标函数（每个样本独立计算）
    # 参数 X: 输入数组，通常形状为 (N, D)，N为样本数，D为维度
    # 返回: 与 X 形状相同的梯度数组
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)  # 存储所有样本梯度的容器，形状与 X 一致
        
        for idx, x in enumerate(X):   # idx 为样本索引，x 为当前样本的一维特征向量
            grad[idx] = _numerical_gradient_1d(f, x)  # 计算当前样本的梯度并存入对应位置
        
        return grad


def numerical_gradient(f, x):
    # 通用的数值梯度计算函数，支持任意维度的输入数组
    # 通过 np.nditer 遍历数组中的每一个元素，逐一计算偏导数
    # 参数 f: 接收与 x 同形状数组并返回标量的目标函数
    # 参数 x: 任意维度的 numpy 数组，表示求梯度的当前点
    # 返回: 与 x 形状相同的梯度数组
    h = 1e-4 # 0.0001  中心差分步长
    grad = np.zeros_like(x)  # 初始化梯度数组，形状与 x 相同
    
    # 创建 numpy 数组迭代器，multi_index 用于获取多维索引，readwrite 允许在迭代过程中修改 x 的元素值
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:        # 当迭代器未遍历完所有元素时继续
        idx = it.multi_index      # 获取当前元素的多维索引（元组形式）
        tmp_val = x[idx]          # 暂存当前元素值，用于后续还原
        # 计算 f(x + h)，即对当前元素施加正向微小扰动后的函数值
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        # 计算 f(x - h)，即对当前元素施加负向微小扰动后的函数值
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        # 中心差分公式计算当前元素对应方向的偏导数
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值，确保数组恢复到原始状态
        it.iternext()    # 移动迭代器至下一个元素
        
    return grad