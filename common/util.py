# coding: utf-8
# 导入数值计算库 numpy
import numpy as np


def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    # 滑动窗口的大小（用于卷积平滑的点数）
    window_len = 11
    # 构造用于卷积的序列：在原始数据 x 两端进行对称扩展，以减小边界效应
    # np.r_[...] 将左扩展、原始数据、右扩展沿第一个轴拼接
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # 生成凯泽窗（Kaiser window），beta=2，形状接近高斯窗，用于加权平滑
    w = np.kaiser(window_len, 2)
    # 将窗函数归一化后与扩展序列做一维卷积，mode='valid' 表示不填充，返回完全重叠部分
    y = np.convolve(w/w.sum(), s, mode='valid')
    # 去掉卷积结果前后多余的边缘部分，使输出长度与原始 x 一致（这里剪掉前后5个点）
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    # 生成一个随机排列的索引，长度为数据集的样本数（第一维大小）
    permutation = np.random.permutation(x.shape[0])
    # 根据数据维度选择正确的索引方式打乱特征
    # 若数据是2维（如全连接网络的输入），则使用x[permutation,:]
    # 若数据是4维（如图像数据 NCHW 格式），则使用x[permutation,:,:,:]
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    # 用同样的随机排列打乱标签
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """计算卷积层或池化层的输出尺寸（高度或宽度）

    Parameters
    ----------
    input_size : int, 输入特征图的尺寸（高或宽）
    filter_size : int, 滤波器尺寸（高或宽）
    stride : int, 步幅，默认为1
    pad : int, 填充数量，默认为0

    Returns
    -------
    int, 输出特征图的尺寸
    """
    # 根据标准卷积输出尺寸公式计算：(W - F + 2P) / S + 1
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """将4维图像数据转换为2维矩阵，便于使用矩阵乘法实现卷积运算

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组，形状为 (N*out_h*out_w, C*filter_h*filter_w)，每一行对应一个卷积窗口拉伸后的结果
    """
    # 获取输入数据的形状：N（批量大小）、C（通道数）、H（高度）、W（宽度）
    N, C, H, W = input_data.shape
    # 计算卷积输出的高度和宽度
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # 对输入数据在高度和宽度维度上进行零填充（constant 模式填充值为0）
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 初始化一个6维数组用于存放每个窗口的数据
    # 各维度含义：(N, C, filter_h, filter_w, out_h, out_w)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 遍历滤波器的高度方向
    for y in range(filter_h):
        # 计算在填充后图像上高度方向需要提取的起始位置范围
        y_max = y + stride*out_h
        # 遍历滤波器的宽度方向
        for x in range(filter_w):
            # 计算在填充后图像上宽度方向需要提取的起始位置范围
            x_max = x + stride*out_w
            # 按步长提取对应窗口区域，赋值给 col 的对应位置
            # img[:, :, y:y_max:stride, x:x_max:stride] 的形状为 (N, C, out_h, out_w)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 调整维度顺序，以便 reshape 为2维矩阵
    # transpose(0,4,5,1,2,3) 将 (N, C, filter_h, filter_w, out_h, out_w)
    # 变为 (N, out_h, out_w, C, filter_h, filter_w)
    # 然后重塑为 (N*out_h*out_w, C*filter_h*filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col 的逆操作，将2维矩阵还原为4维图像数据，常用于反向传播时累加梯度

    Parameters
    ----------
    col : 2维数组，来自 im2col 的输出或梯度，形状为 (N*out_h*out_w, C*filter_h*filter_w)
    input_shape : 原始输入数据的形状，例如 (10, 1, 28, 28)
    filter_h : 滤波器的高
    filter_w : 滤波器的宽
    stride : 步幅
    pad : 填充

    Returns
    -------
    img : 4维数组，形状与 input_shape 相同的图像数据（通常为梯度累加结果）
    """
    # 解析原始输入的形状
    N, C, H, W = input_shape
    # 计算卷积输出的高度和宽度
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 将2维的 col 重新变形并转置回6维数组 (N, C, filter_h, filter_w, out_h, out_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 初始化一个用于累加结果的图像数组，尺寸稍大以确保能容纳 stride>1 时的索引
    # 多出的 stride-1 是为了处理在步长大于1时可能出现的不对齐问题
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    # 遍历滤波器的高度
    for y in range(filter_h):
        # 计算在当前扩展图像上高度方向写入的范围
        y_max = y + stride*out_h
        # 遍历滤波器的宽度
        for x in range(filter_w):
            # 计算在当前扩展图像上宽度方向写入的范围
            x_max = x + stride*out_w
            # 将 col 中对应滤波器位置的值累加到 img 的对应窗口中
            # 注意这里是累加（+=），以正确处理重叠区域（例如步长 < 滤波器尺寸时）
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 裁剪掉填充部分以及为步长预留的多余边缘，恢复为原输入尺寸
    return img[:, :, pad:H + pad, pad:W + pad]