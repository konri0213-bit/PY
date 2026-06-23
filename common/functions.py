# coding: utf-8
import numpy as np


def identity_function(x):
    # 恒等函数：直接将输入原样返回，常用于回归问题的输出层
    return x


def step_function(x):
    # 阶跃函数：输入大于0时返回1，否则返回0
    # 将布尔数组转换为整型数组，确保输出为0或1
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    # Sigmoid 激活函数：将输入映射到 (0, 1) 区间，常用于二分类问题的输出层或隐藏层
    # 公式：1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    # Sigmoid 函数的导数（梯度）
    # 利用 sigmoid 导数的性质：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    # ReLU (Rectified Linear Unit) 激活函数：将负数置零，正数保持不变
    # 可以缓解梯度消失问题，是现代深层网络常用的激活函数
    return np.maximum(0, x)


def relu_grad(x):
    # ReLU 函数的导数（梯度）
    # 当 x >= 0 时导数为1，当 x < 0 时导数为0
    grad = np.zeros(x)            # 初始化梯度矩阵全为0
    grad[x>=0] = 1                # 将输入中非负元素对应的梯度设为1
    return grad
    

def softmax(x):
    # Softmax 函数：将输入向量转换为概率分布，所有输出之和为1，常用于多分类问题的输出层
    # 支持批量处理（当 x 为二维数组时）
    if x.ndim == 2:
        # 二维情况：每一行代表一个样本的特征向量
        x = x.T                   # 转置，以便按列（原始样本）进行运算
        x = x - np.max(x, axis=0) # 减去每列的最大值，防止指数运算溢出（稳定化策略）
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 对每列计算 softmax
        return y.T               # 转置回原形状，使每行仍对应一个样本的 softmax 输出

    # 一维情况：处理单个样本的特征向量
    x = x - np.max(x) # 溢出对策：减去最大值，防止 exp 计算出现上溢，同时不影响最终概率比值
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    # 均方误差（MSE）损失函数：常用于回归任务
    # y：神经网络输出（预测值），t：监督数据（真实值）
    # 公式：(1/2) * Σ(y - t)^2 ，系数 0.5 为方便求导时约去平方的2
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    # 交叉熵误差损失函数：常用于分类任务
    # y：神经网络输出（通常是 softmax 后的概率），t：监督数据（one-hot 表示或标签索引）
    # 先处理维度，统一成二维批量形式
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 将一维标签重塑为二维行向量，代表批量大小为1
        y = y.reshape(1, y.size)  # 将一维预测输出重塑为二维行向量
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)      # 从 one-hot 编码中提取类别索引（沿列方向取最大值的索引）
             
    batch_size = y.shape[0]       # 获取当前批次的样本数量
    # 计算交叉熵误差：- Σ log(正确解对应的预测概率) / batch_size
    # y[np.arange(batch_size), t] 利用 fancy indexing 取出每个样本正确类别对应的预测概率
    # 加微小值 1e-7 是为了防止 log(0) 导致的计算错误
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    # 组合 Softmax 激活函数与交叉熵误差损失的复合损失函数
    # 常用于多分类神经网络输出层的损失计算，相比分开计算能利用解析梯度减少计算量
    # 但此处实现为直接调用二者，保持功能简洁
    y = softmax(X)                # 先通过 softmax 得到类别概率
    return cross_entropy_error(y, t)  # 再计算与真实标签之间的交叉熵误差