# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    """
    ReLU（Rectified Linear Unit）激活函数层。
    正向传播时将小于等于0的输入置为0，其余保持不变；
    反向传播时，对于正向传播中小于等于0的位置，梯度置为0，其余位置梯度不变。
    """
    def __init__(self):
        # mask 记录正向传播时输入 x 小于等于 0 的位置，用于反向传播时关闭对应梯度
        self.mask = None

    def forward(self, x):
        """
        正向传播：保存 mask，计算并返回 ReLU 激活后的结果。
        """
        # 记录 x <= 0 的位置布尔掩码
        self.mask = (x <= 0)
        out = x.copy()                 # 复制输入，避免修改原始数据
        out[self.mask] = 0             # 将小于等于 0 的元素置为 0
        return out

    def backward(self, dout):
        """
        反向传播：对上游梯度 dout，将正向传播时被置零的位置的梯度也置零。
        返回梯度 dx。
        """
        dout[self.mask] = 0            # 被抑制的神经元不传递梯度
        dx = dout
        return dx


class Sigmoid:
    """
    Sigmoid 激活函数层。
    正向：sigmoid(x) = 1 / (1 + exp(-x))
    反向：dy/dx = y * (1 - y)
    """
    def __init__(self):
        # 保存正向传播的输出值，供反向传播使用
        self.out = None

    def forward(self, x):
        """
        正向传播：计算 sigmoid 输出并保存。
        """
        out = sigmoid(x)               # 调用外部 sigmoid 函数
        self.out = out
        return out

    def backward(self, dout):
        """
        反向传播：利用 y = sigmoid(x) 的导数 y*(1-y) 计算梯度。
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    """
    仿射变换层：执行 y = x·W + b 的正向计算。
    反向传播时计算关于输入 x 的梯度以及关于权重 W 和偏置 b 的梯度。
    """
    def __init__(self, W, b):
        self.W = W                     # 权重矩阵
        self.b = b                     # 偏置向量
        
        self.x = None                  # 正向传播时的输入（展开为二维后）
        self.original_x_shape = None   # 输入原始形状，用于反向传播时恢复数据形状
        # 权重和偏置参数的导数
        self.dW = None                 # 关于 W 的梯度
        self.db = None                 # 关于 b 的梯度

    def forward(self, x):
        """
        正向传播：计算仿射变换，保存输入形状并返回输出。
        """
        # 对应张量：记录输入原始形状，以便在反向时还原
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # 将输入展平成 (批大小, 特征数) 的二维数组
        self.x = x

        out = np.dot(self.x, self.W) + self.b  # 矩阵乘法 + 偏置
        return out

    def backward(self, dout):
        """
        反向传播：计算 dx, dW, db。
        dx 将恢复为输入时的形状（对应张量）。
        """
        dx = np.dot(dout, self.W.T)      # 关于输入 x 的梯度
        self.dW = np.dot(self.x.T, dout) # 关于权重 W 的梯度
        self.db = np.sum(dout, axis=0)   # 关于偏置 b 的梯度（沿批次方向求和）

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    """
    Softmax 与交叉熵损失层。
    正向传播时输入 x 经过 softmax 得到概率分布 y，并与监督标签 t 计算交叉熵损失。
    反向传播返回的梯度为 (y - t) / batch_size。
    """
    def __init__(self):
        self.loss = None               # 损失值
        self.y = None                  # softmax 的输出（概率）
        self.t = None                  # 监督数据（one-hot 或标签索引）

    def forward(self, x, t):
        """
        正向传播：计算 softmax 输出和交叉熵损失。
        """
        self.t = t
        self.y = softmax(x)            # softmax 概率
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """
        反向传播：返回 softmax-with-loss 的梯度。
        梯度为 (y - t) / batch_size。
        """
        batch_size = self.t.shape[0]   # 批次大小
        if self.t.size == self.y.size: # 监督数据是 one-hot 向量的情况
            dx = (self.y - self.t) / batch_size
        else:
            # 监督数据为标签索引的情况，将 y 对应真实类别的概率减 1
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Dropout:
    """
    Dropout 层，用于防止过拟合。
    训练时随机丢弃一部分神经元（将其输出置零），测试时使用缩放后的完整输出。
    论文参考：http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio # 丢弃概率
        self.mask = None                   # 正向传播时生成的随机掩码，用于反向传播

    def forward(self, x, train_flg=True):
        """
        正向传播：训练阶段生成与 x 同形状的随机掩码，按比率丢弃神经元；
        测试阶段对整体输出进行缩放，保持输出的期望不变。
        """
        if train_flg:
            # 生成与 x 形状相同的随机数，大于 dropout_ratio 的位置保留（掩码为 1），否则丢弃（掩码为 0）
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask          # 通过掩码保留有效神经元
        else:
            # 测试阶段不使用丢弃，但需乘以保留概率以保持输出的期望一致
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        """
        反向传播：梯度仅通过正向传播时未被丢弃的神经元向后传递。
        """
        return dout * self.mask


class BatchNormalization:
    """
    批量归一化层，加速训练并提高稳定性。
    论文参考：http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma               # 缩放参数 γ
        self.beta = beta                 # 平移参数 β
        self.momentum = momentum         # 移动平均的动量
        self.input_shape = None          # 输入的形状（Conv 层 4 维，全连接层 2 维）

        # 测试时使用的移动平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 时使用的中间数据
        self.batch_size = None           # 批大小
        self.xc = None                   # 输入减去均值后的中心化数据
        self.std = None                  # 标准差
        self.dgamma = None               # γ 的梯度
        self.dbeta = None                # β 的梯度
        self.xn = None                   # 归一化后的数据

    def forward(self, x, train_flg=True):
        """
        正向传播：记录输入形状并根据是否为训练模式调用 __forward。
        最后将输出恢复为原始形状。
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            # 若输入为 4 维图像数据 (N, C, H, W)，将其展平为 (N, C*H*W)
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        """
        内部正向计算：训练时计算当前批次的均值、方差，并更新全局移动平均值；
        测试时直接使用移动平均值和方差。
        """
        if self.running_mean is None:
            # 首次运行时初始化全局均值和方差为零
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            # 训练阶段：计算当前批次的统计数据
            mu = x.mean(axis=0)                        # 均值
            xc = x - mu                                # 中心化
            var = np.mean(xc**2, axis=0)               # 方差
            std = np.sqrt(var + 10e-7)                 # 标准差（加入小量防止除零）
            xn = xc / std                              # 归一化

            # 保存中间变量供反向传播使用
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # 使用移动平均更新全局 running_mean 和 running_var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 测试阶段：使用全局移动平均值和方差
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta   # 缩放与平移
        return out

    def backward(self, dout):
        """
        反向传播：若输入为 4 维则先将梯度展平，调用 __backward 计算 dx，再恢复形状。
        """
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)  # 恢复原始形状
        return dx

    def __backward(self, dout):
        """
        内部反向计算，依据链式法则计算各中间变量的梯度，并保存 dgamma 和 dbeta。
        返回对输入 x 的梯度 dx。
        """
        dbeta = dout.sum(axis=0)            # 对 β 的梯度（沿批次求和）
        dgamma = np.sum(self.xn * dout, axis=0)  # 对 γ 的梯度（沿批次求 xn * dout 的和）
        dxn = self.gamma * dout             # 传到 xn 的梯度
        dxc = dxn / self.std                # 传到中心化数据 xc 的梯度（第一部分）
        # 下面根据 std 链式求梯度
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)  # 对 std 的梯度
        dvar = 0.5 * dstd / self.std        # 对方差 var 的梯度
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # 传递到 xc 的额外梯度（来自方差）
        dmu = np.sum(dxc, axis=0)           # 对均值 mu 的梯度
        dx = dxc - dmu / self.batch_size    # 最终对输入 x 的梯度

        # 保存可学习参数的梯度
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    """
    卷积层：执行卷积运算（im2col 加速）。
    """
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W                     # 卷积核权重，形状 (FN, C, FH, FW)
        self.b = b                     # 偏置，形状 (FN,)
        self.stride = stride           # 步幅
        self.pad = pad                 # 填充

        # 中间数据（backward 时使用）
        self.x = None                  # 正向传播时的输入数据
        self.col = None                # im2col 展开后的输入矩阵
        self.col_W = None              # 卷积核经 reshape 和转置后的矩阵

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        正向传播：利用 im2col 将输入展开为二维矩阵，再用矩阵乘法实现卷积。
        """
        FN, C, FH, FW = self.W.shape      # 卷积核数量、输入通道数、卷积核高、宽
        N, C, H, W = x.shape              # 批大小、通道数、输入高、宽
        # 计算输出特征图的高度和宽度
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # 将输入图像根据卷积参数展开为矩阵
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 将卷积核权重重排为 (FN, -1) 后转置，方便矩阵乘法
        col_W = self.W.reshape(FN, -1).T

        # 矩阵乘法：col (展平图像块) × col_W (展平卷积核)，再加上偏置
        out = np.dot(col, col_W) + self.b
        # 将输出重排为 (N, out_h, out_w, FN) 并转置为 (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 保存中间变量供反向传播使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        反向传播：利用保存的 col 和卷积核矩阵计算梯度。
        """
        FN, C, FH, FW = self.W.shape
        # 将上游梯度 dout 形状由 (N, FN, out_h, out_w) 转为 (N*out_h*out_w, FN)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)       # 偏置梯度：对每个滤波器求和
        # 卷积核梯度 = col.T（展平图像块矩阵的转置）点乘 dout
        self.dW = np.dot(self.col.T, dout)
        # 恢复 dW 形状为 (FN, C, FH, FW)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 通过 col_W 传递梯度回到图像块矩阵
        dcol = np.dot(dout, self.col_W.T)
        # 利用 col2im 将梯度还原为输入图像形状
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    """
    池化层（最大池化），通过 im2col 实现加速。
    """
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h           # 池化窗口高
        self.pool_w = pool_w           # 池化窗口宽
        self.stride = stride           # 步幅
        self.pad = pad                 # 填充

        self.x = None                  # 正向传播时的输入
        self.arg_max = None            # 记录最大值的索引，反向传播时使用

    def forward(self, x):
        """
        正向传播：通过 im2col 展开，在每个池化窗口内取最大值。
        同时记录最大值的位置用于反向传播。
        """
        N, C, H, W = x.shape
        # 计算输出特征图高和宽
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 利用 im2col 将输入展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 重排为 (-1, pool_h * pool_w)，每行对应一个池化窗口内的所有元素
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 对每一行（每个池化窗口）取最大值和最大值索引
        arg_max = np.argmax(col, axis=1)        # 最大值在窗口内的位置
        out = np.max(col, axis=1)               # 最大值
        # 将输出形状变为 (N, out_h, out_w, C) 再转置为 (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        反向传播：梯度仅传递到正向传播时所选的最大值位置，其余位置梯度为 0。
        """
        # 调整 dout 形状为 (N, C, out_h, out_w) -> (N, out_h, out_w, C) 方便后续操作
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w   # 每个池化窗口的元素个数
        # 初始化全零的梯度矩阵，形状 (总元素数, pool_size)
        dmax = np.zeros((dout.size, pool_size))
        # 利用 arg_max 索引，将上游梯度填充到对应最大值位置
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        # dmax 恢复为 (N, out_h, out_w, C, pool_size)
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # 将 dmax 展开为二维，方便 col2im 恢复
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # 利用 col2im 将梯度映射回原始输入形状
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx