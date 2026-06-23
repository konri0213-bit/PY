# coding: utf-8
# 定义一个乘法层，用于计算图中实现乘法操作及其反向传播


class MulLayer:
    def __init__(self):
        """初始化乘法层，前向传播时的输入变量暂置为None"""
        self.x = None  # 保存前向传播时的第一个输入变量x
        self.y = None  # 保存前向传播时的第二个输入变量y

    def forward(self, x, y):
        """前向传播：计算两个输入x与y的乘积，并保存输入值用于反向传播"""
        self.x = x    # 缓存输入x，供反向传播计算梯度时使用
        self.y = y    # 缓存输入y，供反向传播计算梯度时使用
        out = x * y   # 计算乘法输出

        return out    # 返回前向传播结果

    def backward(self, dout):
        """反向传播：根据上游传来的梯度dout，计算本层关于x和y的梯度"""
        dx = dout * self.y  # x的梯度 = 上游梯度 * y （因为 ∂(x*y)/∂x = y）
        dy = dout * self.x  # y的梯度 = 上游梯度 * x （因为 ∂(x*y)/∂y = x）

        return dx, dy       # 返回关于两个输入的梯度


class AddLayer:
    def __init__(self):
        """初始化加法层（加法操作不需要缓存输入，故直接pass）"""
        pass

    def forward(self, x, y):
        """前向传播：计算两个输入x与y的和"""
        out = x + y   # 计算加法输出

        return out    # 返回前向传播结果

    def backward(self, dout):
        """反向传播：根据上游传来的梯度dout，计算本层关于x和y的梯度"""
        dx = dout * 1  # 加法对x的偏导数为1，将上游梯度原样传递
        dy = dout * 1  # 加法对y的偏导数同样为1，将上游梯度原样传递

        return dx, dy  # 返回关于两个输入的梯度（梯度值均等于dout）