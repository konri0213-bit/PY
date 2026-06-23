# coding: utf-8
import numpy as np                          # 导入NumPy数值计算库，用于生成随机数据和数学运算
from simple_convnet import SimpleConvNet    # 从自定义模块中导入简单卷积神经网络类

# 初始化一个简单的卷积神经网络
# 参数说明：
# input_dim : 输入数据的维度 (通道数, 高度, 宽度)，此处为单通道10x10的图像
# conv_param : 卷积层的超参数字典
#   - filter_num : 卷积滤波器（卷积核）的数量
#   - filter_size : 滤波器的大小（正方形，边长为3）
#   - pad : 输入数据的边缘填充宽度，0表示不填充
#   - stride : 滤波器滑动的步幅
# hidden_size : 全连接隐藏层的神经元数量
# output_size : 输出层的神经元数量（通常等于分类的类别数）
# weight_init_std : 权重初始化时使用的高斯分布的标准差
network = SimpleConvNet(input_dim=(1,10, 10), 
                        conv_param = {'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)

# 随机生成一个批次大小为1的输入数据，形状变为 (样本数, 通道数, 高度, 宽度) = (1,1,10,10)
X = np.random.rand(100).reshape((1, 1, 10, 10))
# 构造对应的正确标签（监督数据），形状为 (1,1)，这里标签值为1（索引从0开始表示第2类）
T = np.array([1]).reshape((1,1))

# 通过数值微分方法计算损失函数关于网络各参数的梯度，用于后续的梯度检验
grad_num = network.numerical_gradient(X, T)
# 通过反向传播算法（解析法）计算损失函数关于网络各参数的梯度
grad = network.gradient(X, T)

# 逐一比较数值梯度与反向传播梯度之间的差异，验证反向传播算法实现的正确性
for key, val in grad_num.items():
    # 计算两种梯度之差的绝对值的平均值，该值越小说明反向传播实现得越准确
    print(key, np.abs(grad_num[key] - grad[key]).mean())