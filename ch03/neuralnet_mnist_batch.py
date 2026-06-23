# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    """
    获取经过预处理的 MNIST 测试数据集。
    
    返回:
        x_test : 归一化并展平后的测试图像数据
        t_test : 测试标签（非 one-hot 编码形式）
    """
    # 加载 MNIST 数据集，仅保留测试部分；图像归一化、展平为一维向量，标签保持原始类别编号
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
    从预训练参数文件初始化网络权重与偏置。
    
    返回:
        network : 包含各层参数（W1,b1,W2,b2,W3,b3）的字典
    """
    # 以二进制读取模式打开已保存的参数字典
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    对输入数据进行前向传播，返回各样本的分类概率。
    
    参数:
        network : 包含权重和偏置的字典
        x       : 输入数据，形状为 (batch_size, input_size)
    
    返回:
        y : 经过 softmax 后的输出概率，形状为 (batch_size, num_classes)
    """
    # 从参数字典中提取各层的权重矩阵与偏置向量
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 第一层：线性变换 → sigmoid 激活
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    # 第二层：线性变换 → sigmoid 激活
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    # 第三层（输出层）：线性变换 → softmax 得到类别概率
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


# 获取测试数据和预训练网络
x, t = get_data()
network = init_network()

batch_size = 100  # 批处理大小，每次对 100 个样本同时进行预测
accuracy_cnt = 0   # 记录正确分类的样本总数

# 按批次遍历测试集
for i in range(0, len(x), batch_size):
    # 取出当前批次的数据
    x_batch = x[i:i+batch_size]
    # 对批次数据进行预测，得到每个样本对各类别的概率
    y_batch = predict(network, x_batch)
    # 选取概率最大的类别作为预测结果（axis=1 表示沿着样本方向取最大索引）
    p = np.argmax(y_batch, axis=1)
    # 统计当前批次中预测正确的样本数并累加
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

# 打印整个测试集上的分类准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))