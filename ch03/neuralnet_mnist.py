# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # 加载 MNIST 数据集，仅返回测试图像和测试标签
    # normalize=True: 将像素值归一化到 0.0 ~ 1.0 范围
    # flatten=True: 将 28x28 的图像展平为一维数组（784）
    # one_hot_label=False: 标签保持原始的数字形式（0~9），而非 one-hot 编码
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    # 从保存的 pickle 文件中加载预训练好的网络参数（权重和偏置）
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # 从参数字典中提取各层的权重矩阵和偏置向量
    # W1, W2, W3: 第1、2、3层的权重矩阵
    # b1, b2, b3: 第1、2、3层的偏置向量
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 前向传播：输入层 -> 第1层 -> 激活函数 sigmoid
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 第1层 -> 第2层 -> 激活函数 sigmoid
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 第2层 -> 第3层（输出层） -> softmax 函数得到概率分布 y
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 获取测试数据：x 为测试图像，t 为对应的正确标签
x, t = get_data()
# 初始化网络，加载已训练好的参数
network = init_network()
accuracy_cnt = 0
# 遍历测试集的每一张图像
for i in range(len(x)):
    # 使用当前网络对单张图像进行预测，得到每个类别的概率
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引，即预测的类别
    # 若预测类别与真实标签一致，则正确计数加一
    if p == t[i]:
        accuracy_cnt += 1

# 计算并输出模型在测试集上的准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))