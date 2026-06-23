# coding: utf-8
# 该脚本用于评估已训练好的深度卷积网络（DeepConvNet）在 MNIST 测试集上的性能，
# 并可视化部分分类错误的样本，以便分析模型的薄弱点。
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定（将父目录加入模块搜索路径，便于导入自定义包）
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet          # 导入预先定义好的深度卷积网络类
from dataset.mnist import load_mnist          # 导入 MNIST 数据加载函数


# 加载 MNIST 测试数据，flatten=False 表示保留图像原始形状 (N, 1, 28, 28) 以适配卷积层输入
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 实例化深度卷积网络并加载预训练好的参数文件
network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

print("calculating test accuracy ... ")
# 若只想在小样本上快速测试，可取消下面三行注释，只取前1000个样本
#sampled = 1000
#x_test = x_test[:sampled]
#t_test = t_test[:sampled]

# 用于存储每一批样本对应的预测类别标签
classified_ids = []

# 累加预测正确的样本数
acc = 0.0
# 批大小，可根据内存调整
batch_size = 100

# 按 batch_size 分批遍历测试集（注意：这里仅处理完整批次，末尾不足一批的数据会被丢弃）
for i in range(int(x_test.shape[0] / batch_size)):
    # 提取当前批次的图像数据和标签
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    # 前向传播，train_flg=False 表示不使用 Dropout 等训练时特有的行为
    y = network.predict(tx, train_flg=False)
    # 获取每张图像预测概率最大的类别索引，即预测标签
    y = np.argmax(y, axis=1)
    # 将该批预测结果保存到列表中（y 的形状为 (batch_size,) ）
    classified_ids.append(y)
    # 统计该批次中预测正确的样本个数并累加
    acc += np.sum(y == tt)
    
# 计算整体准确率（注意：分母使用了全部测试样本数，但因丢弃不足一批的部分，
# 实际参与预测的样本数可能略少，这里按原始测试集大小计算准确率）
acc = acc / x_test.shape[0]
print("test accuracy:" + str(acc))

# 将所有批次的预测结果合并为一个一维数组，方便逐样本比较
classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()
 
# 最多显示的错误分类样本数量
max_view = 20
# 当前正在绘制的子图位置编号（1 ~ max_view）
current_view = 1

# 创建画布并调整子图边距，为图像留出更紧凑的展示空间
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

# 字典用于记录错误分类样本的详细信息，键为视图编号，值为 (真实标签, 预测标签)
mis_pairs = {}
# 遍历所有测试样本，val 为 True 表示预测正确，False 表示预测错误
for i, val in enumerate(classified_ids == t_test):
    if not val:
        # 创建一个 4 行 5 列的子图网格，指定当前位置，并隐藏坐标轴刻度
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        # 显示该错误样本的灰度图像（原始形状为 28x28）
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        # 记录错误信息：视图编号 -> (真实标签, 预测标签)
        mis_pairs[current_view] = (t_test[i], classified_ids[i])
            
        current_view += 1
        # 若已显示完设定的最大数量，则停止绘制
        if current_view > max_view:
            break

# 输出错误分类结果概要信息
print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

# 显示所有绘制的图像
plt.show()