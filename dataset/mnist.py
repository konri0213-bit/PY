# coding: utf-8
# 尝试导入 Python 3 的 urllib.request 模块，如果失败则提示需要使用 Python 3
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

# MNIST 数据集的基础下载 URL
url_base = 'http://yann.lecun.com/exdb/mnist/'
# 数据集文件的远程名称与本地标识的对应字典
# 分别对应训练图像、训练标签、测试图像、测试标签
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# 数据集存放目录，设定为当前脚本所在目录
dataset_dir = os.path.dirname(os.path.abspath(__file__))
# 预处理后保存的 pickle 文件路径
save_file = dataset_dir + "/mnist.pkl"

# 训练样本数量
train_num = 60000
# 测试样本数量
test_num = 10000
# 图像维度：(通道数, 高, 宽)  MNIST 为单通道 28x28 灰度图
img_dim = (1, 28, 28)
# 图像展开为一维向量后的长度 28*28 = 784
img_size = 784


def _download(file_name):
    """下载指定的 MNIST 数据文件（如果本地不存在）"""
    file_path = dataset_dir + "/" + file_name
    
    # 如果文件已存在，直接返回，避免重复下载
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    # 使用 urllib 下载文件并保存到本地
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    """下载 MNIST 数据集中所有需要的文件"""
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    """从 gzip 压缩文件中加载标签数据并转换为 NumPy 数组"""
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            # MNIST 标签文件格式：前 8 字节为文件头信息，从第 9 字节开始为标签数据
            # np.frombuffer 直接从缓冲区读取数据，offset=8 跳过文件头
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    """从 gzip 压缩文件中加载图像数据并转换为 NumPy 数组（形状为 (样本数, 784)）"""
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            # MNIST 图像文件格式：前 16 字节为文件头信息，之后为像素数据
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 将数据重排为 (样本数, 784) 的二维数组
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    """将下载的原始 MNIST 文件全部转换为 NumPy 数组，并组织成字典返回"""
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    """初始化 MNIST 数据集：下载、转换格式并保存为 pickle 文件"""
    download_mnist()
    # 将原始数据转换为 NumPy 数组字典
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    # 将数据集字典序列化保存，便于后续快速加载
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    """将标签数组转换为 one-hot 编码形式
    
    参数 X: 一维标签数组，元素为类别索引（0-9）
    返回: 形状为 (样本数, 10) 的 one-hot 矩阵，每行只有对应类别位置为 1，其余为 0
    """
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    # 如果尚未生成预处理过的 pickle 文件，则执行初始化流程
    if not os.path.exists(save_file):
        init_mnist()
        
    # 从 pickle 文件加载已处理好的数据集字典
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # 若需要进行像素值归一化，将数据类型转为 float32 并除以 255，映射到 [0, 1]
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    # 若需要将标签转为 one-hot 编码
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    # 若不需要展平图像，则将每个样本 reshape 为 (1, 28, 28) 的通道优先格式
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 返回训练集和测试集的元组
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


# 当脚本作为主程序运行时，执行初始化（下载、转换、保存）
if __name__ == '__main__':
    init_mnist()