# coding: utf-8
# 导入必要的系统与路径管理模块
import sys, os
# 将父目录添加到系统路径中，以便导入父目录下的模块（如 common.optimizer）
sys.path.append(os.pardir)
import numpy as np
# 从 common.optimizer 模块中导入所有优化器类（SGD、Momentum 等）
from common.optimizer import *

class Trainer:
    """神经网络训练的封装类，负责训练过程中批次迭代、损失记录与准确度评估。
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        # 保存训练所需的神经网络实例
        self.network = network
        # 是否输出训练过程的详细日志
        self.verbose = verbose
        # 训练集输入与标签
        self.x_train = x_train
        self.t_train = t_train
        # 测试集输入与标签
        self.x_test = x_test
        self.t_test = t_test
        # 训练的总轮数（遍历全部训练数据的次数）
        self.epochs = epochs
        # 小批量数据的样本数
        self.batch_size = mini_batch_size
        # 每个 epoch 中使用多少样本进行评估（None 表示使用全部数据）
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 优化器选择字典，将字符串名称映射到对应的优化器类（不区分大小写）
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        # 根据传入的优化器名称（转为小写）实例化优化器，并传入参数字典
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 训练数据总样本数
        self.train_size = x_train.shape[0]
        # 每个 epoch 中所需要的迭代次数（至少为 1），即总样本数 / 小批量大小
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 整个训练过程的最大迭代次数（总轮数 * 每轮迭代数）
        self.max_iter = int(epochs * self.iter_per_epoch)
        # 当前累计迭代次数（全局计数器）
        self.current_iter = 0
        # 当前所处的 epoch 编号
        self.current_epoch = 0
        
        # 用于记录训练过程中每步迭代后的损失值
        self.train_loss_list = []
        # 用于记录每个 epoch 结束时的训练准确度
        self.train_acc_list = []
        # 用于记录每个 epoch 结束时的测试准确度
        self.test_acc_list = []

    def train_step(self):
        """执行单步训练：随机采样小批量数据，计算梯度，更新参数，记录损失。
           同时检查是否到达 epoch 边界，若是则计算并记录训练与测试准确度。
        """
        # 从训练集中随机选择 batch_size 个索引，构成当前小批量
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # 根据当前小批量数据计算网络各参数的梯度
        grads = self.network.gradient(x_batch, t_batch)
        # 调用优化器利用梯度更新网络参数
        self.optimizer.update(self.network.params, grads)
        
        # 计算当前小批量上的损失值
        loss = self.network.loss(x_batch, t_batch)
        # 将本次迭代的损失值存入列表，用于后续观察训练过程
        self.train_loss_list.append(loss)
        # 若 verbose 为 True，则打印当前训练损失
        if self.verbose: print("train loss:" + str(loss))
        
        # 判断是否刚好完成了一个 epoch（即当前迭代次数是 iter_per_epoch 的整数倍）
        if self.current_iter % self.iter_per_epoch == 0:
            # epoch 计数加 1
            self.current_epoch += 1
            
            # 默认使用完整的训练集和测试集来评估准确度
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 如果设定了用于评估的样本数量，则仅取前 evaluate_sample_num_per_epoch 个样本
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            # 计算当前 epoch 下，选定的训练样本上的准确度
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            # 计算当前 epoch 下，选定的测试样本上的准确度
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            # 将训练与测试准确度分别存入对应列表
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            # 若 verbose 为 True，则打印当前 epoch 编号及训练、测试准确度
            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        # 累计全局迭代次数
        self.current_iter += 1

    def train(self):
        """启动完整的训练流程，迭代 max_iter 次，并在训练结束后输出最终测试准确度。"""
        # 循环进行 max_iter 次训练步骤，每次调用 train_step 完成一次迭代
        for i in range(self.max_iter):
            self.train_step()

        # 训练结束后，在整个测试集上评估最终准确度
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        # 若 verbose 为 True，则打印最终的测试准确度信息
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))