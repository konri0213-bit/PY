# coding: utf-8
import numpy as np

class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.01):
        """
        初始化 SGD 优化器

        参数：
            lr : float, 学习率（默认 0.01）
        """
        self.lr = lr
        
    def update(self, params, grads):
        """
        使用随机梯度下降更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组（如权重、偏置）
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        for key in params.keys():
            # 参数沿着负梯度方向移动：θ = θ - lr * ∇L
            params[key] -= self.lr * grads[key] 


class Momentum:
    """带动量的随机梯度下降法（Momentum SGD）"""
    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化 Momentum 优化器

        参数：
            lr       : float, 学习率（默认 0.01）
            momentum : float, 动量系数（默认 0.9），控制历史梯度影响程度
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 动量速度字典，将在首次调用 update 时按参数名初始化
        
    def update(self, params, grads):
        """
        使用带动量的梯度下降更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        # 首次调用时，为每个参数创建与参数相同形状的零数组作为初始速度 v
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            # 更新速度：v = momentum * v - lr * ∇L（摩擦力 + 当前梯度加速）
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            # 参数更新：θ = θ + v
            params[key] += self.v[key]


class Nesterov:
    """Nesterov 加速梯度法（Nesterov's Accelerated Gradient）"""
    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化 Nesterov 优化器

        参数：
            lr       : float, 学习率（默认 0.01）
            momentum : float, 动量系数（默认 0.9）
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 速度字典，首次调用时初始化
        
    def update(self, params, grads):
        """
        使用 Nesterov 加速梯度更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        # 首次调用时初始化每个参数对应的速度 v 为零
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 保留旧的速度值供后续使用
            # 速度更新：v = momentum * v - lr * ∇L
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            # 参数更新采用 Nesterov 前瞻修正，结合当前速度与梯度
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    """自适应梯度算法（AdaGrad）"""
    def __init__(self, lr=0.01):
        """
        初始化 AdaGrad 优化器

        参数：
            lr : float, 学习率（默认 0.01）
        """
        self.lr = lr
        self.h = None  # 累积梯度平方和字典，首次调用时初始化
        
    def update(self, params, grads):
        """
        使用 AdaGrad 自适应学习率更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        # 首次调用时，为每个参数创建形状相同的零数组作为累积平方梯度 h
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 累积历史梯度平方和：h += ∇L ⊙ ∇L
            self.h[key] += grads[key] * grads[key]
            # 参数更新：θ = θ - lr * ∇L / (sqrt(h) + ε)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    """RMSprop 算法"""
    def __init__(self, lr=0.01, decay_rate = 0.99):
        """
        初始化 RMSprop 优化器

        参数：
            lr         : float, 学习率（默认 0.01）
            decay_rate : float, 指数衰减率（默认 0.99），用于控制历史平方梯度的衰减速度
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None  # 指数移动平均的梯度平方字典，首次调用时初始化
        
    def update(self, params, grads):
        """
        使用 RMSprop 自适应学习率更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        # 首次调用时，为每个参数创建形状相同的零数组作为 h
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 指数移动平均更新梯度平方：h = decay_rate * h + (1 - decay_rate) * ∇L^2
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            # 参数更新：θ = θ - lr * ∇L / (sqrt(h) + ε)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """Adam 算法（自适应矩估计）"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        初始化 Adam 优化器

        参数：
            lr    : float, 学习率（默认 0.001）
            beta1 : float, 一阶矩指数衰减率（默认 0.9）
            beta2 : float, 二阶矩指数衰减率（默认 0.999）
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0      # 迭代步数计数器
        self.m = None      # 一阶矩（动量）字典
        self.v = None      # 二阶矩（速度/未中心化的方差）字典
        
    def update(self, params, grads):
        """
        使用 Adam 自适应矩估计更新参数

        参数：
            params : dict，键为参数名，值为对应的参数数组
            grads  : dict，键为参数名，值为对应的梯度数组
        """
        # 首次调用时初始化一阶矩 m 和二阶矩 v 为零数组
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        # 步数加1
        self.iter += 1
        # 计算当前步的偏差修正学习率
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # 更新一阶矩的偏差修正形式：m = m + (1-beta1) * (∇L - m)
            # 等同于原公式：m_hat = beta1 * m + (1-beta1) * ∇L，这里用增量形式避免记录旧值
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # 更新二阶矩的偏差修正形式：v = v + (1-beta2) * (∇L^2 - v)
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 参数更新：θ = θ - lr_t * m / (sqrt(v) + ε)
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            # 以下为注释掉的另一种偏差修正写法（等效但形式不同），保留供参考
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)