# coding: utf-8
# 导入自定义的乘法层（包含 MulLayer 类，用于封装乘法的前向与反向传播逻辑）
from layer_naive import *

# 原始输入变量：苹果单价（元）
apple = 100
# 原始输入变量：苹果数量（个）
apple_num = 2
# 原始输入变量：税率倍率（如 1.1 表示加 10% 税）
tax = 1.1

# 实例化两个乘法层，分别用于计算“苹果总价”和“含税总价”
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# ---------- 前向传播 ----------
# 计算苹果总价：单价 × 数量，前向传播得到 apple_price
apple_price = mul_apple_layer.forward(apple, apple_num)
# 计算最终含税价格：苹果总价 × 税率倍率
price = mul_tax_layer.forward(apple_price, tax)

# ---------- 反向传播 ----------
# 假设最终价格的梯度为 1（损失函数对 price 的梯度）
dprice = 1
# 税率乘法层的反向传播，根据 dprice 获得对 apple_price 和 tax 的梯度
dapple_price, dtax = mul_tax_layer.backward(dprice)
# 苹果乘法层的反向传播，根据 dapple_price 获得对 apple 和 apple_num 的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 打印前向计算结果（含税总价取整）以及各输入变量的梯度
print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)