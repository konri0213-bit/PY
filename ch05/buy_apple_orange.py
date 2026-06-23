# coding: utf-8
# 导入自定义的简单层模块（包含 MulLayer 乘法层、AddLayer 加法层等）
from layer_naive import *

# ---------------- 输入数据定义 ----------------
# 定义商品单价与数量，以及消费税率
apple = 100          # 苹果的单价（元/个）
apple_num = 2        # 购买的苹果数量
orange = 150         # 橘子的单价（元/个）
orange_num = 3       # 购买的橘子数量
tax = 1.1            # 消费税倍率（含税价格为不含税价格的 1.1 倍）

# ---------------- 层的实例化 ----------------
# 实例化乘法层与加法层，用于构建计算图
mul_apple_layer = MulLayer()              # 苹果总价 = 苹果单价 * 数量
mul_orange_layer = MulLayer()             # 橘子总价 = 橘子单价 * 数量
add_apple_orange_layer = AddLayer()       # 水果总价 = 苹果总价 + 橘子总价
mul_tax_layer = MulLayer()                # 最终支付价格 = 水果总价 * 税倍率

# ---------------- 前向传播 ----------------
# 按照计算图顺序依次向前传递，计算出最终价格
apple_price = mul_apple_layer.forward(apple, apple_num)        # (1) 苹果小计 = 单价 * 数量
orange_price = mul_orange_layer.forward(orange, orange_num)    # (2) 橘子小计 = 单价 * 数量
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3) 水果合计 = 苹果小计 + 橘子小计
price = mul_tax_layer.forward(all_price, tax)                  # (4) 含税总价 = 水果合计 * 税率

# ---------------- 反向传播 ----------------
# 从最终输出开始，反向依次回传梯度（链式法则）
dprice = 1                                                      # 最终输出价格对自身的梯度初始化为 1
dall_price, dtax = mul_tax_layer.backward(dprice)              # (4) 回传至 all_price 和 tax 的梯度
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3) 回传至 apple_price 和 orange_price 的梯度
dorange, dorange_num = mul_orange_layer.backward(dorange_price) # (2) 回传至 orange 和 orange_num 的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)    # (1) 回传至 apple 和 apple_num 的梯度

# ---------------- 输出结果 ----------------
# 打印前向计算结果以及各原始输入变量对最终价格的敏感度（梯度）
print("price:", int(price))          # 打印总价（取整）
print("dApple:", dapple)             # apple 单价的梯度
print("dApple_num:", int(dapple_num))# apple 数量的梯度（取整便于观察）
print("dOrange:", dorange)           # orange 单价的梯度
print("dOrange_num:", int(dorange_num))  # orange 数量的梯度（取整）
print("dTax:", dtax)                 # tax 的梯度