# coding: utf-8
# 导入所需的基本逻辑门函数：与门、或门、与非门
from and_gate import AND
from or_gate import OR
from nand_gate import NAND


def XOR(x1, x2):
    """
    异或门（XOR）实现：仅当两个输入信号相异时输出 1。
    使用 NAND、OR、AND 三个基本门级联构建：
    - 先计算两个输入的与非（NAND）
    - 再计算两个输入的或（OR）
    - 最后将上述两个结果做与（AND）运算，得到异或输出
    
    参数:
    x1: 第一个输入信号 (0 或 1)
    x2: 第二个输入信号 (0 或 1)
    
    返回:
    y: 异或运算结果 (0 或 1)
    """
    # 第一层：计算 x1 和 x2 的与非结果
    s1 = NAND(x1, x2)
    # 第一层：计算 x1 和 x2 的或结果
    s2 = OR(x1, x2)
    # 第二层：将 s1 与 s2 进行与运算，得到最终的异或结果
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    # 遍历所有可能的输入组合，验证 XOR 的真值表
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        # 调用 XOR 函数计算当前输入组合的异或值
        y = XOR(xs[0], xs[1])
        # 打印输入组合及其对应的输出结果
        print(str(xs) + " -> " + str(y))