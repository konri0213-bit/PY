# coding: utf-8
# 定义一个名为 Man 的类，用来表示一个具有姓名并能打招呼、告别的人
class Man:
    """示例类"""   # 示例类

    # 类的初始化方法，在创建实例时自动调用，接收姓名参数并输出初始化信息
    def __init__(self, name):
        # 将传入的姓名绑定到实例属性 name 上
        self.name = name
        # 打印初始化完成的提示信息
        print("Initilized!")

    # 打招呼的方法，输出包含实例姓名的问候语
    def hello(self):
        # 组装并打印问候语：Hello 姓名!
        print("Hello " + self.name + "!")

    # 告别的方法，输出包含实例姓名的告别语
    def goodbye(self):
        # 组装并打印告别语：Good-bye 姓名!
        print("Good-bye " + self.name + "!")

# 创建一个 Man 类的实例，姓名为 "David"，此时会自动执行 __init__ 方法
m = Man("David")
# 调用实例的 hello 方法，打印问候语
m.hello()
# 调用实例的 goodbye 方法，打印告别语
m.goodbye()