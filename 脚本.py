import os  # 导入操作系统接口模块，用于文件和目录操作
from openai import OpenAI  # 导入 OpenAI 官方库，用于调用兼容接口的大语言模型

# ==================== 配置区域 ====================
# 1. 填入你的 API Key 和 Base URL
# 如果你用的是 DeepSeek，把下面的地址换成 DeepSeek 的官方 API 地址
API_KEY = ""  # 你的 API 密钥，用于身份验证
BASE_URL = "https://api.deepseek.com"  # API 基础地址，若使用其他厂商需替换
MODEL_NAME = "deepseek-v4-pro"  # 模型名称，根据所选服务可按需修改

# 2. 填写你要处理的 Python 项目文件夹的绝对路径
PROJECT_PATH = r"xxxxxxxx"  # 目标项目根目录，所有 .py 文件都将被处理
# ==================================================

# 创建 OpenAI 客户端实例，后续所有 API 请求都将通过该客户端发出
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_chinese_comments(code_content):
    """调用 AI 为代码添加汉语注释，严格保持代码逻辑不变
    
    参数:
        code_content (str): 原始 Python 源代码字符串
        
    返回:
        str 或 None: 添加完注释后的纯代码字符串，失败时返回 None
    """
    # 构造提示词，明确角色、任务要求以及需要处理的代码
    prompt = (
        "你是一个专业的 Python 算法工程师。请帮我为以下 Python 代码添加详细、优雅的汉语注释（包括函数作用、关键逻辑说明、复杂变量含义等）。\n"
        "⚠️ 注意：\n"
        "1. 严格保持原代码的逻辑、变量名、函数名、缩进和语法完全不变，绝对不要重构或修改代码功能。\n"
        "2. 只返回添加完注释后的纯代码，不要包含任何 Markdown 格式的包裹符号（如 ```python 等），不要有任何前言和后记。\n\n"
        f"【待处理代码】:\n{code_content}"
    )

    try:
        # 调用聊天补全接口，传入提示词并指定模型
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 低温度保证输出稳定，减少随机性，避免修改代码
        )
        # 提取助手返回的内容，并去除首尾空白字符
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 捕获网络错误或 API 返回异常，打印失败信息并返回 None
        print(f"API 调用失败: {e}")
        return None


def process_project(target_dir):
    """遍历指定文件夹，并对其中的所有 .py 文件添加汉语注释
    
    参数:
        target_dir (str): 待处理的 Python 项目根目录的绝对路径
    """
    # 使用 os.walk 递归遍历目录树，root 为当前目录，dirs 为子目录列表，files 为文件列表
    for root, dirs, files in os.walk(target_dir):
        # 排除常见无关目录：虚拟环境、版本控制、缓存、IDE 配置等，避免误处理第三方库或临时文件
        if any(ignored in root for ignored in ['.venv', 'venv', '.git', '__pycache__', '.idea']):
            continue  # 跳过当前目录，不处理其中的任何文件

        for file in files:
            # 只处理 .py 文件，并且跳过脚本自身，防止自修改导致逻辑错乱
            if file.endswith('.py') and file != 'add_comments.py':  # 跳过脚本自身
                file_path = os.path.join(root, file)  # 拼接得到文件的完整路径
                print(f"正在处理: {file_path} ...")

                # 以 UTF-8 编码读取原始代码内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_code = f.read()

                # 跳过空文件，避免无意义的 API 调用
                if not original_code.strip():
                    continue

                # 调用函数获取带注释的代码
                commented_code = get_chinese_comments(original_code)

                if commented_code:
                    # 建议：这里默认是【直接覆盖原文件】。
                    # 如果你害怕代码出错，可以改成：file_path + ".commented.py" 另存为新文件
                    # 以写入模式重新打开文件，将带注释的代码写回原文件（覆盖）
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(commented_code)
                    print(f"【成功】已为 {file} 更新汉语注释。")
                else:
                    print(f"【跳过】{file} 处理失败。")


if __name__ == "__main__":
    # 简单的前置检查：如果 API_KEY 未填写或项目路径不存在，则提示用户配置后重试
    if "你的_API_KEY" in API_KEY or not os.path.exists(PROJECT_PATH):
        print("请先在代码上方配置好您的 API_KEY 和正确的 PROJECT_PATH 路径！")
    else:
        print("🚀 开始批量添加中文注释...")
        process_project(PROJECT_PATH)  # 执行批量处理
        print("✨ 所有文件处理完毕！")