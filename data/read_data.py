import json

def read_jsonl_line(file_path, line_number):
    """
    读取 jsonl 文件中第 line_number 行（从 0 开始）的数据，并返回一个字典。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line)  # 将 JSON 字符串转换为字典
    raise IndexError(f"Line {line_number} not found in {file_path}.")

# 使用示例
if __name__ == "__main__":
    jsonl_file = "data/preference_train.json"  # 替换为你的文件路径
    target_line = 7                 # 读取第 6 行（行号从 0 开始）

    try:
        data = read_jsonl_line(jsonl_file, target_line)
        print(f"Line {target_line} content:")
        print(json.dumps(data, indent=2, ensure_ascii=False))  # 美化输出
    except Exception as e:
        print(f"Error: {e}")
