# from transformers import AutoModel, AutoTokenizer

# def download_huggingface_model(model_id: str, save_path: str = None, trust_remote_code: bool = False):
#     """
#     从 Hugging Face Hub 下载指定的模型和分词器。

#     参数:
#     model_id (str): Hugging Face Hub 上的模型ID (例如 "bert-base-uncased" 或 "HaoyeZhang/MLLM_Excercise_Model")。
#     save_path (str, optional): 模型和分词器保存的本地路径。
#                                如果为 None, 模型将下载到 Hugging Face 的默认缓存目录。
#                                如果提供了路径，模型和分词器将被保存在此路径下。
#     trust_remote_code (bool): 是否允许执行模型仓库中的自定义代码。
#                               对于某些模型 (如 "HaoyeZhang/MLLM_Excercise_Model")，这可能是必需的。
#                               请谨慎使用，确保信任代码来源。
#     """
#     print(f"开始处理模型: {model_id}")
#     if trust_remote_code:
#         print("注意: trust_remote_code=True 已启用。这意味着将允许执行从模型仓库下载的自定义代码。")
#         print("请确保您完全信任此模型的来源和其代码的安全性。")

#     try:
#         # 尝试加载/下载模型
#         print(f"正在下载/加载模型 '{model_id}'...")
#         # 根据模型类型，您可能想使用 AutoModelForCausalLM, AutoModelForSeq2SeqLM 等，
#         # 但 AutoModel 通常足以用于下载权重。
#         model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
#         print(f"模型 '{model_id}' 下载/加载成功。")

#         # 尝试加载/下载分词器
#         print(f"正在下载/加载分词器 '{model_id}'...")
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
#             print(f"分词器 '{model_id}' 下载/加载成功。")
#         except Exception as e:
#             print(f"警告: 无法为 '{model_id}' 加载标准分词器。错误: {e}")
#             print("这可能是因为该模型没有标准分词器，或者分词器配置不同。")
#             print("如果模型是多模态的，可能需要特定的预处理器而不是标准分词器。")
#             print("对于某些模型，分词器可能不需要 trust_remote_code=True，即使模型本身需要。")
#             tokenizer = None

#         if save_path:
#             print(f"正在将模型和分词器保存到目录: {save_path}")
#             model.save_pretrained(save_path)
#             if tokenizer:
#                 tokenizer.save_pretrained(save_path)
#             print(f"模型和分词器已成功保存到: {save_path}")
#         else:
#             print(f"模型和分词器已下载到 Hugging Face 默认缓存目录。")
#             print(f"您通常可以在 ~/.cache/huggingface/hub (或类似路径) 找到它们。")

#     except Exception as e:
#         print(f"下载或加载模型 '{model_id}' 失败。错误: {e}")
#         print("请检查以下几点:")
#         print(f"  1. 模型ID '{model_id}' 是否正确无误。")
#         print("  2. 您的设备是否有稳定的网络连接。")
#         print(f"  3. 对于模型 '{model_id}'，可能需要设置 trust_remote_code=True。")
#         print(f"     当前设置: trust_remote_code={trust_remote_code}")
#         print("     如果错误信息中提及 'remote code' 或类似内容，请尝试在调用函数时将 trust_remote_code 设置为 True。")
#         print("     例如: download_huggingface_model(model_id, trust_remote_code=True)")

# if __name__ == "__main__":
#     # 要下载的模型的ID
#     model_id_to_download = "HaoyeZhang/MLLM_Excercise_Model"

#     # 对于 "HaoyeZhang/MLLM_Excercise_Model"，它是一个基于Llama的自定义模型，
#     # 因此非常可能需要 trust_remote_code=True 才能正确加载。
#     # **重要安全提示**: 启用 trust_remote_code=True 时，您正在允许执行从模型仓库下载的Python代码。
#     #                  请务必确保您信任该模型的来源 (Hugging Face 用户 "HaoyeZhang") 及其提供的代码。
#     #                  如果您不确定，请先在隔离环境中进行测试。
#     should_trust_remote_code = True

#     print(f"准备下载模型: {model_id_to_download}")
#     print(f"将使用 trust_remote_code={should_trust_remote_code} (请阅读上述安全提示)")
#     print("-" * 50)

#     # # 调用函数下载模型。默认情况下，这将下载到 Hugging Face 的缓存目录。
#     # download_huggingface_model(
#     #     model_id_to_download,
#     #     trust_remote_code=should_trust_remote_code
#     # )

#     # --- 可选：下载并保存到指定文件夹 ---
#     # 如果您想将模型文件保存到一个特定的本地文件夹，而不是Hugging Face的默认缓存，
#     # 可以取消注释下面的代码块，并设置 `custom_save_directory` 变量。

#     # custom_save_directory = "./downloaded_models/MLLM_Excercise_Model_files" # 您可以自定义此路径
#     # print(f"\n准备将模型下载并保存到自定义目录: {custom_save_directory}")
#     # print(f"将使用 trust_remote_code={should_trust_remote_code}")
#     # print("-" * 50)
#     # download_huggingface_model(
#     #     model_id_to_download,
#     #     save_path=custom_save_directory,
#     #     trust_remote_code=should_trust_remote_code
#     # )

#     print("-" * 50)
#     print("脚本执行完毕。")



import os
import requests
from tqdm import tqdm
from huggingface_hub import HfApi, snapshot_download
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib

def get_model_files(repo_id, token=None):
    """获取模型仓库的所有文件列表"""
    api = HfApi()
    files = api.list_repo_files(repo_id, token=token)
    return [f for f in files if not f.endswith(('.json', '.txt', '.md'))]  # 过滤掉小文件

def download_file(url, local_path, token=None, chunk_size=8192):
    """下载单个文件（支持断点续传）"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # 检查已下载部分
    if os.path.exists(local_path):
        existing_size = os.path.getsize(local_path)
        headers["Range"] = f"bytes={existing_size}-"
    else:
        existing_size = 0

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0)) + existing_size
        
        # 检查是否支持断点续传
        mode = 'ab' if existing_size else 'wb'
        
        with open(local_path, mode) as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=existing_size
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress.update(len(chunk))

def download_model(
    repo_id,
    save_dir="models",
    token=None,
    max_workers=4,
    revision="main",
    ignore_patterns=[]
):
    """
    下载Hugging Face大模型
    
    参数:
        repo_id: 模型ID (如 "meta-llama/Llama-2-7b-chat-hf")
        save_dir: 本地保存目录
        token: HF访问令牌（私有模型需要）
        max_workers: 并发下载线程数
        revision: 模型版本/分支
        ignore_patterns: 忽略的文件模式
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"⏳ 正在获取模型文件列表: {repo_id}")
    api = HfApi()
    
    # 使用snapshot_download获取文件URLs
    cache_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=save_dir,
        ignore_patterns=ignore_patterns,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"✅ 模型已下载到: {cache_dir}")
    return cache_dir

def verify_model(repo_id, save_dir):
    """验证下载文件的完整性（可选）"""
    print("🔍 正在验证模型文件...")
    api = HfApi()
    files = api.list_repo_files(repo_id)
    
    for file in files:
        if file.endswith(('.json', '.txt', '.md')):
            continue
            
        local_path = os.path.join(save_dir, file)
        if not os.path.exists(local_path):
            print(f"❌ 文件缺失: {file}")
            return False
            
        # 这里可以添加SHA256校验（如果HF提供）
        # ...
    
    print("✅ 所有文件验证通过")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载Hugging Face大模型')
    parser.add_argument('--repo_id', type=str, help='模型ID (如 "meta-llama/Llama-2-7b-chat-hf")')
    parser.add_argument('--save_dir', type=str, default="models", help='本地保存目录')
    parser.add_argument('--token', type=str, default=None, help='HF访问令牌')
    parser.add_argument('--revision', type=str, default="main", help='模型版本/分支')
    parser.add_argument('--max_workers', type=int, default=4, help='并发下载线程数')
    
    args = parser.parse_args()
    
    # 下载模型
    model_path = download_model(
        repo_id=args.repo_id,
        save_dir=args.save_dir,
        token=args.token,
        max_workers=args.max_workers,
        revision=args.revision
    )
    
    # 验证下载
    verify_model(args.repo_id, model_path)