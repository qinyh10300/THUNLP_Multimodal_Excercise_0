#!/usr/bin/env python3
"""
上传MLLM模型到Hugging Face Hub的脚本
上传路径: /home/user0/rlq/THUNLP_Multimodal_Excercise/output/checkpoint-90
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional
import json
import time

try:
    from huggingface_hub import HfApi, Repository, login, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("错误: 请先安装 huggingface_hub")
    print("运行: pip install huggingface_hub")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("警告: 未安装 tqdm，将显示简化进度信息")
    tqdm = None

def login_to_hf(token: Optional[str] = None) -> bool:
    """
    登录到Hugging Face Hub
    
    Args:
        token: HF token，如果为None则尝试从环境变量获取
    
    Returns:
        bool: 登录是否成功
    """
    try:
        # 尝试获取token
        if token is None:
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        
        if token:
            print(f"🔐 使用提供的token登录...")
            login(token=token)
            print(f"✅ 登录成功!")
            return True
        else:
            print(f"⚠️ 未提供token，尝试使用已保存的凭据...")
            # 尝试使用已保存的凭据
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ 已登录用户: {user_info['name']}")
            return True
            
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        print(f"💡 请提供有效的Hugging Face token:")
        print(f"   方法1: 使用 --token 参数")
        print(f"   方法2: 设置环境变量 HF_TOKEN")
        print(f"   方法3: 运行 huggingface-cli login")
        print(f"🔗 获取token: https://huggingface.co/settings/tokens")
        return False

def check_model_directory(model_path: str) -> bool:
    """
    检查模型目录是否包含必要的文件
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查必要的文件
    required_files = [
        "config.json"
    ]
    
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "preprocessor_config.json"
    ]
    
    missing_required = []
    existing_files = []
    
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            existing_files.append(file)
        else:
            missing_required.append(file)
    
    # 检查是否有权重文件
    weight_patterns = ["*.bin", "*.safetensors", "*.pt", "*.pth"]
    weight_files = []
    for pattern in weight_patterns:
        weight_files.extend(list(model_path.glob(pattern)))
    
    print(f"模型目录检查结果:")
    print(f"  路径: {model_path}")
    print(f"  存在的文件: {existing_files}")
    print(f"  权重文件: {[f.name for f in weight_files]}")
    
    if missing_required:
        print(f"  缺少必要文件: {missing_required}")
        
    # 如果有config.json和权重文件，认为是有效的
    if (model_path / "config.json").exists() and weight_files:
        return True
    else:
        print("错误: 模型目录缺少关键文件")
        return False

def create_readme_content() -> str:
    """
    创建README.md内容
    """
    return """---
license: apache-2.0
tags:
- multimodal
- vision-language
- fine-tuned
- pytorch
- safetensors
pipeline_tag: image-text-to-text
library_name: transformers
---

# Fine-tuned MLLM Model (Checkpoint-90)

这是一个经过微调的多模态大语言模型，基于checkpoint-90训练得到。

## 模型信息

- **检查点**: checkpoint-90
- **微调数据**: 自定义数据集
- **用途**: 视觉问答任务
- **准确率**: 67.30% (在测试数据集上)

## 使用方法

```python
# 模型加载代码示例
from transformers import AutoTokenizer, AutoModel

# 注意：可能需要根据具体模型类型调整加载方式
tokenizer = AutoTokenizer.from_pretrained("your-username/your-repo-name")
model = AutoModel.from_pretrained("your-username/your-repo-name")
```

## 训练详情

- 基于checkpoint-90进行微调
- 针对视觉问答任务优化
- 支持多模态输入（图像+文本）
- 在测试数据上准确率: 67.30%

## 文件说明

- `config.json`: 模型配置文件
- `model-*.safetensors`: 模型权重文件（SafeTensors格式）
- `tokenizer*`: 分词器相关文件
- `generation_config.json`: 生成配置

## 许可证

Apache 2.0
"""

def upload_to_hf_direct(
    model_path: str,
    repo_name: str,
    username: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload fine-tuned MLLM checkpoint-90"
):
    """
    直接上传模型到Hugging Face Hub (无复制)
    """
    api = HfApi()
    
    # 获取用户信息（此时应该已经登录）
    try:
        user_info = api.whoami()
        current_user = user_info['name']
        print(f"📋 当前登录用户: {current_user}")
    except Exception as e:
        print(f"❌ 无法获取用户信息: {e}")
        return False
    
    # 设置仓库全名
    if username:
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = f"{current_user}/{repo_name}"
    
    print(f"🎯 目标仓库: {full_repo_name}")
    print(f"🔒 私有仓库: {private}")
    
    try:
        # 检查仓库是否存在，如果不存在则创建
        try:
            repo_info = api.repo_info(full_repo_name)
            print(f"📦 仓库已存在: {repo_info.id}")
        except RepositoryNotFoundError:
            print(f"🆕 创建新仓库: {full_repo_name}")
            create_repo(
                repo_id=full_repo_name,
                private=private,
                repo_type="model"
            )
        
        # 先上传README.md
        print(f"\n📄 上传 README.md...")
        api.upload_file(
            path_or_fileobj=create_readme_content().encode('utf-8'),
            path_in_repo="README.md",
            repo_id=full_repo_name,
            commit_message=f"{commit_message} - README.md"
        )
        print(f"    ✅ README.md 上传成功")
        
        # 统计所有文件
        model_path = Path(model_path)
        all_files = [f for f in model_path.iterdir() if f.is_file()]
        
        print(f"\n📊 文件统计:")
        total_size_bytes = 0
        for file_path in all_files:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 * 1024 * 1024)
            total_size_bytes += size_bytes
            
            if size_gb >= 1:
                print(f"  📁 {file_path.name}: {size_gb:.2f} GB")
            else:
                print(f"  📁 {file_path.name}: {size_mb:.1f} MB")
        
        total_gb = total_size_bytes / (1024 * 1024 * 1024)
        print(f"  📊 总大小: {total_gb:.2f} GB")
        print(f"  📦 文件数量: {len(all_files)}")
        
        # 开始上传模型文件
        print(f"\n🚀 开始上传模型文件...")
        uploaded_files = []
        failed_files = []
        
        # 如果有tqdm，使用进度条，否则显示简单进度
        if tqdm:
            progress_bar = tqdm(all_files, desc="上传进度", unit="file")
        else:
            progress_bar = all_files
        
        start_time = time.time()
        uploaded_size = 0
        
        for i, file_path in enumerate(progress_bar):
            file_size_bytes = file_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
            
            # 显示当前文件信息
            if file_size_gb >= 1:
                size_str = f"{file_size_gb:.2f} GB"
            else:
                size_str = f"{file_size_mb:.1f} MB"
            
            if tqdm:
                progress_bar.set_description(f"上传 {file_path.name} ({size_str})")
            else:
                print(f"  [{i+1}/{len(all_files)}] 上传: {file_path.name} ({size_str})")
            
            file_start_time = time.time()
            
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=full_repo_name,
                    commit_message=f"{commit_message} - {file_path.name}"
                )
                
                uploaded_files.append(file_path.name)
                uploaded_size += file_size_bytes
                
                # 计算上传速度
                file_time = time.time() - file_start_time
                if file_time > 0:
                    speed_mbps = (file_size_mb / file_time)
                    if tqdm:
                        progress_bar.set_postfix({
                            'speed': f'{speed_mbps:.1f} MB/s',
                            'uploaded': f'{uploaded_size/(1024*1024*1024):.2f}GB'
                        })
                    else:
                        print(f"    ✅ {file_path.name} 上传成功 (速度: {speed_mbps:.1f} MB/s)")
                else:
                    if not tqdm:
                        print(f"    ✅ {file_path.name} 上传成功")
                
            except Exception as e:
                failed_files.append((file_path.name, str(e)))
                if tqdm:
                    progress_bar.write(f"    ❌ {file_path.name} 上传失败: {e}")
                else:
                    print(f"    ❌ {file_path.name} 上传失败: {e}")
        
        if tqdm:
            progress_bar.close()
        
        # 显示上传总结
        total_time = time.time() - start_time
        avg_speed = (uploaded_size / (1024 * 1024)) / total_time if total_time > 0 else 0
        
        print(f"\n📈 上传总结:")
        print(f"  ✅ 成功上传: {len(uploaded_files)} 个文件")
        if failed_files:
            print(f"  ❌ 失败文件: {len(failed_files)} 个")
            for filename, error in failed_files:
                print(f"    • {filename}: {error}")
        print(f"  📊 总大小: {uploaded_size/(1024*1024*1024):.2f} GB")
        print(f"  ⏱️  总时间: {total_time/60:.1f} 分钟")
        print(f"  🚀 平均速度: {avg_speed:.1f} MB/s")
        
        if len(uploaded_files) > 0:
            print(f"\n✅ 模型上传完成!")
            print(f"🔗 模型链接: https://huggingface.co/{full_repo_name}")
            return True
        else:
            print(f"\n❌ 没有文件上传成功")
            return False
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def prepare_model_for_upload(model_path: str, temp_dir: str) -> str:
    """
    准备模型用于上传，创建临时目录并复制文件
    """
    model_path = Path(model_path)
    temp_path = Path(temp_dir)
    
    # 创建临时目录
    temp_path.mkdir(exist_ok=True)
    
    print(f"正在准备模型文件...")
    
    # 复制所有文件
    for file_path in model_path.iterdir():
        if file_path.is_file():
            dest_path = temp_path / file_path.name
            print(f"  复制: {file_path.name}")
            shutil.copy2(file_path, dest_path)
    
    # 创建或更新README.md
    readme_path = temp_path / "README.md"
    readme_content = create_readme_content()
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"模型准备完成: {temp_path}")
    return str(temp_path)

def upload_to_hf(
    model_path: str,
    repo_name: str,
    username: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload fine-tuned MLLM checkpoint-90"
):
    """
    上传模型到Hugging Face Hub (带复制)
    """
    api = HfApi()
    
    # 获取用户信息
    try:
        user_info = api.whoami()
        current_user = user_info['name']
        print(f"当前登录用户: {current_user}")
    except Exception as e:
        print(f"错误: 无法获取用户信息，请先登录")
        print(f"运行: huggingface-cli login")
        return False
    
    # 设置仓库全名
    if username:
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = f"{current_user}/{repo_name}"
    
    print(f"目标仓库: {full_repo_name}")
    print(f"私有仓库: {private}")
    
    try:
        # 检查仓库是否存在，如果不存在则创建
        try:
            repo_info = api.repo_info(full_repo_name)
            print(f"仓库已存在: {repo_info.id}")
        except RepositoryNotFoundError:
            print(f"创建新仓库: {full_repo_name}")
            create_repo(
                repo_id=full_repo_name,
                private=private,
                repo_type="model"
            )
        
        # 上传文件
        print(f"开始上传模型文件...")
        
        model_path = Path(model_path)
        for file_path in model_path.iterdir():
            if file_path.is_file():
                print(f"  上传: {file_path.name}")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=full_repo_name,
                    commit_message=f"{commit_message} - {file_path.name}"
                )
        
        print(f"✅ 模型上传成功!")
        print(f"🔗 模型链接: https://huggingface.co/{full_repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='上传MLLM模型到Hugging Face Hub')
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--repo-name', 
        type=str, 
        required=True,
        help='Hugging Face仓库名称'
    )
    parser.add_argument(
        '--username', 
        type=str, 
        default=None,
        help='Hugging Face用户名 (可选，默认使用当前登录用户)'
    )
    parser.add_argument(
        '--private', 
        action='store_true',
        default=False,
        help='创建私有仓库 (默认: True)'
    )
    parser.add_argument(
        '--public', 
        action='store_true',
        help='创建公开仓库'
    )
    parser.add_argument(
        '--commit-message', 
        type=str, 
        default='Upload fine-tuned MLLM checkpoint-90',
        help='提交信息'
    )
    parser.add_argument(
        '--temp-dir', 
        type=str, 
        default='/tmp/hf_upload_temp',
        help='临时目录路径'
    )
    parser.add_argument(
        '--skip-check', 
        action='store_true',
        help='跳过模型文件检查'
    )
    parser.add_argument(
        '--no-copy', 
        action='store_true',
        help='直接上传，不复制文件到临时目录 (推荐，节省磁盘空间)'
    )
    parser.add_argument(
        '--token', 
        type=str, 
        default=None,
        help='Hugging Face token (可选，如果为None则尝试从环境变量获取)'
    )
    
    args = parser.parse_args()
    
    # 处理public参数
    if args.public:
        args.private = False
    
    print("=" * 60)
    print("🚀 Hugging Face 模型上传工具")
    if args.no_copy:
        print("📁 使用无复制模式 (节省磁盘空间)")
    print("=" * 60)
    
    # 检查模型目录
    if not args.skip_check:
        if not check_model_directory(args.model_path):
            print("❌ 模型检查失败，使用 --skip-check 跳过检查")
            return
    
    # 登录到Hugging Face
    if not login_to_hf(args.token):
        return
    
    # 上传模型
    if args.no_copy:
        # 无复制直接上传
        success = upload_to_hf_direct(
            model_path=args.model_path,
            repo_name=args.repo_name,
            username=args.username,
            private=args.private,
            commit_message=args.commit_message
        )
    else:
        # 传统方式（先复制再上传）
        try:
            prepared_path = prepare_model_for_upload(args.model_path, args.temp_dir)
        except Exception as e:
            print(f"❌ 模型准备失败: {e}")
            return
        
        try:
            success = upload_to_hf(
                model_path=prepared_path,
                repo_name=args.repo_name,
                username=args.username,
                private=args.private,
                commit_message=args.commit_message
            )
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(args.temp_dir)
                print(f"🧹 已清理临时文件: {args.temp_dir}")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
    
    if success:
        print("\n🎉 上传完成!")
    else:
        print("\n❌ 上传失败")

if __name__ == '__main__':
    main() 