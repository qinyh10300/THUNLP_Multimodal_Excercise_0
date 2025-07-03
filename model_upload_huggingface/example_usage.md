# Hugging Face 模型上传脚本使用说明

## 🚀 功能特点

- ✅ **无复制上传**: 直接从原始目录上传，节省磁盘空间
- ✅ **自动登录**: 支持token登录，无需手动登录
- ✅ **进度条**: 实时显示上传进度和速度
- ✅ **详细统计**: 显示文件大小、上传时间等信息

## 🔐 获取 Hugging Face Token

1. 访问: https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Write" 权限
4. 复制生成的token

## 📝 使用方法

### 方法1: 命令行参数提供token

```bash
python upload_llm_hf.py \
    --repo-name "your-model-name" \
    --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --no-copy
```

### 方法2: 环境变量

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python upload_llm_hf.py --repo-name "your-model-name" --no-copy
```

### 方法3: 传统登录方式

```bash
# 先登录
huggingface-cli login

# 然后上传
python upload_llm_hf.py --repo-name "your-model-name" --no-copy
```

## 📋 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo-name` | 仓库名称 (必须) | - |
| `--token` | HF token | None |
| `--model-path` | 模型路径 | checkpoint-90 |
| `--username` | 用户名 | 当前用户 |
| `--private` | 创建私有仓库 | True |
| `--public` | 创建公开仓库 | False |
| `--no-copy` | 无复制模式 | False |
| `--skip-check` | 跳过文件检查 | False |

## 🌟 推荐用法

```bash
# 使用token + 无复制模式 (推荐)
python upload_llm_hf.py \
    --repo-name "sft-mllm-checkpoint-90" \
    --token "YOUR_TOKEN_HERE" \
    --no-copy \
    --private
```

## 📊 输出示例

```
============================================================
🚀 Hugging Face 模型上传工具
📁 使用无复制模式 (节省磁盘空间)
============================================================
🔐 使用提供的token登录...
✅ 登录成功!
📋 当前登录用户: YourUsername
🎯 目标仓库: YourUsername/sft-mllm-checkpoint-90
🔒 私有仓库: True

📊 文件统计:
  📁 config.json: 2.1 MB
  📁 model-00001-of-00004.safetensors: 2.73 GB
  📁 model-00002-of-00004.safetensors: 4.60 GB
  📁 model-00003-of-00004.safetensors: 4.60 GB
  📁 model-00004-of-00004.safetensors: 2.31 GB
  📊 总大小: 14.26 GB
  📦 文件数量: 5

🚀 开始上传模型文件...
上传进度: 100%|████████████| 5/5 [15:32<00:00, speed=15.3MB/s, uploaded=14.26GB]

📈 上传总结:
  ✅ 成功上传: 5 个文件
  📊 总大小: 14.26 GB
  ⏱️  总时间: 15.5 分钟
  🚀 平均速度: 15.3 MB/s

✅ 模型上传完成!
🔗 模型链接: https://huggingface.co/YourUsername/sft-mllm-checkpoint-90
```

## ⚠️ 注意事项

1. **token安全**: 不要在公开场所或代码中暴露你的token
2. **网络稳定**: 大文件上传需要稳定的网络连接
3. **磁盘空间**: 无复制模式可以节省一半的磁盘空间
4. **权限**: token需要有 "Write" 权限才能上传模型 