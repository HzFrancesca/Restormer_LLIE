"""检查 DINOv3 模型的模块名称，用于确定 LoRA target_modules"""
import os
import torch
from pathlib import Path
from transformers import AutoModel

# 加载模型（使用 Path 确保路径格式正确）
model_path = Path("D:/Downloads/AI_models/modelscope/models/facebook/dinov3-vith16plus-pretrain-lvd1689m")

# 验证路径存在
if not model_path.exists():
    raise FileNotFoundError(f"模型路径不存在: {model_path}")

# 使用字符串形式的绝对路径
model = AutoModel.from_pretrained(str(model_path.resolve()), trust_remote_code=True, local_files_only=True)

print("=" * 60)
print("DINOv3 模型的所有模块名称:")
print("=" * 60)

# 打印所有包含 attention 相关的模块
for name, module in model.named_modules():
    if any(key in name.lower() for key in ['attn', 'qkv', 'proj', 'query', 'key', 'value']):
        print(f"{name}: {type(module).__name__}")

print("\n" + "=" * 60)
print("所有 Linear 层:")
print("=" * 60)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"{name}: {module.in_features} -> {module.out_features}")
