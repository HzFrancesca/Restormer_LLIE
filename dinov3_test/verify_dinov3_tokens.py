"""
验证 DINOv3 输出格式和 token 结构

目标：确认 DINOv3 的输出是否为 [CLS, reg1, reg2, reg3, reg4, patch1, patch2, ...]
"""

import torch
from transformers import AutoModel, AutoConfig


def verify_dinov3_token_structure():
    """验证 DINOv3 的 token 结构"""
    
    # 使用 ViT-B/16 作为测试（较小，加载快）
    model_name = "facebook/dinov2-base"  # DINOv2 作为参考
    dinov3_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # DINOv3
    
    print("=" * 60)
    print("DINOv3 Token 结构验证")
    print("=" * 60)
    
    # 加载 DINOv3 配置和模型
    print(f"\n加载模型: {dinov3_model_name}")
    try:
        config = AutoConfig.from_pretrained(dinov3_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(dinov3_model_name, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"加载失败: {e}")
        print("尝试使用本地缓存或检查网络连接")
        return
    
    # 打印配置信息
    print("\n" + "-" * 40)
    print("模型配置:")
    print("-" * 40)
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  patch_size: {config.patch_size}")
    print(f"  image_size: {config.image_size}")
    
    # 检查 register tokens 配置
    num_registers = getattr(config, 'num_register_tokens', None)
    print(f"  num_register_tokens: {num_registers}")
    
    # 创建测试输入
    # 假设 image_size=518, patch_size=14 -> 518/14 = 37 patches per side
    # 总 patch 数 = 37 * 37 = 1369
    image_size = config.image_size
    patch_size = config.patch_size
    num_patches_per_side = image_size // patch_size
    total_patches = num_patches_per_side * num_patches_per_side
    
    print(f"\n" + "-" * 40)
    print("输入计算:")
    print("-" * 40)
    print(f"  image_size: {image_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  patches_per_side: {num_patches_per_side}")
    print(f"  total_patches: {total_patches}")
    
    # 创建随机输入图像
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    
    # 前向传播
    print(f"\n" + "-" * 40)
    print("前向传播输出:")
    print("-" * 40)
    
    with torch.no_grad():
        outputs = model(dummy_input, output_hidden_states=True)
    
    # 分析输出
    last_hidden = outputs.last_hidden_state
    print(f"  last_hidden_state shape: {last_hidden.shape}")
    print(f"    - batch_size: {last_hidden.shape[0]}")
    print(f"    - num_tokens: {last_hidden.shape[1]}")
    print(f"    - hidden_dim: {last_hidden.shape[2]}")
    
    # 计算 token 组成
    num_tokens = last_hidden.shape[1]
    expected_with_cls_only = 1 + total_patches  # CLS + patches
    expected_with_registers = 1 + (num_registers or 0) + total_patches  # CLS + registers + patches
    
    print(f"\n" + "-" * 40)
    print("Token 数量分析:")
    print("-" * 40)
    print(f"  实际 token 数: {num_tokens}")
    print(f"  预期 (仅 CLS + patches): {expected_with_cls_only}")
    print(f"  预期 (CLS + {num_registers} registers + patches): {expected_with_registers}")
    
    # 验证
    print(f"\n" + "-" * 40)
    print("验证结果:")
    print("-" * 40)
    
    if num_tokens == expected_with_registers:
        print(f"  ✅ 确认: DINOv3 输出包含 {num_registers} 个 register tokens")
        print(f"  ✅ 输出格式: [CLS, reg1, ..., reg{num_registers}, patch1, ..., patch{total_patches}]")
        print(f"\n  正确的 patch token 提取方式:")
        print(f"    patch_tokens = last_hidden[:, 1 + {num_registers}:]  # 移除 CLS 和 {num_registers} 个 register tokens")
        print(f"    patch_tokens.shape = [B, {total_patches}, {config.hidden_size}]")
    elif num_tokens == expected_with_cls_only:
        print(f"  ⚠️ 该模型不包含 register tokens")
        print(f"  输出格式: [CLS, patch1, ..., patch{total_patches}]")
        print(f"\n  正确的 patch token 提取方式:")
        print(f"    patch_tokens = last_hidden[:, 1:]  # 仅移除 CLS token")
    else:
        print(f"  ❌ 意外的 token 数量!")
        print(f"  可能的 register 数量: {num_tokens - 1 - total_patches}")
    
    # 额外验证：检查不同输入尺寸
    print(f"\n" + "-" * 40)
    print("不同输入尺寸测试:")
    print("-" * 40)
    
    test_sizes = [224, 448]
    for size in test_sizes:
        if size % patch_size != 0:
            size = (size // patch_size) * patch_size
        
        test_input = torch.randn(1, 3, size, size)
        with torch.no_grad():
            test_output = model(test_input, output_hidden_states=True)
        
        test_tokens = test_output.last_hidden_state.shape[1]
        expected_patches = (size // patch_size) ** 2
        expected_total = 1 + (num_registers or 0) + expected_patches
        
        status = "✅" if test_tokens == expected_total else "❌"
        print(f"  {status} 输入 {size}x{size}: {test_tokens} tokens (预期 {expected_total})")


if __name__ == "__main__":
    verify_dinov3_token_structure()
