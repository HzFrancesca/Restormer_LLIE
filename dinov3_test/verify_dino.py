#!/usr/bin/env python
"""
DINOv3 模型加载验证脚本

验证 transformers 库能否正确加载 DINOv3 模型变体:
- ViT-S/16 (facebook/dinov3-vits16-pretrain-lvd1689m)
- ViT-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m)
- ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m)
- ViT-H+/16 (facebook/dinov3-vith16plus-pretrain-lvd1689m)

Note: DINOv3 使用 16x16 patches，需要 transformers >= 4.56.0
"""

import sys
import torch


# DINOv3 模型映射 (HuggingFace 名称)
DINO_MODELS = {
    'ViT-S/16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'ViT-B/16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'ViT-L/16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'ViT-H+/16': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
}

# DINOv3 特征维度映射
DINO_DIM_MAP = {
    'ViT-S/16': 384,
    'ViT-B/16': 768,
    'ViT-L/16': 1024,
    'ViT-H+/16': 1536,
}


def verify_dino_loading(model_name: str = None, verbose: bool = True, local_path: str = None) -> bool:
    """
    验证 DINOv3 模型加载
    
    Args:
        model_name: 指定模型名称，None 则测试默认模型 (ViT-B/16)
        verbose: 是否输出详细信息
        local_path: 本地模型路径（可选）
    
    Returns:
        bool: 是否所有测试通过
    """
    try:
        from transformers import AutoModel, AutoImageProcessor
    except ImportError:
        print("✗ transformers 库未安装")
        print("  请安装: pip install transformers>=4.56.0")
        return False
    
    if verbose:
        print("=" * 60)
        print("DINOv3 模型加载验证")
        print("=" * 60)
    
    # 如果指定了本地路径，只测试该路径
    if local_path:
        models_to_test = {'Local': local_path}
    elif model_name:
        models_to_test = {model_name: DINO_MODELS[model_name]}
    else:
        # 默认只测试 ViT-B/16（避免下载所有模型）
        models_to_test = {'ViT-B/16': DINO_MODELS['ViT-B/16']}
    
    all_passed = True
    
    for variant, hf_name in models_to_test.items():
        if verbose:
            print(f"\n[{variant}] {hf_name}")
        
        try:
            import os
            is_local = os.path.isdir(hf_name)
            
            # 加载模型
            model = AutoModel.from_pretrained(hf_name, local_files_only=is_local)
            processor = AutoImageProcessor.from_pretrained(hf_name, local_files_only=is_local)
            
            # DINOv3 使用 16x16 patches，输入尺寸应为 16 的倍数
            # 标准输入: 224x224 或 518x518
            dummy_input = torch.randn(1, 3, 224, 224)
            inputs = processor(images=dummy_input, return_tensors="pt", do_rescale=False)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # DINOv3 输出: [B, 1 + 4 + num_patches, dim]
            # 1 CLS token + 4 register tokens + patch tokens
            last_hidden = outputs.last_hidden_state
            num_tokens = last_hidden.shape[1]
            hidden_dim = last_hidden.shape[2]
            
            # 计算 patch tokens 数量 (排除 CLS 和 register tokens)
            num_patches = num_tokens - 5  # 1 CLS + 4 registers
            
            if verbose:
                print(f"  ✓ 模型加载成功")
                print(f"  ✓ 输出形状: {last_hidden.shape}")
                print(f"  ✓ 特征维度: {hidden_dim}")
                print(f"  ✓ Patch tokens: {num_patches} (14x14)")
            
        except Exception as e:
            all_passed = False
            if verbose:
                print(f"  ✗ 加载失败: {e}")
                if "404" in str(e) or "not found" in str(e).lower():
                    print("  提示: DINOv3 模型可能尚未发布到 HuggingFace")
                    print("  可以使用本地模型: --local-path <path>")
    
    if verbose:
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ DINOv3 模型加载验证通过！")
        else:
            print("✗ 模型加载失败，请检查:")
            print("  1. transformers 版本 >= 4.56.0")
            print("  2. 网络连接或模型缓存")
            print("  3. 或使用本地模型路径")
        print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='验证 DINOv3 模型加载')
    parser.add_argument('--model', choices=list(DINO_MODELS.keys()),
                        help='指定测试的模型变体')
    parser.add_argument('--local-path', type=str,
                        help='本地模型路径')
    parser.add_argument('--all', action='store_true',
                        help='测试所有模型变体')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式')
    args = parser.parse_args()
    
    if args.all:
        # 测试所有模型
        all_success = True
        for model_name in DINO_MODELS.keys():
            success = verify_dino_loading(model_name, verbose=not args.quiet)
            all_success = all_success and success
        sys.exit(0 if all_success else 1)
    else:
        success = verify_dino_loading(
            args.model, 
            verbose=not args.quiet,
            local_path=args.local_path
        )
        sys.exit(0 if success else 1)
