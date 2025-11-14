"""
简化版权重文件比较工具
无需额外依赖，只使用 PyTorch
"""

import torch
import argparse
import sys


def load_checkpoint(path):
    """加载权重文件"""
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"\n[成功] 加载: {path}")
        
        # 检查是否有 'params' 键
        if isinstance(checkpoint, dict):
            if 'params' in checkpoint:
                print("  -> 使用 checkpoint['params']")
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                print("  -> 使用 checkpoint['params_ema']")
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                print("  -> 使用 checkpoint['state_dict']")
                state_dict = checkpoint['state_dict']
            else:
                print("  -> 直接使用整个 checkpoint")
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        return state_dict
    except Exception as e:
        print(f"\n[错误] 加载失败: {path}")
        print(f"  错误信息: {str(e)}")
        return None


def print_state_dict_info(state_dict, name="模型"):
    """打印 state_dict 的详细信息"""
    print(f"\n{'='*100}")
    print(f"{name} 权重信息")
    print(f"{'='*100}")
    
    print(f"\n总键数量: {len(state_dict)}")
    
    # 统计参数
    param_count = 0
    total_params = 0
    
    print(f"\n{'序号':<6} {'键名':<70} {'形状':<25} {'参数量':<15}")
    print("-" * 116)
    
    for idx, (key, value) in enumerate(state_dict.items(), 1):
        if isinstance(value, torch.Tensor):
            shape_str = str(tuple(value.shape))
            size = value.numel()
            total_params += size
            param_count += 1
            
            print(f"{idx:<6} {key:<70} {shape_str:<25} {size:>12,}")
        else:
            print(f"{idx:<6} {key:<70} {'[非张量]':<25} {'-':<15}")
    
    print("-" * 116)
    print(f"\n总参数数量: {total_params:,}")
    print(f"张量数量: {param_count}")


def compare_state_dicts(state_dict1, state_dict2, name1="模型1", name2="模型2"):
    """比较两个 state_dict"""
    print(f"\n{'='*100}")
    print(f"比较 {name1} vs {name2}")
    print(f"{'='*100}")
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    # 1. 键名比较
    print(f"\n{'='*50}")
    print("【1. 键名比较】")
    print(f"{'='*50}")
    print(f"{name1} 键数量: {len(keys1)}")
    print(f"{name2} 键数量: {len(keys2)}")
    print(f"共同键数量: {len(common_keys)}")
    
    if only_in_1:
        print(f"\n仅在 {name1} 中存在的键 ({len(only_in_1)} 个):")
        for key in sorted(only_in_1):
            shape_str = ""
            if isinstance(state_dict1[key], torch.Tensor):
                shape_str = f" [shape: {tuple(state_dict1[key].shape)}]"
            print(f"  - {key}{shape_str}")
    
    if only_in_2:
        print(f"\n仅在 {name2} 中存在的键 ({len(only_in_2)} 个):")
        for key in sorted(only_in_2):
            shape_str = ""
            if isinstance(state_dict2[key], torch.Tensor):
                shape_str = f" [shape: {tuple(state_dict2[key].shape)}]"
            print(f"  - {key}{shape_str}")
    
    # 2. 形状比较
    print(f"\n{'='*50}")
    print("【2. 形状比较】")
    print(f"{'='*50}")
    
    shape_mismatch = []
    shape_match = []
    
    for key in sorted(common_keys):
        val1 = state_dict1[key]
        val2 = state_dict2[key]
        
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if val1.shape != val2.shape:
                shape_mismatch.append((key, val1.shape, val2.shape))
            else:
                shape_match.append(key)
    
    if shape_mismatch:
        print(f"\n形状不匹配的键 ({len(shape_mismatch)} 个):")
        for key, shape1, shape2 in shape_mismatch:
            print(f"  - {key}")
            print(f"    {name1}: {tuple(shape1)}")
            print(f"    {name2}: {tuple(shape2)}")
    else:
        print(f"[OK] 所有共同键的形状都匹配!")
    
    print(f"\n形状匹配的键: {len(shape_match)} 个")
    
    # 3. 数值比较
    print(f"\n{'='*50}")
    print("【3. 数值比较】")
    print(f"{'='*50}")
    
    identical_count = 0
    similar_count = 0
    different_count = 0
    
    value_comparison = []
    
    for key in sorted(shape_match):
        val1 = state_dict1[key]
        val2 = state_dict2[key]
        
        # 检查是否完全相同
        if torch.equal(val1, val2):
            identical_count += 1
        else:
            # 计算差异
            diff = torch.abs(val1 - val2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            relative_diff = (diff / (torch.abs(val1) + 1e-8)).mean().item()
            
            if max_diff < 1e-6:
                similar_count += 1
                status = f"极小差异 (max: {max_diff:.2e})"
            else:
                different_count += 1
                status = f"有差异 (max: {max_diff:.4f}, mean: {mean_diff:.4f}, rel: {relative_diff:.4f})"
            
            value_comparison.append((key, max_diff, mean_diff, relative_diff, status))
    
    print(f"\n完全相同的键: {identical_count} 个")
    print(f"极小差异的键: {similar_count} 个")
    print(f"明显差异的键: {different_count} 个")
    
    if value_comparison:
        print(f"\n数值差异详情 (显示差异最大的前20个):")
        # 按最大差异排序
        value_comparison.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'键名':<70} {'状态':<50}")
        print("-" * 120)
        for key, max_diff, mean_diff, rel_diff, status in value_comparison[:20]:
            print(f"{key:<70} {status}")
    
    # 4. 总结
    print(f"\n{'='*100}")
    print("【比较总结】")
    print(f"{'='*100}")
    
    structure_same = True
    
    # 检查键名
    if len(only_in_1) == 0 and len(only_in_2) == 0:
        print("[OK] 键名完全相同")
    else:
        print(f"[×] 键名不同 (总差异: {len(only_in_1) + len(only_in_2)} 个)")
        structure_same = False
    
    # 检查形状
    if len(shape_mismatch) == 0 and len(common_keys) > 0:
        print("[OK] 所有共同键的形状都匹配")
    elif len(shape_mismatch) > 0:
        print(f"[×] 形状不匹配 ({len(shape_mismatch)} 个)")
        structure_same = False
    
    # 检查数值
    if identical_count == len(shape_match) and len(shape_match) > 0:
        print("[OK] 所有数值完全相同")
    elif different_count == 0:
        print(f"[~] 数值基本相同 (有 {similar_count} 个极小差异)")
    else:
        print(f"[~] 数值有差异 ({different_count} 个有明显差异)")
    
    print()
    print("="*100)
    # 最终结论
    if structure_same:
        print("[结论] 除了数值大小可能不同，两个权重文件的结构完全相同！")
        print("       (键名相同、形状相同，可以使用 strict=False 互相加载)")
    else:
        print("[结论] 两个权重文件的结构不同！")
        if only_in_1 or only_in_2:
            print("       原因: 键名不同")
        if shape_mismatch:
            print("       原因: 形状不匹配")
    print("="*100)
    
    return structure_same


def main():
    parser = argparse.ArgumentParser(
        description="比较两个 PyTorch 权重文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 比较两个权重文件
  python compare_weights_simple.py mdta_model.pth hta_model.pth
  
  # 指定模型名称
  python compare_weights_simple.py mdta.pth hta.pth --name1 MDTA --name2 HTA
  
  # 只查看单个文件
  python compare_weights_simple.py model.pth
        """
    )
    
    parser.add_argument('file1', type=str, help='第一个权重文件路径')
    parser.add_argument('file2', type=str, nargs='?', default=None, 
                        help='第二个权重文件路径 (可选)')
    parser.add_argument('--name1', type=str, default="模型1",
                        help='第一个模型的名称 (默认: 模型1)')
    parser.add_argument('--name2', type=str, default="模型2",
                        help='第二个模型的名称 (默认: 模型2)')
    
    args = parser.parse_args()
    
    # 加载第一个文件
    print("正在加载权重文件...")
    state_dict1 = load_checkpoint(args.file1)
    if state_dict1 is None:
        sys.exit(1)
    
    # 打印第一个文件的信息
    print_state_dict_info(state_dict1, args.name1)
    
    # 如果提供了第二个文件，进行比较
    if args.file2:
        state_dict2 = load_checkpoint(args.file2)
        if state_dict2 is None:
            sys.exit(1)
        
        # 打印第二个文件的信息
        print_state_dict_info(state_dict2, args.name2)
        
        # 比较两个文件
        compare_state_dicts(state_dict1, state_dict2, args.name1, args.name2)
    else:
        print("\n[提示] 如果要比较两个文件，请提供第二个文件路径作为参数")
        print(f"       例如: python {sys.argv[0]} {args.file1} <第二个文件.pth>")


if __name__ == "__main__":
    main()

