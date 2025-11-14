"""
比较两个 PyTorch 权重文件的结构
用于分析不同模型权重文件的差异
"""

import torch
import argparse
from collections import OrderedDict
from colorama import init, Fore, Style

# 初始化 colorama（用于彩色输出）
try:
    init(autoreset=True)
    USE_COLOR = True
except:
    USE_COLOR = False


def print_colored(text, color=None):
    """打印彩色文本"""
    if USE_COLOR and color:
        print(color + text + Style.RESET_ALL)
    else:
        print(text)


def load_checkpoint(path):
    """加载权重文件"""
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print_colored(f"\n✓ 成功加载: {path}", Fore.GREEN)
        
        # 检查是否有 'params' 键
        if isinstance(checkpoint, dict):
            if 'params' in checkpoint:
                print_colored("  检测到 'params' 键，使用 checkpoint['params']", Fore.CYAN)
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                print_colored("  检测到 'params_ema' 键，使用 checkpoint['params_ema']", Fore.CYAN)
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                print_colored("  检测到 'state_dict' 键，使用 checkpoint['state_dict']", Fore.CYAN)
                state_dict = checkpoint['state_dict']
            else:
                # 尝试直接使用整个 checkpoint
                print_colored("  直接使用整个 checkpoint 作为 state_dict", Fore.CYAN)
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        return state_dict, checkpoint
    except Exception as e:
        print_colored(f"\n✗ 加载失败: {path}", Fore.RED)
        print_colored(f"  错误信息: {str(e)}", Fore.RED)
        return None, None


def print_state_dict_info(state_dict, name="模型"):
    """打印 state_dict 的详细信息"""
    print_colored(f"\n{'='*80}", Fore.YELLOW)
    print_colored(f"{name} 权重信息", Fore.YELLOW)
    print_colored(f"{'='*80}", Fore.YELLOW)
    
    print_colored(f"\n总键数量: {len(state_dict)}", Fore.CYAN)
    
    # 统计不同类型的参数
    param_count = 0
    total_params = 0
    
    print_colored("\n键名列表及形状:", Fore.CYAN)
    print("-" * 80)
    
    for idx, (key, value) in enumerate(state_dict.items(), 1):
        if isinstance(value, torch.Tensor):
            shape_str = str(tuple(value.shape))
            dtype_str = str(value.dtype).replace('torch.', '')
            size = value.numel()
            total_params += size
            param_count += 1
            
            print(f"{idx:3d}. {key:60s} {shape_str:25s} {dtype_str:10s} ({size:,} params)")
        else:
            print(f"{idx:3d}. {key:60s} [非张量类型: {type(value).__name__}]")
    
    print("-" * 80)
    print_colored(f"总参数数量: {total_params:,}", Fore.GREEN)
    print_colored(f"张量数量: {param_count}", Fore.GREEN)


def compare_state_dicts(state_dict1, state_dict2, name1="模型1", name2="模型2"):
    """比较两个 state_dict"""
    print_colored(f"\n{'='*80}", Fore.YELLOW)
    print_colored(f"比较 {name1} vs {name2}", Fore.YELLOW)
    print_colored(f"{'='*80}", Fore.YELLOW)
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    # 1. 键名比较
    print_colored(f"\n【键名比较】", Fore.CYAN)
    print(f"  {name1} 键数量: {len(keys1)}")
    print(f"  {name2} 键数量: {len(keys2)}")
    print(f"  共同键数量: {len(common_keys)}")
    
    if only_in_1:
        print_colored(f"\n  仅在 {name1} 中存在的键 ({len(only_in_1)} 个):", Fore.YELLOW)
        for key in sorted(only_in_1):
            shape_str = ""
            if isinstance(state_dict1[key], torch.Tensor):
                shape_str = f" - shape: {tuple(state_dict1[key].shape)}"
            print(f"    - {key}{shape_str}")
    
    if only_in_2:
        print_colored(f"\n  仅在 {name2} 中存在的键 ({len(only_in_2)} 个):", Fore.YELLOW)
        for key in sorted(only_in_2):
            shape_str = ""
            if isinstance(state_dict2[key], torch.Tensor):
                shape_str = f" - shape: {tuple(state_dict2[key].shape)}"
            print(f"    - {key}{shape_str}")
    
    # 2. 形状比较
    print_colored(f"\n【形状比较】", Fore.CYAN)
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
        print_colored(f"  形状不匹配的键 ({len(shape_mismatch)} 个):", Fore.RED)
        for key, shape1, shape2 in shape_mismatch:
            print(f"    - {key}")
            print(f"      {name1}: {tuple(shape1)}")
            print(f"      {name2}: {tuple(shape2)}")
    else:
        print_colored(f"  ✓ 所有共同键的形状都匹配!", Fore.GREEN)
    
    print_colored(f"  形状匹配的键数量: {len(shape_match)}", Fore.GREEN)
    
    # 3. 数值比较
    print_colored(f"\n【数值比较】", Fore.CYAN)
    
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
            status = "完全相同"
            color = Fore.GREEN
        else:
            # 计算差异
            diff = torch.abs(val1 - val2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            relative_diff = (diff / (torch.abs(val1) + 1e-8)).mean().item()
            
            if max_diff < 1e-6:
                similar_count += 1
                status = f"极小差异 (max: {max_diff:.2e})"
                color = Fore.CYAN
            else:
                different_count += 1
                status = f"有差异 (max: {max_diff:.4f}, mean: {mean_diff:.4f}, rel: {relative_diff:.4f})"
                color = Fore.YELLOW
            
            value_comparison.append((key, max_diff, mean_diff, relative_diff, status, color))
    
    print(f"  完全相同: {identical_count} 个")
    print(f"  极小差异: {similar_count} 个")
    print(f"  有明显差异: {different_count} 个")
    
    if value_comparison:
        print_colored(f"\n  数值差异详情:", Fore.CYAN)
        # 按最大差异排序
        value_comparison.sort(key=lambda x: x[1], reverse=True)
        
        for key, max_diff, mean_diff, rel_diff, status, color in value_comparison[:20]:  # 只显示前20个
            print_colored(f"    - {key:60s} {status}", color)
    
    # 4. 总结
    print_colored(f"\n{'='*80}", Fore.YELLOW)
    print_colored(f"【比较总结】", Fore.YELLOW)
    print_colored(f"{'='*80}", Fore.YELLOW)
    
    if len(only_in_1) == 0 and len(only_in_2) == 0:
        print_colored("✓ 键名完全相同", Fore.GREEN)
    else:
        print_colored(f"✗ 键名不同 (差异: {len(only_in_1) + len(only_in_2)} 个)", Fore.RED)
    
    if len(shape_mismatch) == 0 and len(common_keys) > 0:
        print_colored("✓ 所有共同键的形状都匹配", Fore.GREEN)
    elif len(shape_mismatch) > 0:
        print_colored(f"✗ 形状不匹配 ({len(shape_mismatch)} 个)", Fore.RED)
    
    if identical_count == len(shape_match) and len(shape_match) > 0:
        print_colored("✓ 所有数值完全相同", Fore.GREEN)
    elif different_count == 0:
        print_colored(f"≈ 数值基本相同 (有 {similar_count} 个极小差异)", Fore.CYAN)
    else:
        print_colored(f"≈ 数值有差异 ({different_count} 个有明显差异)", Fore.YELLOW)
    
    # 判断是否除了数值大小，其余都相同
    if len(only_in_1) == 0 and len(only_in_2) == 0 and len(shape_mismatch) == 0:
        print_colored(f"\n✓ 结论: 除了数值大小可能不同，两个权重文件的结构完全相同！", Fore.GREEN)
        return True
    else:
        print_colored(f"\n✗ 结论: 两个权重文件的结构不同！", Fore.RED)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="比较两个 PyTorch 权重文件的结构和数值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 比较两个权重文件
  python compare_weights.py --file1 mdta_model.pth --file2 hta_model.pth
  
  # 只查看单个文件的信息
  python compare_weights.py --file1 model.pth
        """
    )
    
    parser.add_argument('--file1', type=str, required=True,
                        help='第一个权重文件路径 (.pth)')
    parser.add_argument('--file2', type=str, default=None,
                        help='第二个权重文件路径 (.pth, 可选)')
    parser.add_argument('--name1', type=str, default="模型1",
                        help='第一个模型的名称')
    parser.add_argument('--name2', type=str, default="模型2",
                        help='第二个模型的名称')
    parser.add_argument('--no-color', action='store_true',
                        help='禁用彩色输出')
    
    args = parser.parse_args()
    
    if args.no_color:
        global USE_COLOR
        USE_COLOR = False
    
    # 加载第一个文件
    state_dict1, checkpoint1 = load_checkpoint(args.file1)
    if state_dict1 is None:
        return
    
    # 打印第一个文件的信息
    print_state_dict_info(state_dict1, args.name1)
    
    # 如果提供了第二个文件，进行比较
    if args.file2:
        state_dict2, checkpoint2 = load_checkpoint(args.file2)
        if state_dict2 is None:
            return
        
        # 打印第二个文件的信息
        print_state_dict_info(state_dict2, args.name2)
        
        # 比较两个文件
        compare_state_dicts(state_dict1, state_dict2, args.name1, args.name2)
    else:
        print_colored("\n提示: 如果要比较两个文件，请使用 --file2 参数指定第二个文件", Fore.CYAN)


if __name__ == "__main__":
    main()

