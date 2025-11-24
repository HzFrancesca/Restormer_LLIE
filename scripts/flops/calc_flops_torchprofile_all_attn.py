"""
使用 torchprofile 库计算不同注意力模块的 Restormer 参数量和 MACs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import argparse
from torchprofile import profile_macs
from basicsr.models.archs.restormer_arch import Restormer


def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.3f}G"
    elif num >= 1e6:
        return f"{num/1e6:.3f}M"
    elif num >= 1e3:
        return f"{num/1e3:.3f}K"
    else:
        return f"{num:.3f}"


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def calculate_flops_for_attention(attn_type, input_size=(1, 3, 128, 128)):
    """使用 torchprofile 计算特定注意力模块的 FLOPs 和参数量
    
    Args:
        attn_type: 注意力类型 (MDTA, HTA, WTA, IRS, ICS)
        input_size: 输入张量的尺寸，格式为 (batch_size, channels, height, width)
    """
    # 创建模型实例
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        attn_types=[attn_type, attn_type, attn_type, attn_type]  # 所有层使用相同注意力
    )
    
    # 设置为评估模式
    model.eval()
    
    # 创建输入张量
    input_tensor = torch.randn(*input_size)
    
    try:
        # 计算参数量
        params = count_parameters(model)
        
        # 计算 MACs (使用 torchprofile)
        macs = profile_macs(model, input_tensor)
        
        # FLOPs 通常是 MACs 的两倍（一次乘法和一次加法）
        flops = macs * 2
        
        # 格式化输出
        params_formatted = format_number(params)
        macs_formatted = format_number(macs)
        flops_formatted = format_number(flops)
        
        return {
            'attn_type': attn_type,
            'params': params,
            'params_formatted': params_formatted,
            'macs': macs,
            'macs_formatted': macs_formatted,
            'flops': flops,
            'flops_formatted': flops_formatted,
        }
        
    except Exception as e:
        print(f"\n错误 ({attn_type}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main(input_size=(1, 3, 128, 128)):
    """测试所有注意力模块"""
    print("=" * 80)
    print("使用 torchprofile 库计算不同注意力模块的参数量和 MACs")
    print("=" * 80)
    print(f"\n输入尺寸: {input_size}")
    print(f"模型配置: dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8]\n")
    
    # 测试所有注意力类型
    attn_types = ['MDTA', 'HTA', 'WTA', 'IRS', 'ICS']
    results = []
    
    for attn_type in attn_types:
        print(f"\n正在测试 {attn_type}...", end=" ")
        result = calculate_flops_for_attention(attn_type, input_size)
        if result:
            results.append(result)
            print("✓")
        else:
            print("✗")
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"{'注意力类型':<12} {'参数量 (Params)':<20} {'MACs':<20}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['attn_type']:<12} {result['params_formatted']:<20} {result['macs_formatted']:<20}")
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), "results_all_attention_torchprofile.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("不同注意力模块的参数量和 MACs 对比\n")
        f.write("=" * 80 + "\n")
        f.write(f"输入尺寸: {input_size}\n")
        f.write(f"模型配置: dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8]\n\n")
        
        f.write(f"{'注意力类型':<12} {'参数量':<25} {'MACs':<25}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['attn_type']:<12} {result['params_formatted']:<10} ({result['params']:>12,}) "
                   f"{result['macs_formatted']:<10} ({result['macs']:>12,})\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 分析差异
    print("\n" + "=" * 80)
    print("参数量差异分析")
    print("=" * 80)
    if len(results) > 1:
        base_params = results[0]['params']  # MDTA 作为基准
        for result in results[1:]:
            diff = result['params'] - base_params
            diff_percent = (diff / base_params) * 100
            print(f"{result['attn_type']} vs MDTA: {diff:+,} ({diff_percent:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("MACs 差异分析")
    print("=" * 80)
    if len(results) > 1:
        base_macs = results[0]['macs']  # MDTA 作为基准
        for result in results[1:]:
            diff = result['macs'] - base_macs
            diff_percent = (diff / base_macs) * 100
            print(f"{result['attn_type']} vs MDTA: {diff:+,} ({diff_percent:+.2f}%)")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用 torchprofile 计算不同注意力模块的参数量和 MACs')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 128, 128],
                        metavar=('B', 'C', 'H', 'W'),
                        help='输入尺寸 (batch_size, channels, height, width)，默认为 1 3 128 128')
    args = parser.parse_args()
    
    # 转换为元组
    input_size = tuple(args.input_size)
    
    main(input_size)
