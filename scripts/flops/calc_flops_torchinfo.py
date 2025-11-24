"""
使用 torchinfo 库计算 Restormer 的参数量和 FLOPs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from torchinfo import summary
from basicsr.models.archs.restormer_arch import Restormer


def calculate_flops_torchinfo(input_size=(1, 3, 128, 128)):
    """使用 torchinfo 计算 FLOPs 和参数量
    
    Args:
        input_size: 输入张量的尺寸，格式为 (batch_size, channels, height, width)
    """
    print("=" * 60)
    print("使用 torchinfo 库计算 Restormer 的参数量和 FLOPs")
    print("=" * 60)
    
    # 创建模型实例（标准 Restormer 配置）
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias"
    )
    
    # 设置为评估模式
    model.eval()
    
    print("\n输入尺寸: {}".format(input_size))
    print("模型配置: dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8]")
    
    try:
        # 使用 torchinfo 生成详细信息
        model_stats = summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=0,  # 不打印详细层信息
            depth=3,
            device='cpu'
        )
        
        # 提取关键信息
        total_params = model_stats.total_params
        trainable_params = model_stats.trainable_params
        total_mult_adds = model_stats.total_mult_adds
        
        # 格式化输出
        def format_number(num):
            if num >= 1e9:
                return f"{num/1e9:.3f}G"
            elif num >= 1e6:
                return f"{num/1e6:.3f}M"
            elif num >= 1e3:
                return f"{num/1e3:.3f}K"
            else:
                return f"{num:.3f}"
        
        params_formatted = format_number(total_params)
        macs_formatted = format_number(total_mult_adds)  # total_mult_adds 实际是 MACs
        flops_value = total_mult_adds * 2  # FLOPs = MACs * 2
        flops_formatted = format_number(flops_value)
        
        print("\n--- torchinfo 计算结果 ---")
        print(f"总参数量: {params_formatted} ({total_params:,})")
        print(f"可训练参数量: {format_number(trainable_params)} ({trainable_params:,})")
        print(f"MACs: {macs_formatted} ({total_mult_adds:,})")
        print(f"FLOPs: {flops_formatted} ({flops_value:,})")
        print("换算关系: 1 MAC = 2 FLOPs")
        
        # 打印模型摘要
        print("\n--- 模型摘要 ---")
        print(model_stats)
        
        # 返回原始数值用于保存
        return {
            'method': 'torchinfo',
            'params': total_params,
            'params_formatted': params_formatted,
            'trainable_params': trainable_params,
            'macs': total_mult_adds,
            'macs_formatted': macs_formatted,
            'flops': flops_value,
            'flops_formatted': flops_formatted,
            'input_size': str(input_size)
        }
        
    except Exception as e:
        print("\n错误: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用 torchinfo 计算 Restormer 的参数量和 FLOPs')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 128, 128],
                        metavar=('B', 'C', 'H', 'W'),
                        help='输入尺寸 (batch_size, channels, height, width)，默认为 1 3 128 128')
    args = parser.parse_args()
    
    # 转换为元组
    input_size = tuple(args.input_size)
    print(f"\n使用输入尺寸: {input_size}\n")
    
    result = calculate_flops_torchinfo(input_size)
    
    if result:
        # 保存结果到文件
        output_file = os.path.join(os.path.dirname(__file__), "results_torchinfo.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("torchinfo 计算结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"输入尺寸: {result['input_size']}\n")
            f.write(f"总参数量: {result['params_formatted']} ({result['params']:,})\n")
            f.write(f"可训练参数量: {result['trainable_params']:,}\n")
            f.write(f"MACs: {result['macs_formatted']} ({result['macs']:,})\n")
            f.write(f"FLOPs: {result['flops_formatted']} ({result['flops']:,})\n")
            f.write("=" * 60 + "\n")
        
        print("\n结果已保存到: {}".format(output_file))
    else:
        print("\n计算失败！")
        sys.exit(1)
