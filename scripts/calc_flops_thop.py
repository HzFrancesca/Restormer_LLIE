"""
使用 thop 库计算 Restormer 的参数量和 FLOPs
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from thop import profile, clever_format
from basicsr.models.archs.restormer_arch import Restormer


def calculate_flops_thop(input_size=(1, 3, 128, 128)):
    """使用 thop 计算 FLOPs 和参数量

    Args:
        input_size: 输入张量的尺寸，格式为 (batch_size, channels, height, width)
    """
    print("=" * 60)
    print("使用 thop 库计算 Restormer 的参数量和 FLOPs")
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
        LayerNorm_type="WithBias",
    )

    # 设置为评估模式
    model.eval()

    # 创建输入张量
    input_tensor = torch.randn(*input_size)

    print("\n输入尺寸: {}".format(input_tensor.shape))
    print("模型配置: dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8]")

    # 计算 MACs 和参数量
    try:
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)

        # 格式化输出
        macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
        flops_value = macs * 2
        flops_formatted = clever_format([flops_value], "%.3f")[0]

        print(f"\n--- thop 计算结果 ---")
        print(f"参数量 (Params): {params_formatted} ({params:,})")
        print(f"MACs: {macs_formatted} ({macs:,})")
        print(f"FLOPs: {flops_formatted} ({flops_value:,})")
        print("换算关系: 1 MAC = 2 FLOPs")

        # 返回原始数值用于保存
        return {
            'method': 'thop',
            'params': params,
            'params_formatted': params_formatted,
            'flops': flops_value,
            'flops_formatted': flops_formatted,
            'macs': macs,
            'macs_formatted': macs_formatted,
            'input_size': str(input_size)
        }

    except Exception as e:
        print("\n错误: {}".format(str(e)))
        return None


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="使用 thop 计算 Restormer 的参数量和 FLOPs"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=4,
        default=[1, 3, 128, 128],
        metavar=("B", "C", "H", "W"),
        help="输入尺寸 (batch_size, channels, height, width)，默认为 1 3 128 128",
    )
    args = parser.parse_args()

    # 转换为元组
    input_size = tuple(args.input_size)
    print(f"\n使用输入尺寸: {input_size}\n")

    result = calculate_flops_thop(input_size)

    if result:
        # 保存结果到文件
        output_file = os.path.join(os.path.dirname(__file__), "results_thop.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("thop 计算结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"输入尺寸: {result['input_size']}\n")
            f.write(f"参数量: {result['params_formatted']} ({result['params']:,})\n")
            f.write(f"MACs: {result['macs_formatted']} ({result['macs']:,})\n")
            f.write(f"FLOPs: {result['flops_formatted']} ({result['flops']:,})\n")
            f.write("=" * 60 + "\n")

        print("\n结果已保存到: {}".format(output_file))
    else:
        print("\n计算失败！")
        sys.exit(1)
