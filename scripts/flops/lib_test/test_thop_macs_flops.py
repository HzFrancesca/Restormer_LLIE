"""
测试thop返回的是MACs还是FLOPs
使用简单的线性层进行验证
"""
import torch
import torch.nn as nn
from thop import profile, clever_format

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.linear(x)

# 测试参数
batch_size = 1
in_features = 10
out_features = 5

print("=" * 80)
print("thop测试: 验证返回MACs还是FLOPs")
print("=" * 80)

print("\n测试 1: 使用 bias")
print("=" * 80)

# 创建模型和输入
model_with_bias = SimpleLinear(in_features, out_features, bias=True)
input_data = torch.randn(batch_size, in_features)

print(f"\n模型配置:")
print(f"  输入维度: {in_features}")
print(f"  输出维度: {out_features}")
print(f"  是否使用bias: True")
print(f"  batch_size: {batch_size}")

# 理论计算
print(f"\n理论计算:")
print(f"  线性层 y = Wx + b")
print(f"  其中 W: ({out_features}, {in_features}), x: ({batch_size}, {in_features})")

# 计算矩阵乘法的MACs和FLOPs
matmul_macs = batch_size * out_features * in_features
matmul_flops = 2 * matmul_macs
bias_flops = batch_size * out_features
total_macs = matmul_macs
total_flops = matmul_flops + bias_flops

print(f"\n  矩阵乘法 Wx:")
print(f"    MACs  = batch × out × in = {batch_size} × {out_features} × {in_features} = {matmul_macs:,}")
print(f"    FLOPs = 2 × MACs = 2 × {matmul_macs:,} = {matmul_flops:,}")

print(f"\n  加bias (+b):")
print(f"    FLOPs = batch × out = {batch_size} × {out_features} = {bias_flops:,}")

print(f"\n  总计:")
print(f"    总MACs  = {total_macs:,}")
print(f"    总FLOPs = {total_flops:,}")

# 使用thop统计
print(f"\n{'=' * 80}")
print("thop统计结果:")
print("=" * 80)

macs, params = profile(model_with_bias, inputs=(input_data,), verbose=False)

print(f"\nthop报告的MACs: {macs:,}")
print(f"thop报告的Params: {params:,}")

# 使用clever_format格式化
macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
print(f"\n格式化后:")
print(f"  MACs: {macs_formatted}")
print(f"  Params: {params_formatted}")

# 对比分析
print(f"\n{'=' * 80}")
print("对比分析 (有bias):")
print("=" * 80)
print(f"理论MACs:  {total_macs:,}")
print(f"理论FLOPs: {total_flops:,}")
print(f"thop:      {macs:,}")

if macs == total_macs:
    print(f"\n✓ 结论: thop返回的是 MACs")
    print(f"  验证: {macs:,} == {total_macs:,}")
elif macs == total_flops:
    print(f"\n✓ 结论: thop返回的是 FLOPs")
    print(f"  验证: {macs:,} == {total_flops:,}")
elif macs == matmul_macs:
    print(f"\n✓ 结论: thop返回的是 MACs (仅矩阵乘法，不含bias)")
    print(f"  验证: {macs:,} == {matmul_macs:,}")
elif macs == matmul_flops:
    print(f"\n✓ 结论: thop返回的是 FLOPs (仅矩阵乘法，不含bias)")
    print(f"  验证: {macs:,} == {matmul_flops:,}")
else:
    print(f"\n✗ 警告: thop的值不匹配预期")

print(f"\n{'=' * 80}")
print("\n测试 2: 不使用 bias")
print("=" * 80)

# 测试2: 不使用bias
model_no_bias = SimpleLinear(in_features, out_features, bias=False)
macs_no_bias, params_no_bias = profile(model_no_bias, inputs=(input_data,), verbose=False)

print(f"\n无bias时:")
print(f"  理论MACs:  {matmul_macs:,}")
print(f"  理论FLOPs: {matmul_flops:,}")
print(f"  thop:      {macs_no_bias:,}")

if macs_no_bias == matmul_macs:
    print(f"  ✓ 匹配: thop = 理论MACs")
elif macs_no_bias == matmul_flops:
    print(f"  ✓ 匹配: thop = 理论FLOPs")
else:
    print(f"  ✗ 不匹配")

print(f"\n有bias时:")
print(f"  理论MACs:  {total_macs:,}")
print(f"  理论FLOPs: {total_flops:,}")
print(f"  thop:      {macs:,}")

if macs == total_macs + bias_flops:
    print(f"  ✓ 可能: thop = MACs + bias加法")
    print(f"         {macs} = {total_macs} + {bias_flops}")

# 最终结论
print(f"\n{'=' * 80}")
print("最终结论:")
print("=" * 80)

if macs_no_bias == matmul_macs and macs == matmul_macs:
    print("thop返回: MACs (仅矩阵乘法，不计bias)")
    print(f"  无bias: {macs_no_bias:,} = {matmul_macs:,}")
    print(f"  有bias: {macs:,} = {matmul_macs:,} (bias未计入)")
elif macs_no_bias == matmul_flops and macs == matmul_flops:
    print("thop返回: FLOPs (仅矩阵乘法，不计bias)")
    print(f"  无bias: {macs_no_bias:,} = {matmul_flops:,}")
    print(f"  有bias: {macs:,} = {matmul_flops:,} (bias未计入)")
elif macs_no_bias == matmul_macs and macs == total_macs:
    print("thop返回: MACs (包含bias)")
elif macs_no_bias == matmul_flops and macs == total_flops:
    print("thop返回: FLOPs (包含bias)")
elif macs_no_bias == matmul_macs and macs == matmul_macs + bias_flops:
    print("thop返回: MACs + bias加法操作")
    print(f"  无bias: {macs_no_bias:,} = {matmul_macs:,} (纯MACs)")
    print(f"  有bias: {macs:,} = {matmul_macs:,} + {bias_flops:,}")

print(f"\n说明:")
print("  - thop的返回值名为'macs'，但实际可能是MACs或FLOPs")
print("  - MACs: Multiply-Accumulate operations")
print("  - FLOPs: Floating Point Operations")
print("  - 关系: 1 MAC = 2 FLOPs (1个乘法 + 1个加法)")
print("=" * 80)
