"""
测试fvcore返回的是MACs还是FLOPs
使用简单的线性层进行验证
"""
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

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
print("fvcore测试: 验证返回MACs还是FLOPs")
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

# 使用fvcore统计
print(f"\n{'=' * 80}")
print("fvcore统计结果:")
print("=" * 80)

flops_counter = FlopCountAnalysis(model_with_bias, input_data)
fvcore_result = flops_counter.total()

print(f"\nfvcore报告的值: {fvcore_result:,}")

# 对比分析
print(f"\n{'=' * 80}")
print("对比分析 (有bias):")
print("=" * 80)
print(f"理论MACs:  {total_macs:,}")
print(f"理论FLOPs: {total_flops:,}")
print(f"fvcore:    {fvcore_result:,}")

if fvcore_result == total_macs:
    print(f"\n✓ 结论: fvcore返回的是 MACs")
    print(f"  验证: {fvcore_result:,} == {total_macs:,}")
elif fvcore_result == total_flops:
    print(f"\n✓ 结论: fvcore返回的是 FLOPs")
    print(f"  验证: {fvcore_result:,} == {total_flops:,}")
elif fvcore_result == matmul_macs:
    print(f"\n✓ 结论: fvcore返回的是 MACs (仅矩阵乘法，不含bias)")
    print(f"  验证: {fvcore_result:,} == {matmul_macs:,}")
elif fvcore_result == matmul_flops:
    print(f"\n✓ 结论: fvcore返回的是 FLOPs (仅矩阵乘法，不含bias)")
    print(f"  验证: {fvcore_result:,} == {matmul_flops:,}")
else:
    print(f"\n✗ 警告: fvcore的值不匹配预期")

print(f"\n{'=' * 80}")
print("\n测试 2: 不使用 bias")
print("=" * 80)

# 测试2: 不使用bias
model_no_bias = SimpleLinear(in_features, out_features, bias=False)
flops_counter_no_bias = FlopCountAnalysis(model_no_bias, input_data)
fvcore_no_bias = flops_counter_no_bias.total()

print(f"\n无bias时:")
print(f"  理论MACs:  {matmul_macs:,}")
print(f"  理论FLOPs: {matmul_flops:,}")
print(f"  fvcore:    {fvcore_no_bias:,}")

if fvcore_no_bias == matmul_macs:
    print(f"  ✓ 匹配: fvcore = 理论MACs")
elif fvcore_no_bias == matmul_flops:
    print(f"  ✓ 匹配: fvcore = 理论FLOPs")
else:
    print(f"  ✗ 不匹配")

print(f"\n有bias时:")
print(f"  理论MACs:  {total_macs:,}")
print(f"  理论FLOPs: {total_flops:,}")
print(f"  fvcore:    {fvcore_result:,}")

# 最终结论
print(f"\n{'=' * 80}")
print("最终结论:")
print("=" * 80)

if fvcore_no_bias == matmul_macs and fvcore_result == matmul_macs:
    print("fvcore返回: MACs (仅矩阵乘法，不计bias)")
    print(f"  无bias: {fvcore_no_bias:,} = {matmul_macs:,}")
    print(f"  有bias: {fvcore_result:,} = {matmul_macs:,} (bias未计入)")
elif fvcore_no_bias == matmul_flops and fvcore_result == matmul_flops:
    print("fvcore返回: FLOPs (仅矩阵乘法，不计bias)")
    print(f"  无bias: {fvcore_no_bias:,} = {matmul_flops:,}")
    print(f"  有bias: {fvcore_result:,} = {matmul_flops:,} (bias未计入)")
elif fvcore_no_bias == matmul_macs and fvcore_result == total_macs:
    print("fvcore返回: MACs (包含bias)")
elif fvcore_no_bias == matmul_flops and fvcore_result == total_flops:
    print("fvcore返回: FLOPs (包含bias)")

print(f"\n说明:")
print("  - MACs: Multiply-Accumulate operations")
print("  - FLOPs: Floating Point Operations")
print("  - 关系: 1 MAC = 2 FLOPs (1个乘法 + 1个加法)")
print("=" * 80)
