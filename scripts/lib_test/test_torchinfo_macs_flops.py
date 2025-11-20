"""
测试torchinfo返回的是MACs还是FLOPs
使用简单的线性层进行验证
"""
import torch
import torch.nn as nn
from torchinfo import summary

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
use_bias = True

print("=" * 80)
print("测试 1: 使用 bias")
print("=" * 80)

# 创建模型
model = SimpleLinear(in_features, out_features, bias=use_bias)

# 创建输入
input_data = torch.randn(batch_size, in_features)

print("=" * 80)
print("线性层测试: 验证torchinfo返回MACs还是FLOPs")
print("=" * 80)
print(f"\n模型配置:")
print(f"  输入维度: {in_features}")
print(f"  输出维度: {out_features}")
print(f"  是否使用bias: {use_bias}")
print(f"  batch_size: {batch_size}")

# 理论计算
print(f"\n理论计算:")
print(f"  线性层 y = Wx + b")
print(f"  其中 W: ({out_features}, {in_features}), x: ({batch_size}, {in_features})")

# 计算矩阵乘法的MACs和FLOPs
# 对于矩阵乘法 (batch, in_features) @ (in_features, out_features)^T
# 每个输出元素需要 in_features 次乘加操作
matmul_macs = batch_size * out_features * in_features
matmul_flops = 2 * matmul_macs  # 每个MAC = 1个乘法 + 1个加法 = 2个FLOPs

print(f"\n  矩阵乘法 Wx:")
print(f"    MACs  = batch × out × in = {batch_size} × {out_features} × {in_features} = {matmul_macs:,}")
print(f"    FLOPs = 2 × MACs = 2 × {matmul_macs:,} = {matmul_flops:,}")

if use_bias:
    # 加bias需要额外的加法操作
    bias_flops = batch_size * out_features
    print(f"\n  加bias (+b):")
    print(f"    FLOPs = batch × out = {batch_size} × {out_features} = {bias_flops:,}")
    
    total_macs = matmul_macs
    total_flops = matmul_flops + bias_flops
else:
    total_macs = matmul_macs
    total_flops = matmul_flops

print(f"\n  总计:")
print(f"    总MACs  = {total_macs:,}")
print(f"    总FLOPs = {total_flops:,}")

# 使用torchinfo统计
print(f"\n{'=' * 80}")
print("torchinfo统计结果:")
print("=" * 80)
stats = summary(
    model, 
    input_size=(batch_size, in_features),
    verbose=0,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"]
)

# 获取torchinfo的结果
torchinfo_macs = stats.total_mult_adds

print(f"\ntorchinfo报告的 mult_adds (MACs): {torchinfo_macs:,}")

# 对比分析
print(f"\n{'=' * 80}")
print("对比分析:")
print("=" * 80)
print(f"理论MACs:  {total_macs:,}")
print(f"理论FLOPs: {total_flops:,}")
print(f"torchinfo: {torchinfo_macs:,}")

if torchinfo_macs == total_macs:
    print(f"\n✓ 结论: torchinfo返回的是 MACs (Multiply-Accumulate operations)")
    print(f"  验证: {torchinfo_macs:,} == {total_macs:,}")
elif torchinfo_macs == total_flops:
    print(f"\n✓ 结论: torchinfo返回的是 FLOPs (Floating Point Operations)")
    print(f"  验证: {torchinfo_macs:,} == {total_flops:,}")
else:
    print(f"\n✗ 警告: torchinfo的值既不等于MACs也不等于FLOPs")
    print(f"  可能的原因: 计算方法不同或统计口径差异")

print(f"\n{'=' * 80}")
print("=" * 80)
print("\n测试 2: 不使用 bias")
print("=" * 80)

# 测试2: 不使用bias
model_no_bias = SimpleLinear(in_features, out_features, bias=False)
stats_no_bias = summary(
    model_no_bias, 
    input_size=(batch_size, in_features),
    verbose=0,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"]
)

torchinfo_no_bias = stats_no_bias.total_mult_adds
print(f"\n无bias时:")
print(f"  理论MACs:  {matmul_macs:,}")
print(f"  torchinfo: {torchinfo_no_bias:,}")

if torchinfo_no_bias == matmul_macs:
    print(f"  ✓ 匹配: torchinfo = 理论MACs")
else:
    print(f"  ✗ 不匹配")

print(f"\n有bias时:")
print(f"  理论MACs:  {matmul_macs:,}")
print(f"  bias加法:  {bias_flops:,}")
print(f"  torchinfo: {torchinfo_macs:,}")

if torchinfo_macs == matmul_macs + bias_flops:
    print(f"  ✓ 匹配: torchinfo = MACs + bias加法")
    print(f"         {torchinfo_macs} = {matmul_macs} + {bias_flops}")

print(f"\n{'=' * 80}")
print("说明:")
print("=" * 80)
print("1. MACs (Multiply-Accumulate): 一个MAC = 一次乘法 + 一次加法")
print("2. FLOPs (Floating Point Ops): 每个浮点运算都单独计数")
print("3. 关系: 1 MAC = 2 FLOPs (在大多数定义中)")
print("4. torchinfo的 mult_adds 字段对应的是 MACs")
print("\n结论:")
print("=" * 80)
print("torchinfo的mult_adds计算方式:")
print("  - 矩阵乘法: 计为MACs (乘加操作)")
print("  - bias加法: 也计入mult_adds (虽然只是加法)")
print("  - 因此: mult_adds ≈ MACs + 所有加法操作")
print("=" * 80)
