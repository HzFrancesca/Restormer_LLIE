"""
测试各个库计算的是 MACs 还是 FLOPs
使用简单的 Linear 层进行验证

理论：
- Linear(1, 1, bias=False): y = w * x
  - 1 次乘法
  - MACs = 1, FLOPs = 1
  
- Linear(1, 1, bias=True): y = w * x + b
  - 1 次乘法 + 1 次加法 = 1 MAC
  - MACs = 1, FLOPs = 2
"""

import torch
import torch.nn as nn

print("=" * 80)
print("测试各库计算的是 MACs 还是 FLOPs")
print("=" * 80)

# 创建输入
inputs = torch.randn(1, 1)

print("\n测试模型: Linear(1, 1)")
print(f"输入: {inputs.shape}")
print("\n" + "=" * 80)

# ============================================================================
# 1. 测试 torchprofile
# ============================================================================
print("\n1. torchprofile 测试")
print("-" * 80)
try:
    from torchprofile import profile_macs
    
    # 无偏置
    model_no_bias = nn.Linear(1, 1, bias=False)
    macs_no_bias = profile_macs(model_no_bias, inputs)
    print(f"Linear(1, 1, bias=False):")
    print(f"  profile_macs 返回: {macs_no_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 1")
    print(f"  结论: torchprofile 返回 {'MACs' if macs_no_bias == 1 else 'FLOPs' if macs_no_bias == 1 else '未知'}")
    
    # 有偏置
    model_with_bias = nn.Linear(1, 1, bias=True)
    macs_with_bias = profile_macs(model_with_bias, inputs)
    print(f"\nLinear(1, 1, bias=True):")
    print(f"  profile_macs 返回: {macs_with_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 2")
    print(f"  结论: torchprofile 返回 {'MACs' if macs_with_bias == 1 else 'FLOPs' if macs_with_bias == 2 else '未知'}")
    
except ImportError as e:
    print(f"[X] torchprofile 未安装: {e}")
except Exception as e:
    print(f"[X] 错误: {e}")

# ============================================================================
# 2. 测试 thop
# ============================================================================
print("\n" + "=" * 80)
print("\n2. thop 测试")
print("-" * 80)
try:
    from thop import profile
    
    # 无偏置
    model_no_bias = nn.Linear(1, 1, bias=False)
    flops_no_bias, params_no_bias = profile(model_no_bias, inputs=(inputs,), verbose=False)
    print(f"Linear(1, 1, bias=False):")
    print(f"  thop.profile 返回: {flops_no_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 1")
    print(f"  结论: thop 返回 {'MACs' if flops_no_bias == 1 else 'FLOPs' if flops_no_bias == 1 else '未知'}")
    
    # 有偏置
    model_with_bias = nn.Linear(1, 1, bias=True)
    flops_with_bias, params_with_bias = profile(model_with_bias, inputs=(inputs,), verbose=False)
    print(f"\nLinear(1, 1, bias=True):")
    print(f"  thop.profile 返回: {flops_with_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 2")
    print(f"  结论: thop 返回 {'MACs' if flops_with_bias == 1 else 'FLOPs' if flops_with_bias == 2 else '未知'}")
    
except ImportError as e:
    print(f"[X] thop 未安装: {e}")
except Exception as e:
    print(f"[X] 错误: {e}")

# ============================================================================
# 3. 测试 torchinfo
# ============================================================================
print("\n" + "=" * 80)
print("\n3. torchinfo 测试")
print("-" * 80)
try:
    from torchinfo import summary
    
    # 无偏置
    model_no_bias = nn.Linear(1, 1, bias=False)
    stats_no_bias = summary(model_no_bias, input_size=(1, 1), verbose=0)
    mult_adds_no_bias = stats_no_bias.total_mult_adds
    print(f"Linear(1, 1, bias=False):")
    print(f"  torchinfo.total_mult_adds 返回: {mult_adds_no_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 1")
    print(f"  结论: torchinfo 返回 {'MACs' if mult_adds_no_bias == 1 else 'FLOPs' if mult_adds_no_bias == 1 else '未知'}")
    
    # 有偏置
    model_with_bias = nn.Linear(1, 1, bias=True)
    stats_with_bias = summary(model_with_bias, input_size=(1, 1), verbose=0)
    mult_adds_with_bias = stats_with_bias.total_mult_adds
    print(f"\nLinear(1, 1, bias=True):")
    print(f"  torchinfo.total_mult_adds 返回: {mult_adds_with_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 2")
    print(f"  结论: torchinfo 返回 {'MACs' if mult_adds_with_bias == 1 else 'FLOPs' if mult_adds_with_bias == 2 else '未知'}")
    
except ImportError as e:
    print(f"[X] torchinfo 未安装: {e}")
except Exception as e:
    print(f"[X] 错误: {e}")

# ============================================================================
# 4. 测试 fvcore
# ============================================================================
print("\n" + "=" * 80)
print("\n4. fvcore 测试")
print("-" * 80)
try:
    from fvcore.nn import FlopCountAnalysis
    
    # 无偏置
    model_no_bias = nn.Linear(1, 1, bias=False)
    flops_analyzer_no_bias = FlopCountAnalysis(model_no_bias, inputs)
    flops_no_bias = flops_analyzer_no_bias.total()
    print(f"Linear(1, 1, bias=False):")
    print(f"  fvcore.FlopCountAnalysis 返回: {flops_no_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 1")
    print(f"  结论: fvcore 返回 {'MACs' if flops_no_bias == 1 else 'FLOPs' if flops_no_bias == 1 else '未知'}")
    
    # 有偏置
    model_with_bias = nn.Linear(1, 1, bias=True)
    flops_analyzer_with_bias = FlopCountAnalysis(model_with_bias, inputs)
    flops_with_bias = flops_analyzer_with_bias.total()
    print(f"\nLinear(1, 1, bias=True):")
    print(f"  fvcore.FlopCountAnalysis 返回: {flops_with_bias}")
    print(f"  理论 MACs: 1, 理论 FLOPs: 2")
    print(f"  结论: fvcore 返回 {'MACs' if flops_with_bias == 1 else 'FLOPs' if flops_with_bias == 2 else '未知'}")
    
except ImportError as e:
    print(f"[X] fvcore 未安装: {e}")
except Exception as e:
    print(f"[X] 错误: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("\n总结")
print("=" * 80)
print("""
根据测试结果判断：
- 如果 bias=False 返回 1，bias=True 返回 1 → 该库返回 MACs
- 如果 bias=False 返回 1，bias=True 返回 2 → 该库返回 FLOPs

理论基础：
- Linear(1,1, bias=False): y = w*x  → 1乘法 = 1 MAC = 1 FLOP
- Linear(1,1, bias=True):  y = w*x+b → 1乘法+1加法 = 1 MAC = 2 FLOPs
""")
