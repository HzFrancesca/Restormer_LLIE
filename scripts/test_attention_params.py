"""
直接测试不同注意力模块的参数量和 MACs（独立版本）
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchprofile import profile_macs

# 直接导入注意力模块
from basicsr.models.archs.restormer_arch import Attention
from basicsr.models.archs.extra_attention import HTA, WTA, IRS, ICS


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


def test_attention_module(attn_class, name, dim=48, num_heads=1):
    """测试单个注意力模块
    
    Args:
        attn_class: 注意力模块类
        name: 注意力模块名称
        dim: 特征维度
        num_heads: 注意力头数
    """
    # 创建注意力模块
    if name == "MDTA":
        attn = attn_class(dim, num_heads, bias=False)
    else:
        attn = attn_class(dim, num_heads, bias=False)
    
    attn.eval()
    
    # 创建输入（单个特征图）
    x = torch.randn(1, dim, 128, 128)
    
    # 计算参数量
    params = count_parameters(attn)
    
    # 计算 MACs
    macs = profile_macs(attn, x)
    
    return {
        'name': name,
        'params': params,
        'params_formatted': format_number(params),
        'macs': macs,
        'macs_formatted': format_number(macs)
    }


def main():
    """测试所有注意力模块"""
    print("=" * 80)
    print("单个注意力模块参数量和 MACs 测试")
    print("=" * 80)
    print(f"测试配置: dim=48, num_heads=1, input=(1, 48, 128, 128)\n")
    
    # 定义所有注意力模块
    attention_modules = [
        (Attention, "MDTA"),
        (HTA, "HTA"),
        (WTA, "WTA"),
        (IRS, "IRS"),
        (ICS, "ICS")
    ]
    
    results = []
    
    for attn_class, name in attention_modules:
        print(f"正在测试 {name}...", end=" ")
        try:
            result = test_attention_module(attn_class, name, dim=48, num_heads=1)
            results.append(result)
            print("✓")
        except Exception as e:
            print(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("测试结果汇总（单层注意力模块）")
    print("=" * 80)
    print(f"{'模块':<10} {'参数量':<20} {'MACs':<20}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<10} {result['params_formatted']:<8} ({result['params']:>7,})  "
              f"{result['macs_formatted']:<8} ({result['macs']:>10,})")
    
    # 差异分析
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("相对于 MDTA 的差异")
        print("=" * 80)
        
        base_params = results[0]['params']
        base_macs = results[0]['macs']
        
        print(f"{'模块':<10} {'参数差异':<30} {'MACs差异':<30}")
        print("-" * 80)
        
        for result in results[1:]:
            param_diff = result['params'] - base_params
            param_percent = (param_diff / base_params) * 100 if base_params > 0 else 0
            
            macs_diff = result['macs'] - base_macs
            macs_percent = (macs_diff / base_macs) * 100 if base_macs > 0 else 0
            
            print(f"{result['name']:<10} {param_diff:+8,} ({param_percent:+6.2f}%)  "
                  f"{format_number(macs_diff):>8} ({macs_percent:+6.2f}%)")
    
    # 结构说明
    print("\n" + "=" * 80)
    print("结构说明")
    print("=" * 80)
    print("MDTA: 1个 QKV Conv1x1 + 1个 QKV DWConv3x3 + 1个 output Conv1x1")
    print("HTA:  3个独立 DWConv3x3 (Q,K,V) + 1个 MDC (V) + 1个 QKV Conv1x1 + 1个 output Conv1x1")
    print("WTA:  结构同 HTA")
    print("IRS:  结构同 HTA")
    print("ICS:  结构同 HTA")
    print("\n注: HTA/WTA/IRS/ICS 的主要区别在于注意力计算方式（矩阵乘法维度），而非参数量")


if __name__ == "__main__":
    main()
