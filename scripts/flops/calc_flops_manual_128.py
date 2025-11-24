"""
手动计算 Restormer 的参数量和 FLOPs (改进版)
输入尺寸：(1, 3, 128, 128)
包含PyTorch模型参数验证
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
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


def count_model_params(model):
    """统计PyTorch模型的实际参数量"""
    return sum(p.numel() for p in model.parameters())


def calc_conv2d_params(in_c, out_c, k, bias=False):
    """Conv2d参数量"""
    return k * k * in_c * out_c + (out_c if bias else 0)


def calc_conv2d_flops(in_c, out_c, k, h, w, groups=1, bias=False):
    """Conv2d FLOPs"""
    return 2 * k * k * (in_c // groups) * out_c * h * w + (out_c * h * w if bias else 0)


def manual_calculate_detailed():
    """详细手动计算，并与实际模型对比"""
    print("=" * 100)
    print("手动计算 Restormer 参数量和 FLOPs (改进版 - 包含验证)")
    print("=" * 100)
    
    # 配置
    inp_c, out_c = 3, 3
    dim = 48
    num_blocks = [4, 6, 6, 8]
    num_refine = 4
    heads = [1, 2, 4, 8]
    ffn_exp = 2.66
    bias = False
    has_ln_bias = True
    
    h, w = 128, 128
    print(f"\n输入尺寸: (1, {inp_c}, {h}, {w})")
    print(f"模型配置: dim={dim}, num_blocks={num_blocks}, heads={heads}")
    print(f"FFN expansion: {ffn_exp}, bias={bias}, LayerNorm with bias: {has_ln_bias}\n")
    
    # 创建实际模型进行验证
    print("创建PyTorch模型进行参数验证...")
    model = Restormer(
        inp_channels=inp_c,
        out_channels=out_c,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refine,
        heads=heads,
        ffn_expansion_factor=ffn_exp,
        bias=bias,
        LayerNorm_type="WithBias"
    )
    model.eval()
    
    actual_params = count_model_params(model)
    print(f"实际模型参数量: {format_number(actual_params)} ({actual_params:,})\n")
    
    print("=" * 100)
    print("开始逐层手动计算...")
    print("=" * 100)
    
    total_params = 0
    total_flops = 0
    
    # 详细计算每个组件
    def calc_ln_params(d):
        return d * 2 if has_ln_bias else d
    
    def calc_ln_flops(d, h, w):
        # mean, var, normalize, scale, shift
        return 10 * d * h * w
    
    def calc_attention(d, heads, h, w):
        p = 0
        f = 0
        # qkv conv
        p += d * d * 3
        f += 2 * d * d * 3 * h * w
        # qkv_dwconv (depthwise)
        p += 3 * 3 * d * 3
        f += 2 * 3 * 3 * (d * 3) * h * w
        # Q@K^T, softmax, attn@V
        N = h * w
        d_head = d // heads
        # normalize q, k along tokens (last dim of size N)
        f += 4 * heads * d_head * N * 2
        # Q @ K^T: (d_head, N) x (N, d_head) per head -> (d_head, d_head)
        f += 2 * heads * d_head * d_head * N
        # temperature scaling on (d_head, d_head)
        f += heads * d_head * d_head
        # softmax over last dim (d_head)
        f += 4 * heads * d_head * d_head
        # attn @ V: (d_head, d_head) x (d_head, N) -> (d_head, N)
        f += 2 * heads * d_head * d_head * N
        # project_out
        p += d * d
        f += 2 * d * d * h * w
        # temperature
        p += heads
        return p, f
    
    def calc_ffn(d, ffn, h, w):
        hidden = int(d * ffn)
        p = 0
        f = 0
        # project_in
        p += d * hidden * 2
        f += 2 * d * hidden * 2 * h * w
        # dwconv
        p += 3 * 3 * hidden * 2
        f += 2 * 3 * 3 * (hidden * 2) * h * w
        # GELU + multiply
        f += 9 * hidden * h * w
        # project_out
        p += hidden * d
        f += 2 * hidden * d * h * w
        return p, f
    
    def calc_transformer_block(d, heads, ffn, h, w):
        p, f = 0, 0
        # norm1
        p += calc_ln_params(d)
        f += calc_ln_flops(d, h, w)
        # attention
        ap, af = calc_attention(d, heads, h, w)
        p += ap
        f += af
        # residual
        f += d * h * w
        # norm2
        p += calc_ln_params(d)
        f += calc_ln_flops(d, h, w)
        # ffn
        fp, ff = calc_ffn(d, ffn, h, w)
        p += fp
        f += ff
        # residual
        f += d * h * w
        return p, f
    
    # 1. Patch Embedding
    print("\n1. Patch Embedding")
    p = calc_conv2d_params(inp_c, dim, 3, bias)
    f = calc_conv2d_flops(inp_c, dim, 3, h, w, 1, bias)
    print(f"   Params: {format_number(p)}, FLOPs: {format_number(f)}")
    total_params += p
    total_flops += f
    
    # Encoder
    dims = [dim, dim*2, dim*4, dim*8]
    hw = [(h, w), (h//2, w//2), (h//4, w//4), (h//8, w//8)]
    
    for i in range(4):
        d = dims[i]
        hi, wi = hw[i]
        nb = num_blocks[i]
        hd = heads[i]
        
        print(f"\n{i+2}. Encoder Level {i+1} ({nb} blocks)")
        print(f"   Dimension: {d}, Size: ({hi}, {wi}), Heads: {hd}")
        p, f = 0, 0
        for _ in range(nb):
            bp, bf = calc_transformer_block(d, hd, ffn_exp, hi, wi)
            p += bp
            f += bf
        print(f"   Params: {format_number(p)}, FLOPs: {format_number(f)}")
        total_params += p
        total_flops += f
        
        if i < 3:
            print(f"   Downsample {i+1}->{i+2}")
            dp = calc_conv2d_params(dims[i], dims[i]//2, 3, bias)
            df = calc_conv2d_flops(dims[i], dims[i]//2, 3, hw[i][0], hw[i][1], 1, bias)
            print(f"   Params: {format_number(dp)}, FLOPs: {format_number(df)}")
            total_params += dp
            total_flops += df
    
    # Decoder
    for i in range(2, -1, -1):
        d = dims[i]
        hi, wi = hw[i]
        nb = num_blocks[i]
        hd = heads[i]
        
        # Upsample
        print(f"\n{6+2-i}. Upsample Level {i+2}->{i+1}")
        up_p = calc_conv2d_params(dims[i+1], dims[i+1]*2, 3, bias)
        up_f = calc_conv2d_flops(dims[i+1], dims[i+1]*2, 3, hw[i+1][0], hw[i+1][1], 1, bias)
        print(f"   Params: {format_number(up_p)}, FLOPs: {format_number(up_f)}")
        total_params += up_p
        total_flops += up_f
        
        # Reduce channels (L3, L2, not L1)
        if i > 0:
            print(f"   Reduce channels (concat: {d}*2 -> {d})")
            red_p = calc_conv2d_params(dims[i+1], d, 1, bias)
            red_f = calc_conv2d_flops(dims[i+1], d, 1, hi, wi, 1, bias)
            print(f"   Params: {format_number(red_p)}, FLOPs: {format_number(red_f)}")
            total_params += red_p
            total_flops += red_f
        
        # Decoder blocks
        dec_d = dim*2 if i == 0 else d  # L1 uses dim*2 after concat
        dec_h = heads[0] if i == 0 else hd
        print(f"   Decoder blocks ({nb}x), Dim: {dec_d}, Heads: {dec_h}")
        p, f = 0, 0
        for _ in range(nb):
            bp, bf = calc_transformer_block(dec_d, dec_h, ffn_exp, hi, wi)
            p += bp
            f += bf
        print(f"   Params: {format_number(p)}, FLOPs: {format_number(f)}")
        total_params += p
        total_flops += f
    
    # Refinement
    print(f"\n9. Refinement ({num_refine} blocks)")
    print(f"   Dimension: {dim*2}, Size: ({h}, {w}), Heads: {heads[0]}")
    p, f = 0, 0
    for _ in range(num_refine):
        bp, bf = calc_transformer_block(dim*2, heads[0], ffn_exp, h, w)
        p += bp
        f += bf
    print(f"   Params: {format_number(p)}, FLOPs: {format_number(f)}")
    total_params += p
    total_flops += f
    
    # Output
    print(f"\n10. Output Conv")
    out_p = calc_conv2d_params(dim*2, out_c, 3, bias)
    out_f = calc_conv2d_flops(dim*2, out_c, 3, h, w, 1, bias)
    out_f += out_c * h * w  # residual add with input
    print(f"   Params: {format_number(out_p)}, FLOPs: {format_number(out_f)}")
    total_params += out_p
    total_flops += out_f
    
    # 总结
    print("\n" + "=" * 100)
    print("手动计算结果:")
    print(f"  参数量: {format_number(total_params)} ({total_params:,})")
    print(f"  FLOPs: {format_number(total_flops)} ({total_flops:,})")
    print(f"  MACs: {format_number(total_flops/2)} ({total_flops//2:,})")
    print("\n实际模型参数量:")
    print(f"  参数量: {format_number(actual_params)} ({actual_params:,})")
    print("\n参数量差异:")
    diff = abs(total_params - actual_params)
    diff_pct = (diff / actual_params) * 100
    print(f"  绝对差异: {format_number(diff)} ({diff:,})")
    print(f"  相对差异: {diff_pct:.2f}%")
    print("=" * 100)
    
    return {
        'manual_params': total_params,
        'actual_params': actual_params,
        'flops': total_flops,
        'macs': total_flops / 2,
        'input_size': '(1, 3, 128, 128)'
    }


if __name__ == "__main__":
    result = manual_calculate_detailed()
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), "results_manual_v2.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("手动计算结果 (改进版 - 包含验证)\n")
        f.write("=" * 100 + "\n")
        f.write(f"输入尺寸: {result['input_size']}\n")
        f.write(f"手动计算参数量: {format_number(result['manual_params'])} ({result['manual_params']:,})\n")
        f.write(f"实际模型参数量: {format_number(result['actual_params'])} ({result['actual_params']:,})\n")
        f.write(f"参数量差异: {abs(result['manual_params'] - result['actual_params']):,}\n")
        f.write(f"FLOPs: {format_number(result['flops'])} ({result['flops']:,})\n")
        f.write(f"MACs: {format_number(result['macs'])} ({int(result['macs']):,})\n")
        f.write("=" * 100 + "\n")
    
    print(f"\n结果已保存到: {output_file}")
