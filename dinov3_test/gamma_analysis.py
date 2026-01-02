"""
Gamma 校正分析 - 纯数值版本（无图像加载）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("生成 Gamma 校正分析图...")
    
    # === 图1: Gamma 曲线 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：曲线
    ax1 = axes[0]
    x = np.linspace(0.001, 1, 256)
    gamma_values = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for gamma, color in zip(gamma_values, colors):
        y = np.power(x, gamma)
        label = f'γ={gamma}'
        if gamma == 1.0:
            label += ' (no change)'
        elif gamma == 0.4:
            label += ' (recommended)'
        ax1.plot(x, y, label=label, linewidth=2.5, color=color)
    
    ax1.set_xlabel('Input pixel value', fontsize=12)
    ax1.set_ylabel('Output pixel value', fontsize=12)
    ax1.set_title('Gamma Correction: I_out = I_in^γ', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 标注暗区放大效果
    ax1.annotate('Dark region\n(significant boost)', 
                xy=(0.1, 0.5), xytext=(0.3, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
    
    # 右图：放大倍数
    ax2 = axes[1]
    input_vals = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    bar_width = 0.1
    x_pos = np.arange(len(input_vals))
    
    for i, gamma in enumerate([0.5, 0.4, 0.3, 0.2]):
        amplification = [np.power(v, gamma) / v for v in input_vals]
        offset = (i - 1.5) * bar_width
        ax2.bar(x_pos + offset, amplification, bar_width, 
               label=f'γ={gamma}', alpha=0.8)
    
    ax2.set_xlabel('Original pixel value', fontsize=12)
    ax2.set_ylabel('Amplification factor', fontsize=12)
    ax2.set_title('Brightness Amplification for Dark Pixels', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{v}' for v in input_vals])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_analysis.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_dir / 'gamma_analysis.png'}")
    plt.close()
    
    # === 图2: 合成图像对比 ===
    size = 200
    
    # 创建简单的合成低光照图像
    image = np.zeros((size, size, 3), dtype=np.float32)
    
    # 渐变背景
    for i in range(size):
        for j in range(size):
            image[i, j] = [0.3 * j/size, 0.2 * i/size, 0.15]
    
    # 添加圆形
    cy, cx = size//3, size//3
    for i in range(size):
        for j in range(size):
            if (i-cy)**2 + (j-cx)**2 < (size//5)**2:
                image[i, j] = [0.7, 0.2, 0.1]
    
    # 添加方块
    image[size//2:size//2+size//4, size//2:size//2+size//4] = [0.1, 0.5, 0.7]
    
    # 模拟低光照
    image = image * 0.12
    
    # 绘制对比图
    gamma_test = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for idx, gamma in enumerate(gamma_test):
        ax = axes[idx // 3, idx % 3]
        corrected = np.clip(np.power(np.clip(image, 1e-8, 1), gamma), 0, 1)
        ax.imshow(corrected)
        
        mean_val = corrected.mean()
        if gamma == 1.0:
            title = f'γ={gamma} (original)\nmean={mean_val:.3f}'
        elif gamma == 0.4:
            title = f'γ={gamma} (RECOMMENDED)\nmean={mean_val:.3f}'
        else:
            title = f'γ={gamma}\nmean={mean_val:.3f}'
        ax.set_title(title, fontsize=11, fontweight='bold' if gamma == 0.4 else 'normal')
        ax.axis('off')
    
    plt.suptitle('Gamma Correction Effect on Synthetic Low-light Image', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_synthetic_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_dir / 'gamma_synthetic_comparison.png'}")
    plt.close()
    
    # === 打印数值分析 ===
    print("\n" + "="*70)
    print("Gamma 校正数值分析 (I_out = I_in ^ γ)")
    print("="*70)
    
    print("\n【不同 γ 值对暗像素的放大效果】")
    print("-"*70)
    print(f"{'输入值':<12} {'γ=1.0':<12} {'γ=0.5':<12} {'γ=0.4':<12} {'γ=0.3':<12} {'γ=0.2':<12}")
    print("-"*70)
    
    for inp in [0.01, 0.02, 0.05, 0.1, 0.2]:
        row = f"{inp:<12}"
        for gamma in [1.0, 0.5, 0.4, 0.3, 0.2]:
            out = inp ** gamma
            row += f"{out:.4f}      "
        print(row)
    
    print("-"*70)
    
    print("\n【放大倍数 (output / input)】")
    print("-"*70)
    print(f"{'输入值':<12} {'γ=0.5':<12} {'γ=0.4':<12} {'γ=0.3':<12} {'γ=0.2':<12}")
    print("-"*70)
    
    for inp in [0.01, 0.02, 0.05, 0.1, 0.2]:
        row = f"{inp:<12}"
        for gamma in [0.5, 0.4, 0.3, 0.2]:
            amp = (inp ** gamma) / inp
            row += f"{amp:.1f}x        "
        print(row)
    
    print("-"*70)
    
    print("\n【为什么选择 γ = 0.4】")
    print("  1. 对极暗像素 (0.01-0.05) 有 6-10x 的放大，足以让 DINO 提取特征")
    print("  2. 对中等暗度像素 (0.1-0.2) 放大适中 (2-4x)，不会过曝")
    print("  3. 保持相对亮度关系，不会完全破坏图像结构")
    print("  4. 经验值：在 LOL、SID 等低光照数据集上效果良好")
    
    print("\n【不使用 Gamma 校正的问题】")
    print("  - DINO 在 ImageNet (正常光照) 上预训练")
    print("  - 低光照图像像素集中在 [0, 0.2]，与训练分布差异大")
    print("  - 导致 DINO 特征质量下降，语义引导失效")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
