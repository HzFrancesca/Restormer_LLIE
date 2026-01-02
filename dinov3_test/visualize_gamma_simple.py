"""
简化版 Gamma 校正可视化
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    image = np.clip(image, 1e-8, 1.0)
    return np.power(image, gamma)


def main():
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载真实图像
    img_path = "datasets/LOL-v2/Real_captured/Test/Low"
    image = None
    img_name = "synthetic"
    
    if os.path.exists(img_path):
        images = list(Path(img_path).glob("*.png"))
        if images:
            try:
                img = Image.open(str(images[0])).convert('RGB')
                # 缩小尺寸
                img.thumbnail((400, 400))
                image = np.array(img).astype(np.float32) / 255.0
                img_name = images[0].name
                print(f"加载图像: {img_name}")
            except Exception as e:
                print(f"加载失败: {e}")
    
    # 如果没有真实图像，创建合成图像
    if image is None:
        print("使用合成低光照图像")
        size = 256
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        
        image = np.zeros((size, size, 3))
        # 圆形
        for i in range(size):
            for j in range(size):
                if (i - size//3)**2 + (j - size//3)**2 < (size//6)**2:
                    image[i, j] = [0.8, 0.3, 0.2]
        # 方块
        image[size//2:size//2+size//4, size//2:size//2+size//4] = [0.2, 0.6, 0.8]
        # 背景
        mask = np.all(image == 0, axis=2)
        image[mask] = np.stack([0.3*xx[mask], 0.4*yy[mask], 0.2*(xx[mask]+yy[mask])/2], axis=1)
        # 压暗
        image = image * 0.15
    
    # === 图1: Gamma 曲线 ===
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 1, 256)
    gamma_values = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    
    for gamma in gamma_values:
        y = np.power(x, gamma)
        label = f'γ={gamma}' + (' (no change)' if gamma == 1.0 else ' (recommended)' if gamma == 0.4 else '')
        ax.plot(x, y, label=label, linewidth=2)
    
    ax.set_xlabel('Input pixel value', fontsize=12)
    ax.set_ylabel('Output pixel value', fontsize=12)
    ax.set_title('Gamma Correction Curve: I_out = I_in ^ γ', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_curve.png", dpi=150)
    print(f"保存: {output_dir / 'gamma_curve.png'}")
    plt.close()
    
    # === 图2: 不同 Gamma 效果对比 ===
    gamma_values = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // 3, idx % 3]
        corrected = gamma_correction(image, gamma)
        ax.imshow(np.clip(corrected, 0, 1))
        
        mean_val = corrected.mean()
        if gamma == 1.0:
            title = f'γ={gamma} (original)\nmean={mean_val:.3f}'
        elif gamma == 0.4:
            title = f'γ={gamma} (recommended)\nmean={mean_val:.3f}'
        else:
            title = f'γ={gamma}\nmean={mean_val:.3f}'
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    plt.suptitle(f'Gamma Correction Effect on Low-light Image\n({img_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_comparison.png", dpi=150)
    print(f"保存: {output_dir / 'gamma_comparison.png'}")
    plt.close()
    
    # === 图3: 直方图对比 ===
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    gamma_test = [1.0, 0.4, 0.2]
    
    for idx, gamma in enumerate(gamma_test):
        corrected = gamma_correction(image, gamma)
        
        # 图像
        ax_img = axes[0, idx]
        ax_img.imshow(np.clip(corrected, 0, 1))
        title = 'γ=1.0 (original)' if gamma == 1.0 else f'γ={gamma}' + (' (recommended)' if gamma == 0.4 else '')
        ax_img.set_title(title, fontsize=11)
        ax_img.axis('off')
        
        # 直方图
        ax_hist = axes[1, idx]
        gray = 0.299*corrected[:,:,0] + 0.587*corrected[:,:,1] + 0.114*corrected[:,:,2]
        ax_hist.hist(gray.flatten(), bins=50, range=(0, 1), alpha=0.7, color='steelblue')
        ax_hist.axvline(x=gray.mean(), color='red', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Pixel value')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'Histogram (mean={gray.mean():.3f})')
        ax_hist.set_xlim(0, 1)
    
    plt.suptitle('Gamma Correction: Image and Histogram Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_histogram.png", dpi=150)
    print(f"保存: {output_dir / 'gamma_histogram.png'}")
    plt.close()
    
    # === 打印数值分析 ===
    print("\n" + "="*60)
    print("Gamma 校正数值分析")
    print("="*60)
    print("\n低光照像素值 0.05 在不同 gamma 下的变化:")
    for gamma in [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]:
        out = 0.05 ** gamma
        amp = out / 0.05
        print(f"  γ={gamma}: {0.05:.3f} → {out:.3f} (放大 {amp:.1f}x)")
    
    print("\n完成！图像保存在 scripts/results/")


if __name__ == "__main__":
    main()
