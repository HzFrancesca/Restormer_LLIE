"""
可视化 Gamma 校正对低光照图像的影响

目标：
1. 展示不同 gamma 值对图像亮度的影响
2. 对比 DINO 特征在不同 gamma 下的差异
3. 说明为什么 gamma=0.4 是合理的选择

Gamma 校正公式: I_out = I_in ^ gamma
- gamma < 1: 提亮暗区（用于低光照图像预处理）
- gamma = 1: 无变化
- gamma > 1: 压暗亮区
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    应用 gamma 校正
    
    Args:
        image: 输入图像 [H, W, C]，值域 [0, 1]
        gamma: gamma 值
    
    Returns:
        校正后的图像
    """
    # 避免数值问题
    image = np.clip(image, 1e-8, 1.0)
    return np.power(image, gamma)


def create_synthetic_low_light_image(size: int = 256) -> np.ndarray:
    """创建合成的低光照测试图像"""
    # 创建渐变背景
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # 基础图像（模拟场景）
    image = np.zeros((size, size, 3))
    
    # 添加一些形状
    # 圆形
    center = (size // 3, size // 3)
    radius = size // 6
    for i in range(size):
        for j in range(size):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                image[i, j] = [0.8, 0.3, 0.2]  # 红色圆
    
    # 矩形
    image[size//2:size//2+size//4, size//2:size//2+size//4] = [0.2, 0.6, 0.8]  # 蓝色方块
    
    # 渐变背景
    background_mask = np.all(image == 0, axis=2)
    image[background_mask] = np.stack([
        0.3 * xx[background_mask],
        0.4 * yy[background_mask],
        0.2 * (xx[background_mask] + yy[background_mask]) / 2
    ], axis=1)
    
    # 模拟低光照：整体压暗
    low_light_factor = 0.15  # 模拟非常暗的场景
    image = image * low_light_factor
    
    return image


def load_real_low_light_image() -> tuple:
    """尝试加载真实的低光照图像"""
    from PIL import Image
    
    # 尝试从 LOL-v2 数据集加载
    possible_paths = [
        "datasets/LOL-v2/Real_captured/Test/Low",
        "datasets/LOL-v2/Synthetic/Test/Low",
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            images = list(Path(base_path).glob("*.png")) + list(Path(base_path).glob("*.jpg"))
            if images:
                img_path = str(images[0])
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img).astype(np.float32) / 255.0
                    # 缩放到合理大小
                    if max(img.shape[:2]) > 512:
                        scale = 512 / max(img.shape[:2])
                        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(new_size)) / 255.0
                    return img, img_path
                except Exception as e:
                    print(f"加载图像失败: {e}")
                    continue
    
    return None, None


def visualize_gamma_effects():
    """可视化不同 gamma 值的效果"""
    
    # 尝试加载真实图像，否则使用合成图像
    real_img, img_path = load_real_low_light_image()
    
    if real_img is not None:
        image = real_img
        title_prefix = f"真实低光照图像: {Path(img_path).name}"
    else:
        image = create_synthetic_low_light_image()
        title_prefix = "合成低光照图像"
    
    # 测试的 gamma 值
    gamma_values = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix}\nGamma 校正效果对比 (I_out = I_in ^ γ)', fontsize=14)
    
    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // 3, idx % 3]
        
        corrected = gamma_correction(image, gamma)
        ax.imshow(np.clip(corrected, 0, 1))
        
        # 计算统计信息
        mean_val = corrected.mean()
        std_val = corrected.std()
        
        if gamma == 1.0:
            label = f'γ = {gamma} (原图)\n均值: {mean_val:.3f}, 标准差: {std_val:.3f}'
        elif gamma == 0.4:
            label = f'γ = {gamma} (推荐)\n均值: {mean_val:.3f}, 标准差: {std_val:.3f}'
        else:
            label = f'γ = {gamma}\n均值: {mean_val:.3f}, 标准差: {std_val:.3f}'
        
        ax.set_title(label)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "gamma_correction_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存图像到: {output_dir / 'gamma_correction_comparison.png'}")
    
    plt.close()


def visualize_gamma_curve():
    """可视化 gamma 校正曲线"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：gamma 曲线
    ax1 = axes[0]
    x = np.linspace(0, 1, 256)
    gamma_values = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
    
    for gamma, color in zip(gamma_values, colors):
        y = np.power(x, gamma)
        label = f'γ = {gamma}' + (' (原图)' if gamma == 1.0 else ' (推荐)' if gamma == 0.4 else '')
        ax1.plot(x, y, label=label, color=color, linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='线性 (γ=1)')
    ax1.set_xlabel('输入像素值', fontsize=12)
    ax1.set_ylabel('输出像素值', fontsize=12)
    ax1.set_title('Gamma 校正曲线: I_out = I_in ^ γ', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 添加注释
    ax1.annotate('低光照区域\n(暗部提升明显)', 
                xy=(0.1, 0.4), xytext=(0.25, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # 右图：不同输入值的放大倍数
    ax2 = axes[1]
    input_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    gamma_test = [0.2, 0.3, 0.4, 0.5, 0.7]
    
    bar_width = 0.12
    x_pos = np.arange(len(input_values))
    
    for i, gamma in enumerate(gamma_test):
        amplification = [np.power(v, gamma) / v for v in input_values]
        offset = (i - len(gamma_test)/2 + 0.5) * bar_width
        bars = ax2.bar(x_pos + offset, amplification, bar_width, 
                      label=f'γ = {gamma}', alpha=0.8)
    
    ax2.set_xlabel('原始像素值', fontsize=12)
    ax2.set_ylabel('亮度放大倍数', fontsize=12)
    ax2.set_title('不同 Gamma 对暗部像素的放大效果', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{v}' for v in input_values])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加参考线
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='无放大')
    
    plt.tight_layout()
    
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "gamma_curve_analysis.png", dpi=150, bbox_inches='tight')
    print(f"保存图像到: {output_dir / 'gamma_curve_analysis.png'}")
    
    plt.close()


def visualize_histogram_comparison():
    """可视化 gamma 校正前后的直方图变化"""
    
    # 加载或创建图像
    real_img, _ = load_real_low_light_image()
    image = real_img if real_img is not None else create_synthetic_low_light_image()
    
    gamma_values = [1.0, 0.4, 0.2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, gamma in enumerate(gamma_values):
        corrected = gamma_correction(image, gamma)
        
        # 上排：图像
        ax_img = axes[0, idx]
        ax_img.imshow(np.clip(corrected, 0, 1))
        title = 'γ = 1.0 (原图)' if gamma == 1.0 else f'γ = {gamma}' + (' (推荐)' if gamma == 0.4 else '')
        ax_img.set_title(title, fontsize=12)
        ax_img.axis('off')
        
        # 下排：直方图
        ax_hist = axes[1, idx]
        # 转换为灰度计算直方图
        gray = 0.299 * corrected[:,:,0] + 0.587 * corrected[:,:,1] + 0.114 * corrected[:,:,2]
        ax_hist.hist(gray.flatten(), bins=50, range=(0, 1), alpha=0.7, color='steelblue', edgecolor='black')
        ax_hist.set_xlabel('像素值', fontsize=10)
        ax_hist.set_ylabel('频数', fontsize=10)
        ax_hist.set_title(f'亮度直方图 (均值: {gray.mean():.3f})', fontsize=10)
        ax_hist.set_xlim(0, 1)
        
        # 添加均值线
        ax_hist.axvline(x=gray.mean(), color='red', linestyle='--', linewidth=2, label=f'均值')
        ax_hist.legend()
    
    plt.suptitle('Gamma 校正对图像直方图的影响\n(将暗部像素向右移动，增加整体亮度)', fontsize=14)
    plt.tight_layout()
    
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "gamma_histogram_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存图像到: {output_dir / 'gamma_histogram_comparison.png'}")
    
    plt.close()


def print_gamma_explanation():
    """打印 gamma 校正的解释"""
    
    print("=" * 70)
    print("Gamma 校正在 DINO 低光照预处理中的作用")
    print("=" * 70)
    print()
    print("【问题背景】")
    print("  DINO/DINOv3 是在 ImageNet 上预训练的，ImageNet 图像通常是正常光照。")
    print("  低光照图像的像素值集中在 [0, 0.2] 范围，与 DINO 的训练分布差异很大。")
    print()
    print("【Gamma 校正公式】")
    print("  I_out = I_in ^ γ")
    print()
    print("【不同 γ 值的效果】")
    print("-" * 70)
    print(f"  {'γ 值':<10} {'效果':<30} {'适用场景':<25}")
    print("-" * 70)
    print(f"  {'1.0':<10} {'无变化':<30} {'正常光照图像':<25}")
    print(f"  {'0.7':<10} {'轻微提亮':<30} {'轻度低光照':<25}")
    print(f"  {'0.5':<10} {'中等提亮 (平方根)':<30} {'中度低光照':<25}")
    print(f"  {'0.4':<10} {'较强提亮 (推荐)':<30} {'典型低光照场景':<25}")
    print(f"  {'0.3':<10} {'强提亮':<30} {'极暗场景':<25}")
    print(f"  {'0.2':<10} {'极强提亮':<30} {'几乎全黑场景':<25}")
    print("-" * 70)
    print()
    print("【为什么选择 γ = 0.4】")
    print("  1. 平衡性：既能有效提亮暗区，又不会过度放大噪声")
    print("  2. 对比度保持：保留图像的相对亮度关系")
    print("  3. 经验值：在多个低光照数据集上表现良好")
    print()
    print("【数值示例】")
    print("  原始像素值 0.05 (很暗):")
    print(f"    γ=1.0: {0.05**1.0:.4f} (不变)")
    print(f"    γ=0.5: {0.05**0.5:.4f} (提亮 {0.05**0.5/0.05:.1f}x)")
    print(f"    γ=0.4: {0.05**0.4:.4f} (提亮 {0.05**0.4/0.05:.1f}x)")
    print(f"    γ=0.3: {0.05**0.3:.4f} (提亮 {0.05**0.3/0.05:.1f}x)")
    print()
    print("【不使用 Gamma 校正的后果】")
    print("  - DINO 特征质量下降：模型难以从极暗图像中提取有意义的语义")
    print("  - 语义引导失效：DINO 无法正确识别物体和场景")
    print("  - 训练不稳定：损失函数中的 DINO 语义损失可能产生噪声梯度")
    print()


if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 打印解释
    print_gamma_explanation()
    
    # 可视化
    print("\n生成可视化图像...")
    visualize_gamma_curve()
    visualize_gamma_effects()
    visualize_histogram_comparison()
    
    print("\n完成！所有图像已保存到 scripts/results/ 目录")
