#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合图像质量评估脚本
计算 PSNR↑ | SSIM↑ | LPIPS↓ | BRISQUE↓ | NIQE↓ 指标
并将结果保存到 CSV 文件

与项目现有实现保持一致：
- PSNR/SSIM: 使用 skimage.metrics (与 metrics_calc_1.py 一致)
- LPIPS: 使用 lpips 库 (与 metrics_calc_1.py 一致)
- BRISQUE: 使用 brisque 库 (与 metrics_calc_2.py 一致)
- NIQE: 使用 pyiqa 库 (与 metrics_calc_2.py 一致)

使用方法:
    python scripts/metrics/calculate_all_metrics.py \
        --enhanced_dir <增强图像目录> \
        --gt_dir <原始/GT图像目录> \
        --output <输出CSV路径> \
        [--use_gpu]
"""

import argparse
import csv
import os
import sys
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
import lpips

# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 可选依赖
try:
    import pyiqa
    HAVE_PYIQA = True
except ImportError:
    HAVE_PYIQA = False
    print("警告: pyiqa 未安装，NIQE 将使用 basicsr 实现。安装: pip install pyiqa")

try:
    from brisque import BRISQUE
    HAVE_BRISQUE = True
except ImportError:
    HAVE_BRISQUE = False
    print("警告: brisque 未安装，BRISQUE 将不可用。安装: pip install brisque")


def imread(path: str) -> np.ndarray:
    """读取图像，返回 RGB 格式的 numpy 数组 [0, 255]"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img[:, :, [2, 1, 0]]  # BGR -> RGB


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """将图像转换为 LPIPS 所需的 tensor 格式"""
    # [0, 255] -> [-1, 1], HWC -> CHW -> NCHW
    img = np.expand_dims(np.transpose(img, [2, 0, 1]), axis=0)
    return torch.Tensor(img) / 127.5 - 1


class MetricsCalculator:
    """指标计算器，与项目现有实现保持一致"""
    
    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # LPIPS 模型 (与 metrics_calc_1.py 一致)
        self.lpips_model = lpips.LPIPS(net='alex')
        self.lpips_model.to(self.device)
        self.lpips_model.eval()
        
        # NIQE 模型 (与 metrics_calc_2.py 一致)
        self.niqe_metric = None
        if HAVE_PYIQA:
            self.niqe_metric = pyiqa.create_metric('niqe')
            if use_gpu and torch.cuda.is_available():
                self.niqe_metric = self.niqe_metric.to(self.device)
        
        # BRISQUE 模型 (与 metrics_calc_2.py 一致)
        self.brisque_model = None
        if HAVE_BRISQUE:
            self.brisque_model = BRISQUE(url=False)
    
    def calculate_psnr(self, img_enhanced: np.ndarray, img_gt: np.ndarray) -> float:
        """计算 PSNR (与 metrics_calc_1.py 一致，使用 skimage)"""
        return float(calc_psnr(img_gt, img_enhanced))
    
    def calculate_ssim(self, img_enhanced: np.ndarray, img_gt: np.ndarray) -> float:
        """计算 SSIM (与 metrics_calc_1.py 一致，使用 skimage)"""
        score, _ = calc_ssim(img_gt, img_enhanced, full=True, channel_axis=2)
        return float(score)
    
    def calculate_lpips(self, img_enhanced: np.ndarray, img_gt: np.ndarray) -> float:
        """计算 LPIPS (与 metrics_calc_1.py 一致)"""
        tensor_enhanced = to_tensor(img_enhanced).to(self.device)
        tensor_gt = to_tensor(img_gt).to(self.device)
        with torch.no_grad():
            dist = self.lpips_model.forward(tensor_enhanced, tensor_gt)
        return float(dist.item())
    
    def calculate_niqe(self, img_rgb: np.ndarray) -> float:
        """计算 NIQE (与 metrics_calc_2.py 一致，使用 pyiqa)"""
        if self.niqe_metric is None:
            # 回退到 basicsr 实现
            return self._calculate_niqe_basicsr(img_rgb)
        
        try:
            # pyiqa 需要 [0, 1] 范围的 tensor
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            with torch.no_grad():
                niqe_val = self.niqe_metric(img_tensor)
            return float(niqe_val.item())
        except Exception as e:
            print(f"pyiqa NIQE 计算失败: {e}，使用 basicsr 实现")
            return self._calculate_niqe_basicsr(img_rgb)
    
    def _calculate_niqe_basicsr(self, img_rgb: np.ndarray) -> float:
        """使用 basicsr 实现计算 NIQE"""
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(ROOT_DIR))
        from basicsr.metrics.niqe import calculate_niqe
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return float(calculate_niqe(img_bgr, crop_border=0, input_order='HWC', convert_to='y'))
    
    def calculate_brisque(self, img_bgr: np.ndarray) -> float:
        """计算 BRISQUE (与 metrics_calc_2.py 一致，使用 brisque 库)"""
        if self.brisque_model is None:
            return float('nan')
        
        try:
            score = self.brisque_model.score(img_bgr)
            return float(score)
        except Exception as e:
            print(f"BRISQUE 计算失败: {e}")
            return float('nan')


def extract_core_name(filename: str) -> str:
    """
    提取文件的核心名称，去除常见前缀 (与 compare_folders_metrics.py 一致)
    例如: normal_001.png -> 001, low001.png -> 001
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # 常见前缀（带下划线和不带下划线）
    prefixes = [
        "normal_", "normal", "low_", "low", 
        "high_", "high", "gt_", "gt", "ref_", "ref",
        "enhanced_", "enhanced", "output_", "output",
    ]
    for prefix in prefixes:
        if name_without_ext.lower().startswith(prefix.lower()):
            core = name_without_ext[len(prefix):]
            if core:
                return core
    
    return name_without_ext


def get_image_pairs(enhanced_dir: str, gt_dir: str, img_ext: str = "png") -> list:
    """
    获取增强图像和GT图像的配对列表
    支持两种模式：
    1. 文件名完全相同 -> 直接匹配
    2. 文件名有前缀差异 (low_xxx vs normal_xxx) -> 按核心名称匹配
    """
    # 获取文件列表
    enhanced_files = natsorted(glob.glob(os.path.join(enhanced_dir, f"*.{img_ext}")))
    gt_files = natsorted(glob.glob(os.path.join(gt_dir, f"*.{img_ext}")))
    
    if not enhanced_files:
        raise ValueError(f"在 {enhanced_dir} 中没有找到 .{img_ext} 图像")
    if not gt_files:
        raise ValueError(f"在 {gt_dir} 中没有找到 .{img_ext} 图像")
    
    # 构建核心名称到文件路径的映射
    enhanced_map = {extract_core_name(os.path.basename(f)): f for f in enhanced_files}
    gt_map = {extract_core_name(os.path.basename(f)): f for f in gt_files}
    
    # 找到共同的核心名称
    common_cores = set(enhanced_map.keys()) & set(gt_map.keys())
    
    if not common_cores:
        # 如果核心名称匹配失败，回退到顺序配对
        print("警告: 无法按核心名称匹配，回退到顺序配对模式")
        if len(enhanced_files) != len(gt_files):
            raise ValueError(
                f"目录中文件数量不一致：{len(enhanced_files)} vs {len(gt_files)}。"
            )
        pairs = []
        for enhanced_path, gt_path in zip(enhanced_files, gt_files):
            pairs.append({
                'name': os.path.basename(enhanced_path),
                'enhanced': enhanced_path,
                'gt': gt_path
            })
    else:
        # 按核心名称匹配
        pairs = []
        for core in natsorted(common_cores):
            pairs.append({
                'name': core,
                'enhanced': enhanced_map[core],
                'gt': gt_map[core]
            })
        
        if len(common_cores) < len(enhanced_files):
            print(f"注意: 有 {len(enhanced_files) - len(common_cores)} 张增强图像没有匹配的GT")
    
    print(f"找到 {len(pairs)} 对匹配的图像")
    return pairs


def save_results_to_csv(results: list, output_path: str, avg_metrics: dict):
    """将结果保存到 CSV 文件"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fieldnames = ['Image', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'BRISQUE↓', 'NIQE↓']
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 写入每张图像的结果
        for result in results:
            row = {
                'Image': result['name'],
                'PSNR↑': f"{result['PSNR']:.4f}",
                'SSIM↑': f"{result['SSIM']:.4f}",
                'LPIPS↓': f"{result['LPIPS']:.4f}",
                'BRISQUE↓': f"{result['BRISQUE']:.4f}" if not np.isnan(result['BRISQUE']) else 'N/A',
                'NIQE↓': f"{result['NIQE']:.4f}"
            }
            writer.writerow(row)
        
        # 空行
        writer.writerow({fn: '' for fn in fieldnames})
        
        # 平均值
        avg_row = {
            'Image': 'Average',
            'PSNR↑': f"{avg_metrics['PSNR']:.4f}",
            'SSIM↑': f"{avg_metrics['SSIM']:.4f}",
            'LPIPS↓': f"{avg_metrics['LPIPS']:.4f}",
            'BRISQUE↓': f"{avg_metrics['BRISQUE']:.4f}" if not np.isnan(avg_metrics['BRISQUE']) else 'N/A',
            'NIQE↓': f"{avg_metrics['NIQE']:.4f}"
        }
        writer.writerow(avg_row)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='计算图像质量评估指标 (PSNR, SSIM, LPIPS, BRISQUE, NIQE)')
    parser.add_argument('--enhanced_dir', type=str, required=True,
                        help='增强图像目录路径')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground Truth (原始/参考) 图像目录路径')
    parser.add_argument('--output', type=str, default='metrics_results.csv',
                        help='输出 CSV 文件路径 (默认: metrics_results.csv)')
    parser.add_argument('--type', type=str, default='png',
                        help='图像文件扩展名 (默认: png)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用 GPU 加速计算')
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.isdir(args.enhanced_dir):
        print(f"错误: 增强图像目录不存在: {args.enhanced_dir}")
        sys.exit(1)
    if not os.path.isdir(args.gt_dir):
        print(f"错误: GT 图像目录不存在: {args.gt_dir}")
        sys.exit(1)
    
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化计算器
    print("初始化指标计算器...")
    calculator = MetricsCalculator(use_gpu=args.use_gpu)
    
    # 获取图像对 (与 metrics_calc_1.py 一致的匹配逻辑)
    image_pairs = get_image_pairs(args.enhanced_dir, args.gt_dir, args.type)
    
    # 存储结果
    all_results = []
    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0, 'BRISQUE': 0, 'NIQE': 0}
    brisque_count = 0
    
    print("\n开始计算指标...")
    for pair in tqdm(image_pairs, desc="处理图像"):
        try:
            # 加载图像 (RGB)
            img_enhanced = imread(pair['enhanced'])
            img_gt = imread(pair['gt'])
            
            # 检查尺寸
            if img_enhanced.shape != img_gt.shape:
                print(f"\n警告: {pair['name']} 尺寸不匹配，跳过")
                continue
            
            # 计算有参考指标
            psnr_val = calculator.calculate_psnr(img_enhanced, img_gt)
            ssim_val = calculator.calculate_ssim(img_enhanced, img_gt)
            lpips_val = calculator.calculate_lpips(img_enhanced, img_gt)
            
            # 计算无参考指标 (只需要增强图像)
            niqe_val = calculator.calculate_niqe(img_enhanced)
            
            # BRISQUE 需要 BGR 格式
            img_enhanced_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)
            brisque_val = calculator.calculate_brisque(img_enhanced_bgr)
            
            # 保存结果
            result = {
                'name': pair['name'],
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'LPIPS': lpips_val,
                'BRISQUE': brisque_val,
                'NIQE': niqe_val
            }
            all_results.append(result)
            
            # 累加
            metrics_sum['PSNR'] += psnr_val
            metrics_sum['SSIM'] += ssim_val
            metrics_sum['LPIPS'] += lpips_val
            metrics_sum['NIQE'] += niqe_val
            if not np.isnan(brisque_val):
                metrics_sum['BRISQUE'] += brisque_val
                brisque_count += 1
                
        except Exception as e:
            print(f"\n处理 {pair['name']} 时出错: {e}")
            continue
    
    if not all_results:
        print("错误: 没有成功处理任何图像")
        sys.exit(1)
    
    # 计算平均值
    num_images = len(all_results)
    avg_metrics = {
        'PSNR': metrics_sum['PSNR'] / num_images,
        'SSIM': metrics_sum['SSIM'] / num_images,
        'LPIPS': metrics_sum['LPIPS'] / num_images,
        'BRISQUE': metrics_sum['BRISQUE'] / brisque_count if brisque_count > 0 else float('nan'),
        'NIQE': metrics_sum['NIQE'] / num_images
    }
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("指标汇总 (平均值)")
    print("=" * 60)
    print(f"  PSNR↑:    {avg_metrics['PSNR']:.4f}")
    print(f"  SSIM↑:    {avg_metrics['SSIM']:.4f}")
    print(f"  LPIPS↓:   {avg_metrics['LPIPS']:.4f}")
    if not np.isnan(avg_metrics['BRISQUE']):
        print(f"  BRISQUE↓: {avg_metrics['BRISQUE']:.4f}")
    else:
        print(f"  BRISQUE↓: N/A (brisque 库未安装)")
    print(f"  NIQE↓:    {avg_metrics['NIQE']:.4f}")
    print("=" * 60)
    print(f"共处理 {num_images} 张图像")
    
    # 保存到 CSV
    save_results_to_csv(all_results, args.output, avg_metrics)


if __name__ == '__main__':
    main()
