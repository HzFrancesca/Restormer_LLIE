import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import warnings

# 忽略特定警告以保持输出整洁
warnings.filterwarnings('ignore')

try:
    import piq
    import torch
    HAVE_PIQ = True
except ImportError:
    HAVE_PIQ = False
    print("Warning: piq library not found. Install with: pip install piq")

try:
    from brisque import BRISQUE
    HAVE_BRISQUE = True
except ImportError:
    HAVE_BRISQUE = False
    print("Warning: brisque library not found. Install with: pip install brisque")


def calculate_niqe_piq(img_rgb, niqe_metric_obj):
    """
    使用预初始化的 piq 对象计算 NIQE
    """
    if not HAVE_PIQ or img_rgb is None:
        return None
    
    try:
        # 转换为 Tensor 并归一化到 [0, 1]
        # Shape: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 确保不需要梯度计算以节省内存
        with torch.no_grad():
            niqe_value = niqe_metric_obj(img_tensor)
        
        return niqe_value.item()
    except Exception as e:
        # 仅在调试时打印详细错误，避免刷屏
        # print(f"Error calculating NIQE: {e}")
        return None


def calculate_brisque(img_bgr, brisque_obj):
    """
    使用预初始化的对象计算 BRISQUE
    """
    if not HAVE_BRISQUE or img_bgr is None:
        return None
    
    try:
        # brisque 库通常可以接受 BGR numpy 数组
        score = brisque_obj.score(img_bgr)
        return score
    except Exception as e:
        return None


def process_images(input_dir, image_extensions=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']):
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, f'*.{ext}')))
        image_files.extend(glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print("=" * 80)

    # --- 性能优化：在循环外初始化评估对象 ---
    niqe_metric = piq.NIQE() if HAVE_PIQ else None
    brisque_obj = BRISQUE(url=False) if HAVE_BRISQUE else None
    
    # 用于存储结果的列表（存储字典，确保文件名与分数绑定）
    results = []
    
    # 仅用于计算平均值的列表
    valid_niqe_scores = []
    valid_brisque_scores = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = os.path.basename(img_path)
        
        # 读取图像 (读取一次供两个指标使用)
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            print(f"\nFailed to read image: {img_name}")
            results.append({
                'name': img_name,
                'niqe': None,
                'brisque': None
            })
            continue

        # 准备 RGB 图像供 NIQE 使用
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if HAVE_PIQ else None

        # 计算指标
        niqe_score = calculate_niqe_piq(img_rgb, niqe_metric)
        brisque_score = calculate_brisque(img_bgr, brisque_obj)
        
        # 收集有效分数
        if niqe_score is not None:
            valid_niqe_scores.append(niqe_score)
        if brisque_score is not None:
            valid_brisque_scores.append(brisque_score)
            
        # 保存该图片的具体结果
        results.append({
            'name': img_name,
            'niqe': niqe_score,
            'brisque': brisque_score
        })

        # 实时打印（可选）
        # tqdm 会处理进度条，这里如果频繁 print 可能会打乱进度条显示
        # 仅在出错或特定情况下打印较为整洁
        
    # --- 结果汇总与保存 ---
    
    print("\n" + "=" * 80)
    print("AVERAGE SCORES:")
    print("=" * 80)
    
    if valid_niqe_scores:
        avg_niqe = np.mean(valid_niqe_scores)
        std_niqe = np.std(valid_niqe_scores)
        print(f"NIQE:    {avg_niqe:.4f} ± {std_niqe:.4f} (Lower is better)")
    else:
        print("NIQE:    Not calculated or all failed")
    
    if valid_brisque_scores:
        avg_brisque = np.mean(valid_brisque_scores)
        std_brisque = np.std(valid_brisque_scores)
        print(f"BRISQUE: {avg_brisque:.4f} ± {std_brisque:.4f} (Lower is better, 0-100)")
    else:
        print("BRISQUE: Not calculated or all failed")
    
    print("=" * 80)
    
    # 保存结果到文件
    output_file = os.path.join(input_dir, 'no_reference_metrics.txt')
    with open(output_file, 'w') as f:
        f.write("No-Reference Image Quality Metrics\n")
        f.write("=" * 80 + "\n\n")
        
        # 遍历结果字典，确保数据严格对应
        for res in results:
            f.write(f"{res['name']}\n")
            
            if res['niqe'] is not None:
                f.write(f"  NIQE:    {res['niqe']:.4f}\n")
            else:
                f.write(f"  NIQE:    Failed/NaN\n")
                
            if res['brisque'] is not None:
                f.write(f"  BRISQUE: {res['brisque']:.4f}\n")
            else:
                f.write(f"  BRISQUE: Failed/NaN\n")
                
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("AVERAGE SCORES:\n")
        f.write("=" * 80 + "\n")
        
        if valid_niqe_scores:
            f.write(f"NIQE:    {np.mean(valid_niqe_scores):.4f} ± {np.std(valid_niqe_scores):.4f}\n")
        
        if valid_brisque_scores:
            f.write(f"BRISQUE: {np.mean(valid_brisque_scores):.4f} ± {np.std(valid_brisque_scores):.4f}\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate no-reference image quality metrics (NIQE and BRISQUE)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing images to evaluate'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['png', 'jpg', 'jpeg', 'bmp'],
        help='Image file extensions to process'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Process images
    process_images(args.input_dir, args.extensions)


if __name__ == '__main__':
    main()