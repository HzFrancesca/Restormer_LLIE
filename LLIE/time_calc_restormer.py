## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
import time
from tqdm import tqdm
import yaml

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte

# -------------------------------------------------------------------
# 1. 参数设置
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Image Low-Light Enhancement using Restormer"
)

parser.add_argument(
    "--input_dir",
    default="./datasets/LOL-v2/Real_captured/Test/Low/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", default="./results/LOL-v2/", type=str, help="Directory for results"
)
parser.add_argument(
    "--weights",
    default="./pretrained_models/lowlight.pth",
    type=str,
    help="Path to weights",
)
parser.add_argument(
    "--opt",
    type=str,
    default="LLIM/Options/LowLight_Restormer.yml",
    help="Path to option YAML file.",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=1,
    help="Number of inference runs per image for averaging",
)

args = parser.parse_args()

# -------------------------------------------------------------------
# 2. 模型加载
# -------------------------------------------------------------------
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(args.opt, mode="r", encoding="utf-8"), Loader=Loader)
s = x["network_g"].pop("type")

model_restoration = Restormer(**x["network_g"])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print(f"===> Loading weights: {args.weights}")

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# -------------------------------------------------------------------
# 3. GPU 预热 (Warm-up) - [关键改进步骤]
# -------------------------------------------------------------------
print("\n===> Performing GPU Warm-up...")
with torch.no_grad():
    # 创建一个随机张量，尺寸建议接近真实图片 (例如 128x128 或 256x256)
    # 通道数必须与模型输入一致 (通常是 3)
    dummy_input = torch.randn(1, 3, 400, 600).cuda()
    
    # 进行 10 次空跑，激活 CUDA 核心和缓存
    for _ in range(10):
        _ = model_restoration(dummy_input)
    
    # 必须同步，确保预热彻底完成
    torch.cuda.synchronize()
print("===> Warm-up finished.")

# -------------------------------------------------------------------
# 4. 数据准备
# -------------------------------------------------------------------
factor = 8
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = args.input_dir
files = natsorted(
    glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg"))
)

print(f"\n===> Processing {len(files)} images...")

# -------------------------------------------------------------------
# 5. 推理循环 (Inference Loop)
# -------------------------------------------------------------------
total_inference_time = 0
image_count = 0
first_image_time = None

with torch.no_grad():
    for idx, file_ in enumerate(tqdm(files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # 读取与预处理
        img = np.float32(utils.load_img(file_)) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

        # --- [多次推理取平均] ---
        run_times = []
        for run_idx in range(args.num_runs):
            # 1. 第一次同步：确保之前的操作全部完成
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 2. 模型推理
            restored = model_restoration(input_)
            
            # 3. 第二次同步：确保 GPU 跑完了模型才停止计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            run_times.append(end_time - start_time)
        
        # 计算平均推理时间
        avg_inference_time = np.mean(run_times)
        
        # 如果是第一张图片，单独记录但不计入统计
        if idx == 0:
            first_image_time = avg_inference_time
            print(f"\n[First image] Average time: {avg_inference_time:.4f}s (excluded from statistics)")
        else:
            total_inference_time += avg_inference_time
            image_count += 1
        # --- [计时结束] ---

        # 后处理与保存（使用最后一次推理的结果）
        restored = restored[:, :, :h, :w]
        restored = (
            torch.clamp(restored, 0, 1)
            .cpu()
            .detach()
            .permute(0, 2, 3, 1)
            .squeeze(0)
            .numpy()
        )

        utils.save_img(
            (
                os.path.join(
                    result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + ".png"
                )
            ),
            img_as_ubyte(restored),
        )

# -------------------------------------------------------------------
# 6. 结果统计
# -------------------------------------------------------------------
print("\n" + "="*60)
print("INFERENCE TIMING STATISTICS")
print("="*60)
print(f"Number of runs per image: {args.num_runs}")
print(f"Total images processed:   {len(files)}")
print(f"Images in statistics:     {image_count} (first image excluded)")

if first_image_time is not None:
    print(f"\nFirst image time:         {first_image_time:.4f} s ({first_image_time*1000:.2f} ms)")

if image_count > 0:
    avg_time = total_inference_time / image_count
    fps = 1 / avg_time
    print(f"\nAverage time per image:   {avg_time:.4f} s ({avg_time*1000:.2f} ms)")
    print(f"Processing speed:         {fps:.2f} FPS")
    print(f"Total inference time:     {total_inference_time:.4f} s")
print("="*60)