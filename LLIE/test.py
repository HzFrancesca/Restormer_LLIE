"""
Restormer: Efficient Transformer for High-Resolution Image Restoration
Low-Light Image Enhancement (LLIE) Test Script

Usage Document / 使用文档:
--------------------------
This script is used to perform low-light image enhancement using a trained Restormer model.
本脚本用于使用训练好的 Restormer 模型对低光图像进行增强。

Arguments / 参数说明:
--input_dir:  Directory containing the images to be enhanced.
              存放待增强图像的目录。
              Default: ./datasets/LOL-v2/Real_captured/Test/Low/
--result_dir: Directory where the enhanced images will be saved.
              增强后图像的保存目录。
              Default: ./results/LOL-v2/
--weights:    Path to the pretrained model weights (.pth file).
              预训练模型权重文件的路径。
              Default: ./pretrained_models/lowlight.pth
--opt:        Path to the configuration file (.yml file) that defines the network architecture.
              定义网络结构的配置文件路径。
              Default: LLIE/Options/Restormer.yml

Example command / 使用示例:
python LLIE/test.py --input_dir path/to/low_light_images --result_dir path/to/save_results --weights path/to/model.pth --opt LLIE/Options/config.yml

"""

import numpy as np
import os
import argparse
import time
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

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
    default="LLIE/Options/Restormer_LOLv2_128_16_cos_25k+FFT_Serial.yml",
    help="Path to option YAML file.",
)


args = parser.parse_args()

####### Load yaml #######
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(args.opt, mode="r", encoding="utf-8"), Loader=Loader)

s = x["network_g"].pop("type")
##########################

model_restoration = Restormer(**x["network_g"])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
# Automatically get the padding factor from the yaml configuration's val window_size
window_size = x.get("val", {}).get("window_size", 8)
factor = window_size
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = args.input_dir
files = natsorted(
    glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg"))
)

print(f"\n===> Processing {len(files)} images...")
total_inference_time = 0
image_count = 0

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_)) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

        # Start timing for inference
        start_time = time.time()
        restored = model_restoration(input_)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.time()

        # Record inference time
        inference_time = end_time - start_time
        total_inference_time += inference_time
        image_count += 1

        # Unpad images to original dimensions
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

# Display timing statistics
print("\n" + "=" * 60)
print("INFERENCE TIMING STATISTICS")
print("=" * 60)
print(f"Total images processed: {image_count}")
print(
    f"Total inference time: {total_inference_time:.4f} seconds ({total_inference_time / 60:.2f} minutes)"
)
if image_count > 0:
    avg_time = total_inference_time / image_count
    print(f"Average time per image: {avg_time:.4f} seconds ({avg_time * 1000:.2f} ms)")
    print(f"Processing speed: {1 / avg_time:.2f} images/second")
print("=" * 60)
