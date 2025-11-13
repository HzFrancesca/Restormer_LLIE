import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class Measure:
    def __init__(self, net="alex", use_gpu=False):
        self.device = "cuda" if use_gpu else "cpu"
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=False):
        if gray_scale:
            score, diff = ssim(
                cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY),
                full=True,
                channel_axis=True,
            )
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim(imgA, imgB, full=True, channel_axis=2)
            # score, diff = ssim(imgA, imgB, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f"{psnr:8.2f} {ssim:8.4f} {lpips:8.4f}"


def measure_dirs(dirA, dirB, img_ext="png", use_gpu=False, verbose=False):
    def vprint(message: str):
        if verbose:
            print(message)

    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f"*.{img_ext}"))
    paths_B = fiFindByWildcard(os.path.join(dirB, f"*.{img_ext}"))

    if len(paths_A) != len(paths_B):
        raise ValueError(
            f"目录中文件数量不一致：{len(paths_A)} vs {len(paths_B)}。请确保两侧文件名一一对应。"
        )

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)
    header = f"{'Reference':<32} {'Enhanced':<32} {'PSNR(dB)':>10} {'SSIM':>8} {'LPIPS':>8} {'Time(s)':>8}"
    vprint(header)
    vprint("-" * len(header))

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        result["psnr"], result["ssim"], result["lpips"] = measure.measure(
            imread(pathA), imread(pathB)
        )
        d = time.time() - t
        vprint(
            f"{os.path.basename(pathA):<32} {os.path.basename(pathB):<32} "
            f"{format_result(**result)} {d:8.2f}"
        )

        results.append(result)

    psnr = np.mean([result["psnr"] for result in results])
    ssim = np.mean([result["ssim"] for result in results])
    lpips = np.mean([result["lpips"] for result in results])
    total_time = time.time() - t_init

    vprint("-" * len(header))
    vprint(
        f"{'Average':<32} {'-':<32} {format_result(psnr, ssim, lpips)} {total_time:8.1f}"
    )
    vprint(f"Processed {len(results)} image pairs in {total_time:0.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dirA", default=r"C:\Users\CTS\Desktop\send_single\1", type=str
    )
    parser.add_argument(
        "-dirB", default=r"C:\Users\CTS\Desktop\send_single\2", type=str
    )
    parser.add_argument("-type", default="png")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    img_ext = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, img_ext=img_ext, use_gpu=use_gpu, verbose=True)
