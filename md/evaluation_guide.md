# 图像质量评估指南 (Evaluation Guide)

本文档详细说明了如何使用 `LLIE/metrics_cal.py` 脚本来评估图像增强的效果。

## 1. 脚本概述

`LLIE/metrics_cal.py` 是一个用于定量评估图像质量的工具。它通过计算并报告三种业界常用的图像质量指标，来比较两组图像（通常是模型输出结果和高质量参考图像）之间的差异。

## 2. 评估指标

脚本计算并报告以下三种指标的平均值：

1.  **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比。值越大，表示图像失真越小，质量越好。
2.  **SSIM (Structural Similarity Index)**: 结构相似性指数。衡量两张图片在结构上的相似度，值越接近 1，表示越相似。
3.  **LPIPS (Learned Perceptual Image Patch Similarity)**: 学习感知图像块相似度。一种更符合人类视觉感知的相似度度量，值越小，表示两张图在感知上越相似。

## 3. 核心参数

脚本通过命令行参数来指定需要比较的图像目录。

-   `-dirA`: **目录A (通常是参考图像)**
    -   **作用**: 指定高质量的、未经处理的参考图像（Ground Truth）所在的文件夹。
-   `-dirB`: **目录B (通常是待评估图像)**
    -   **作用**: 指定由模型生成的、需要进行质量评估的增强图像所在的文件夹。
-   `-type`: **图像文件类型**
    -   **作用**: 指定要评估的图像文件的扩展名。
    -   **默认值**: `png`
-   `--use_gpu`: **使用GPU**
    -   **作用**: 一个可选标志，启用后会使用 GPU 来加速 LPIPS 指标的计算。

## 4. 命名要求 (重要)

该脚本的正确运行依赖于两个目录中文件的**排序结果必须完全一致**。脚本内部使用 `zip` 函数将两个文件列表进行配对比较。

为了确保评估的准确性（即 `dirA` 中的 `image_1` 与 `dirB` 中的 `image_1` 进行比较），最可靠的方式是**确保两个目录中对应图像的文件名完全相同**。

**正确示例**:

-   **`dirA` (参考图像)**:
    -   `001.png`
    -   `002.png`
-   **`dirB` (待评估图像)**:
    -   `001.png`
    -   `002.png`

幸运的是，`LLIE/test.py` 脚本生成的输出文件名与输入文件名主体一致，天然满足此要求。

## 5. 执行示例

要评估 `LLIE/test.py` 在 `LOL-v2` 数据集上的增强效果，你需要将**参考图像目录**和**增强结果目录**分别传给 `-dirA` 和 `-dirB`。

在项目根目录下执行以下命令：

```bash
python LLIE/metrics_cal.py \
    -dirA ./datasets/LOL-v2/Real_captured/Test/Normal \
    -dirB ./results/LOL-v2/
```

-   `-dirA` 指向了 `LOL-v2` 测试集中的**高质量正常光照图像** (Ground Truth)。
-   `-dirB` 指向了 `LLIE/test.py` **生成的增强图像**。

如果希望使用 GPU 加速计算，可以添加 `--use_gpu` 标志：
```bash
python LLIE/metrics_cal.py \
    -dirA ./datasets/LOL-v2/Real_captured/Test/Normal \
    -dirB ./results/LOL-v2/ \
    --use_gpu
```
脚本执行完毕后，会在终端输出 PSNR, SSIM, LPIPS 三个指标的平均值。
