# 模型推理指南 (Inference Guide)

本文档详细说明了如何使用 `LLIE/test.py` 脚本对低光照图像进行增强推理。

## 1. 脚本概述

推理过程的核心入口是 `LLIE/test.py` 脚本。该脚本负责加载预训练的 Restormer 模型，处理输入的低光照图像，并保存增强后的结果。

## 2. 核心参数

脚本通过命令行参数来控制推理过程。主要参数如下：

- `--input_dir`: **输入图像目录**
  - **作用**: 指定待处理的低光照图像所在的文件夹。
  - **默认值**: `./datasets/LOL-v2/Real_captured/Test/Low/`

- `--result_dir`: **结果保存目录**
  - **作用**: 指定增强后图像的保存位置。
  - **默认值**: `./results/LOL-v2/`

- `--weights`: **模型权重文件**
  - **作用**: 指向已经训练好的模型权重文件 (`.pth` 格式)。
  - **默认值**: `./pretrained_models/lowlight.pth`

- `--opt`: **模型配置文件**
  - **作用**: 指定定义 Restormer 网络结构的 YAML 配置文件。
  - **默认值**: `LLIE/Options/LowLight_Restormer.yml`

## 3. 推理流程

脚本的执行流程如下：

1. **解析参数**: 读取用户在命令行中提供的路径和配置。
2. **加载配置**: 解析 `--opt` 指定的 `.yml` 文件，获取 `network_g` 字段下的模型结构参数（如 `dim`, `num_blocks` 等）。
3. **构建模型**: 根据配置参数初始化 `Restormer` 网络模型。
4. **加载权重**: 从 `--weights` 指定的路径加载预训练权重到模型中。
5. **图像处理与推理**:
    - 脚本会遍历 `--input_dir` 目录下的所有 `.png` 和 `.jpg` 图像文件。
    - 对每张图像，依次执行以下操作：
        a.  加载图像并归一化。
        b.  转换为 PyTorch 张量。
        c.  对图像尺寸进行**填充 (Padding)**，使其宽高满足模型输入要求（8的倍数）。
        d.  将处理后的张量送入模型进行推理。
        e.  对模型输出的结果进行**裁剪 (Unpadding)**，恢复至原始尺寸。
        f.  将像素值裁剪至 `[0, 1]` 范围，并转换为图像格式。
6. **保存结果**: 将增强后的图像保存到 `--result_dir` 目录。

## 4. 命名规则

脚本在保存结果时遵循以下命名规则：

- **输出文件名**与**输入文件名**的主体部分保持一致。
- **输出文件的扩展名**统一为 `.png`。

**示例**:

- 输入图像: `[--input_dir]/img_123.jpg`
- 输出图像: `[--result_dir]/img_123.png`

## 5. 执行示例

要在 `LOL-v2` 测试集上运行推理，可以使用以下命令（在项目根目录下执行）：

```bash
python LLIE/test.py \
    --input_dir ./datasets/LOL-v2/Real_captured/Test/Low/ \
    --result_dir ./results/LOL-v2/ \
    --weights ./pretrained_models/lowlight.pth
```

该命令会处理 `Low` 目录下的所有低光照图像，并将增强结果保存在 `./results/LOL-v2/` 目录下。
