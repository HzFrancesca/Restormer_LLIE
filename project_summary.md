# Restormer 项目文档

## 1. 项目概述

Restormer 是一个用于高分辨率图像恢复的高效 Transformer 模型。该项目在 CVPR 2022 上发表，并获得了口头报告的资格。它旨在解决传统 CNN 模型在图像恢复任务中的感受野有限和对输入内容适应性不强的问题，同时通过创新的模型设计，克服了 Transformer 模型在处理高分辨率图像时计算复杂度过高的问题。

Restormer 在多个图像恢复任务上取得了业界领先的成果，包括：

* 图像去雨 (Image Deraining)
* 单图像运动去模糊 (Single-Image Motion Deblurring)
* 散焦去模糊 (Defocus Deblurring)，包括单图像和双像素数据
* 图像去噪 (Image Denoising)，包括高斯灰度/彩色去噪和真实图像去噪

该项目的代码基于 [BasicSR](https://github.com/xinntao/BasicSR) 工具箱和 [HINet](https://github.com/megvii-model/HINet) 实现。

## 2. 目录结构

```
.
├── basicsr/              # 核心库，包含模型、数据加载、损失函数等
│   ├── archs/            # 模型架构，主要是 Restormer 的实现
│   ├── data/             # 数据集加载和预处理
│   ├── losses/           # 损失函数的定义
│   ├── models/           # 模型的封装和训练流程
│   ├── metrics/          # 评估指标 (PSNR, SSIM, etc.)
│   ├── utils/            # 工具函数
│   ├── train.py          # 训练脚本入口
│   └── test.py           # 测试脚本入口
├── Defocus_Deblurring/   # 散焦去模糊任务
├── Denoising/            # 图像去噪任务
├── Deraining/            # 图像去雨任务
├── Motion_Deblurring/    # 运动去模糊任务
│   ├── Datasets/         # 数据集相关说明和下载脚本
│   ├── Options/          # 任务配置文件 (.yml)
│   ├── pretrained_models/# 预训练模型
│   └── README.md         # 任务相关的训练和评估指南
├── demo/                 # 存放用于演示的输入和输出图像
├── datasets/             # 存放下载的数据集
├── INSTALL.md            # 安装指南
├── README.md             # 项目主 README
├── setup.py              # Python 包安装脚本
└── demo.py               # 演示脚本
```

### 2.1 `basicsr`

这是项目的核心库，提供了图像恢复任务所需的基础模块。

* `archs/`: 包含了 `restormer_arch.py` 文件，这是 Restormer 模型结构的核心实现。
* `data/`: 包含各种数据集的加载器，如 `paired_image_dataset.py`, `reds_dataset.py` 等，以及数据预处理的逻辑。
* `losses/`: 定义了项目中使用的损失函数，如 `losses.py` 中的 `CharbonnierLoss`。
* `models/`: 包含了基础模型 `base_model.py` 和图像恢复模型的具体实现 `image_restoration_model.py`。
* `train.py` / `test.py`: 分别是用于模型训练和测试的入口脚本。

### 2.2 任务目录

项目针对不同的图像恢复任务，在 `Defocus_Deblurring`, `Denoising`, `Deraining`, `Motion_Deblurring` 等目录中提供了独立的配置和脚本。每个任务目录下通常包含：

* `Options/`: 存放该任务的 `.yml` 配置文件，用于定义模型参数、训练策略、数据集路径等。
* `Datasets/`: 包含下载和准备该任务所需数据集的说明和脚本。
* `pretrained_models/`: 存放该任务的预训练模型。
* `README.md`: 提供针对该特定任务的详细训练和评估指南。

## 3. 安装

项目的安装依赖 PyTorch 1.8.1 和 Python 3.7。详细步骤请参考 `INSTALL.md` 文件。主要步骤如下：

1. **克隆仓库**: `git clone https://github.com/swz30/Restormer.git`
2. **创建 Conda 环境**:

    ```bash
    conda create -n pytorch181 python=3.7
    conda activate pytorch181
    ```

3. **安装依赖**:

    ```bash
    conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
    pip install -r requirements.txt
    ```

4. **安装 `basicsr`**:

    ```bash
    python setup.py develop --no_cuda_ext
    ```

## 4. 使用方法

### 4.1 演示 (Demo)

项目提供了 `demo.py` 脚本，可以方便地使用预训练模型在自己的图像上进行测试。

**命令格式**:

```bash
python demo.py --task <Task_Name> --input_dir <path_to_images> --result_dir <save_images_here>
```

**示例**:

```bash
# 对单张图片进行散焦去模糊
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'

# 对整个目录的图片进行散焦去模糊
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
```

### 4.2 训练与评估

每个具体的图像恢复任务（如去雨、去噪等）都有独立的训练和评估流程。请参考对应任务目录下的 `README.md` 文件获取详细指南。

例如，要进行图像去雨任务的训练，可以参考 `Deraining/README.md` 中的 "Training" 部分。通常，训练过程通过以下命令启动：

```bash
python basicsr/train.py -opt <path_to_yml_config> --launcher pytorch
```

评估过程也类似，参考相应任务 `README.md` 中的 "Evaluation" 部分。

## 5. 关键文件

* `basicsr/models/archs/restormer_arch.py`: Restormer 模型的核心架构实现。
* `basicsr/train.py`: 模型训练的入口脚本。
* `basicsr/test.py`: 模型测试的入口脚本。
* `demo.py`: 用于快速测试预训练模型的演示脚本。
* `setup.py`: 用于安装 `basicsr` 库的脚本。
* `Deraining/Options/Deraining_Restormer.yml` (及其他任务的 yml 文件): 训练和测试的配置文件，定义了所有实验参数。
