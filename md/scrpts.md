# Restormer LLIE 训练与测试脚本使用指南

本文档包含了 Restormer LLIE 项目的完整训练、测试和指标计算流程说明。

---

## 1. 训练脚本

### PowerShell 方式（推荐）

```powershell
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
$env:CUDA_VISIBLE_DEVICES = "2"

# 启动训练
python basicsr\train.py -opt LLIE\Options\LowLight_Restormer_128_2_60k.yml
```

### CMD 方式

```cmd
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
set CUDA_VISIBLE_DEVICES=2

# 启动训练
python basicsr\train.py -opt LLIE\Options\LowLight_Restormer_128_2_60k.yml
```

### Bash/Linux 方式

```bash
# 设置使用的 GPU 设备编号
export CUDA_VISIBLE_DEVICES=2

# 启动训练
python basicsr/train.py -opt LLIE/Options/LowLight_Restormer_128_2_60k.yml
```

**参数说明：**

- `-opt`: 训练配置文件路径，包含网络结构、数据集路径、训练超参数等设置

**注意事项：**

- 确保数据集已正确放置在配置文件指定的目录
- 根据实际 GPU 数量和显存大小调整 `CUDA_VISIBLE_DEVICES` 和批处理大小
- 训练日志和检查点会保存在 `experiments/` 目录下

---

## 2. 测试脚本

### 基本用法

```bash
# 使用训练好的模型进行推理测试
python LLIE/test.py --input_dir datasets/LOL-v2/Real_captured/Test/Low/ --result_dir results/LowLight_Restormer_128_2_60k_MDTA/ --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_44000.pth --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml
```

### 示例命令

```bash
# 测试 LOL-v2 Real 数据集
python LLIE/test.py --input_dir datasets/LOL-v2/Real_captured/Test/Low/ --result_dir results/LowLight_Restormer_128_2_60k_MDTA_44000 --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_44000.pth --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml

# 测试 LOL-v2 Synthetic 数据集
python LLIE/test.py --input_dir datasets/LOL-v2/Synthetic/Test/Low/ --result_dir results/LOL_v2_Synthetic_Test --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_44000.pth --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml
```

**参数说明：**

- `--input_dir`: 输入的低光照图像目录
- `--result_dir`: 输出结果保存目录（自动创建）
- `--weights`: 训练好的模型权重文件路径
- `--opt`: 配置文件路径（需与训练时使用的配置一致）

**注意事项：**

- 结果目录会自动创建，无需手动创建
- 确认配置文件与训练时使用的配置一致
- 输出图像格式与输入格式保持一致

---

## 3. 指标计算脚本

### 3.1 全参考指标计算 (Full-Reference Metrics)

计算增强结果与真实图像之间的质量指标（需要 Ground Truth）。

#### 基本用法

```bash
# 计算 PSNR、SSIM 等全参考指标
python LLIE/metrics_calc_1.py --dirA datasets/LOL-v2/Real_captured/Test/Normal --dirB results/LowLight_Restormer_128_2_60k_MDTA_44000 --type png --use_gpu
```

#### 示例命令

```bash
# 使用 GPU 加速计算（推荐）
python LLIE/metrics_calc_1.py --dirA datasets/LOL-v2/Real_captured/Test/Normal --dirB results/LowLight_Restormer_128_2_60k_MDTA_44000 --type png --use_gpu

# 使用 CPU 计算
python LLIE/metrics_calc_1.py --dirA datasets/LOL-v2/Real_captured/Test/Normal --dirB results/LowLight_Restormer_128_2_60k_MDTA_44000 --type png
```

**参数说明：**

- `--dirA`: 真实图像（Ground Truth）目录
- `--dirB`: 增强结果图像目录
- `--type`: 图像文件类型（png/jpg/jpeg/bmp 等）
- `--use_gpu`: （可选）使用 GPU 加速计算

**计算指标：**

- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，数值越高越好
- **SSIM** (Structural Similarity Index): 结构相似性指数，数值越高越好（0-1）
- 其他可能的全参考指标

**注意事项：**

- 确保真实图像和增强结果图像文件名一一对应
- 两个目录中的图像数量应该相同
- 建议在测试完成后立即进行指标计算

---

### 3.2 无参考指标计算 (No-Reference Metrics)

计算图像质量的无参考指标（不需要 Ground Truth），适用于评估增强图像的整体质量。

#### 安装依赖

```bash
# 安装 pyiqa 库（用于 NIQE 和 MUSIQ）
pip install pyiqa

# 安装 brisque 库（用于 BRISQUE）
pip install brisque

# 可选：如果下载模型权重较慢，设置 Hugging Face 镜像
# 脚本已自动设置为 https://hf-mirror.com
```

#### 基本用法

```bash
# 计算无参考图像质量指标
python LLIE/metrics_calc_2.py --input_dir results/LowLight_Restormer_128_2_60k_MDTA_44000
```

#### 示例命令

```bash
# 评估增强结果（默认支持 png, jpg, jpeg, bmp）
python LLIE/metrics_calc_2.py --input_dir results/LowLight_Restormer_128_2_60k_MDTA_44000

# 指定特定的图像格式
python LLIE/metrics_calc_2.py --input_dir results/LowLight_Restormer_128_2_60k_MDTA_44000 --extensions png jpg

# 评估原始低光照图像（对比用）
python LLIE/metrics_calc_2.py --input_dir datasets/LOL-v2/Real_captured/Test/Low

# 评估真实图像（参考用）
python LLIE/metrics_calc_2.py --input_dir datasets/LOL-v2/Real_captured/Test/Normal
```

**参数说明：**

- `--input_dir`: 待评估的图像目录（必需）
- `--extensions`: 图像文件扩展名列表（可选，默认：png jpg jpeg bmp）

**计算指标：**

- **NIQE** (Natural Image Quality Evaluator): 
  - 范围：0-100（实际可能更大）
  - **数值越低越好**
  - 基于自然场景统计特征评估图像质量
  
- **MUSIQ** (Multi-Scale Image Quality Transformer):
  - 范围：0-100
  - **数值越高越好**
  - 基于深度学习的多尺度图像质量评估
  
- **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator):
  - 范围：0-100
  - **数值越低越好**
  - 基于自然场景统计和失真特征

**输出结果：**

脚本会在指定的图像目录下生成 `no_reference_metrics.txt` 文件，包含：
- 每张图像的详细指标
- 所有指标的统计信息：
  - 平均值 ± 标准差
  - 中位数
  - 最小值和最大值

**注意事项：**

- 首次运行会自动下载预训练模型权重（NIQE 和 MUSIQ）
- 脚本已配置使用 Hugging Face 镜像加速下载
- 无参考指标不需要 Ground Truth，可单独对任何图像集进行评估
- 建议同时计算低光照图像、增强结果和真实图像的无参考指标进行对比

---

## 4. 完整工作流程

### 步骤 1: 训练模型

```powershell
# PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
python basicsr\train.py -opt LLIE\Options\LowLight_Restormer_128_2_60k.yml
```

### 步骤 2: 测试模型

```bash
python LLIE/test.py --input_dir datasets/LOL-v2/Real_captured/Test/Low/ --result_dir results/LowLight_Restormer_128_2_60k_MDTA_44000_60k --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_60000.pth --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml
```

### 步骤 3: 计算全参考指标

```bash
python LLIE/metrics_calc_1.py --dirA datasets/LOL-v2/Real_captured/Test/Normal --dirB results/LowLight_Restormer_128_2_60k_MDTA_44000_60k --type png --use_gpu
```

### 步骤 4: 计算无参考指标

```bash
# 评估增强结果
python LLIE/metrics_calc_2.py --input_dir results/LowLight_Restormer_128_2_60k_MDTA_44000_60k

# 对比：评估原始低光照图像
python LLIE/metrics_calc_2.py --input_dir datasets/LOL-v2/Real_captured/Test/Low

# 参考：评估真实图像
python LLIE/metrics_calc_2.py --input_dir datasets/LOL-v2/Real_captured/Test/Normal
```

---

## 5. 常见问题

### Q1: 训练时显存不足怎么办？

- 减小配置文件中的 `batch_size`
- 减小图像块大小 `gt_size`
- 使用更小的网络配置

### Q2: 测试时找不到模型权重？

- 检查 `--weights` 参数路径是否正确
- 确认训练已完成并保存了检查点
- 查看 `experiments/` 目录下的模型文件

### Q3: 指标计算时文件数量不匹配？

- 确保两个目录中的图像文件名完全一致
- 检查图像文件扩展名是否正确
- 确认没有隐藏文件或系统文件

### Q4: NIQE/MUSIQ/BRISQUE 计算失败？

- 确认已安装 `pyiqa` 和 `brisque` 库
- 检查网络连接，首次运行需要下载模型权重
- 如果下载慢，确认脚本已设置 HF_ENDPOINT 镜像

### Q5: 如何选择最佳检查点？

- 查看训练日志中的验证指标
- 测试多个检查点（如 40k, 50k, 60k）
- 综合考虑全参考和无参考指标

---

## 6. 目录结构参考

```
Restormer_LLIE/
├── datasets/
│   └── LOL-v2/
│       ├── Real_captured/
│       │   ├── Train/
│       │   │   ├── Low/
│       │   │   └── Normal/
│       │   └── Test/
│       │       ├── Low/
│       │       └── Normal/
│       └── Synthetic/
│           ├── Train/
│           └── Test/
├── LLIE/
│   ├── Options/
│   │   └── LowLight_Restormer_128_2_60k.yml
│   ├── test.py
│   ├── metrics_calc_1.py  # 全参考指标
│   └── metrics_calc_2.py  # 无参考指标
├── experiments/
│   └── LowLight_Restormer_128_2_60k_MDTA/
│       ├── models/
│       │   ├── net_g_20000.pth
│       │   ├── net_g_40000.pth
│       │   └── net_g_60000.pth
│       └── training_states/
└── results/
    └── LowLight_Restormer_128_2_60k_MDTA_44000/
```
