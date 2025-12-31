# 技术栈

## 语言与框架
- Python 3.7+
- PyTorch（支持 CUDA）
- BasicSR 框架（自定义分支）

## 主要依赖
- `torch`, `torchvision` - 深度学习框架
- `einops` - 张量操作（rearrange 模式）
- `opencv-python` (cv2) - 图像读写和处理
- `numpy`, `scipy` - 数值计算
- `lpips` - 感知相似度指标
- `transformers` - 可选，用于加载本地 DINO 模型
- `tqdm` - 进度条
- `yaml` - 配置解析
- `tensorboard` / `wandb` - 训练日志

## 构建与安装

```bash
# 开发模式安装（不编译 CUDA 扩展）
python setup.py develop --no_cuda_ext

# 安装 CUDA 扩展（需要 CUDA 工具包）
python setup.py develop
```

## 常用命令

### 训练
```bash
# 单 GPU 训练
python basicsr/train.py -opt LLIE/Options/LowLight_Restormer.yml

# 多 GPU 分布式训练
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt <config.yml> --launcher pytorch
```

### 测试/推理
```bash
python LLIE/test.py -opt LLIE/Options/LowLight_Restormer.yml --weights <模型路径.pth>
```

### 指标评估
```bash
# 有参考指标 (PSNR, SSIM, LPIPS)
python LLIE/metrics_cal.py --dirA ./ground_truth --dirB ./enhanced --use_gpu

# 无参考指标 (NIQE)
python LLIE/unsupervised_metrics_cal.py --dir ./enhanced --use_gpu
```

### FLOPs 计算
```bash
# 运行所有 FLOPs 计算器
powershell scripts/flops/run_all_flops_calc.ps1
```

## 配置说明
训练/测试配置为 `LLIE/Options/` 中的 YAML 文件。关键设置：
- `network_g.type`: 架构类名（如 `Restormer`、`DINOGuidedRestormer`）
- `datasets`: 训练/验证数据路径和数据增强设置
- `train.scheduler`: 学习率调度（CosineAnnealingRestartCyclicLR）
- `val.metrics`: 验证指标配置
