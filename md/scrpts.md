# Restormer LLIE 训练与测试脚本

## 1. 训练脚本

### PowerShell 方式（推荐）

```powershell
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
$env:CUDA_VISIBLE_DEVICES = "2"

# 启动训练
python basicsr\train.py -opt LLIE\Options\LowLight_Restormer_128_2.yml
```

### CMD 方式

```cmd
# 设置使用的 GPU 设备编号（例如：使用第 2 块 GPU）
set CUDA_VISIBLE_DEVICES=2

# 启动训练
python basicsr\train.py -opt LLIE\Options\LowLight_Restormer_128_2.yml
```

---

## 2. 测试脚本

```bash
# 使用训练好的模型进行推理测试
python LLIE/test.py \
    --input_dir datasets/LOL-v2/Real_captured/Test/Low/ \
    --result_dir results/LowLight_Restormer_128_2_60k/ \
    --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_60000.pth \
    --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml
```

**参数说明：**

- `--input_dir`: 输入的低光照图像目录
- `--result_dir`: 输出结果保存目录
- `--weights`: 训练好的模型权重文件路径
- `--opt`: 配置文件路径

---

## 3. 指标计算脚本

```bash
# 计算增强结果与真实图像之间的质量指标（PSNR、SSIM 等）
python LLIE/metrics_cal.py \
    --dirA datasets/LOL-v2/Real_captured/Test/Normal \
    --dirB results/LowLight_Restormer_128_2_60k \
    --type png \
    --use_gpu
```

**参数说明：**

- `--dirA`: 真实图像（Ground Truth）目录
- `--dirB`: 增强结果图像目录
- `--type`: 图像文件类型（png/jpg 等）
- `--use_gpu`: 使用 GPU 加速计算

---

## 使用说明

1. **训练前准备：**
   - 确保数据集已正确放置在指定目录
   - 检查配置文件路径是否正确
   - 根据实际 GPU 数量调整 `CUDA_VISIBLE_DEVICES`

2. **测试前准备：**
   - 确保已训练好模型并保存权重文件
   - 检查输入目录和输出目录路径
   - 确认配置文件与训练时使用的配置一致

3. **指标计算：**
   - 确保真实图像和增强结果图像文件名一一对应
   - 建议在测试完成后立即进行指标计算
