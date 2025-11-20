# 无监督图像质量评估工具

这个工具用于计算图像的无参考（no-reference）质量评估指标，主要包括 **NIQE** 和 **LPIPS** 两个指标。

## 📋 支持的指标

### 1. NIQE (Natural Image Quality Evaluator)
- **说明**: 自然图像质量评估器，基于自然场景统计模型
- **范围**: 通常在 0-10 之间，**越低越好**
- **特点**: 完全无参考，不需要对比原始图像
- **用途**: 评估图像的自然度和质量

### 2. LPIPS (Learned Perceptual Image Patch Similarity)
- **说明**: 基于深度学习的感知图像质量评估
- **计算方式**: 与轻微模糊版本对比，评估感知质量稳定性
- **范围**: 通常在 0-1 之间，**越低越好**
- **特点**: 反映人类感知的图像质量
- **用途**: 评估图像增强后的感知质量

## 🚀 使用方法

### 基本用法

```bash
# 评估单个目录中的图像（CPU 模式）
python unsupervised_metrics_cal.py --dir /path/to/images

# 使用 GPU 加速（推荐）
python unsupervised_metrics_cal.py --dir /path/to/images --use_gpu

# 指定图像格式
python unsupervised_metrics_cal.py --dir /path/to/images --type jpg --use_gpu

# 保存结果到文本文件
python unsupervised_metrics_cal.py --dir /path/to/images --use_gpu --save_txt results.txt
```

### 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--dir` | 图像目录路径（必需） | - | `--dir ./results/enhanced` |
| `--type` | 图像文件扩展名 | `png` | `--type jpg` |
| `--use_gpu` | 是否使用 GPU 加速 | `False` | `--use_gpu` |
| `--save_txt` | 保存结果的文本文件路径 | `None` | `--save_txt metrics.txt` |

## 📊 输出示例

### 终端输出
```
================================================================================
无监督图像质量评估
================================================================================
图像目录: ./results/enhanced
图像格式: *.png
使用 GPU: True
================================================================================

找到 100 张图像
Filename                                      NIQE     LPIPS   Time(s)
--------------------------------------------------------------------------------
image_001.png                              4.5123   0.0234     0.12
image_002.png                              4.6789   0.0198     0.11
...
--------------------------------------------------------------------------------
Average                                    4.5891   0.0216     12.5

成功处理 100 张图像，总用时 12.5s

指标说明:
  NIQE: 自然图像质量评估器（越低越好，通常范围 0-10）
  LPIPS: 感知图像相似度（越低表示感知质量越稳定）

结果已保存到: metrics.txt
```

### 保存的文本文件格式
```
Filename                                      NIQE     LPIPS   Time(s)
--------------------------------------------------------------------------------
image_001.png                              4.5123   0.0234     0.12
image_002.png                              4.6789   0.0198     0.11
...
--------------------------------------------------------------------------------
Average                                    4.5891   0.0216     12.5

成功处理 100 张图像，总用时 12.5s

指标说明:
  NIQE: 自然图像质量评估器（越低越好，通常范围 0-10）
  LPIPS: 感知图像相似度（越低表示感知质量越稳定）
```

## 🔧 实际应用场景

### 1. 评估低光增强结果
```bash
# 评估 LOL-v2 数据集的增强结果
python unsupervised_metrics_cal.py \
    --dir ./results/LOL-v2/Real_captured \
    --type png \
    --use_gpu \
    --save_txt ./results/unsupervised_metrics.txt
```

### 2. 批量评估多个模型结果
```bash
# 模型 A
python unsupervised_metrics_cal.py --dir ./results/model_A --use_gpu --save_txt ./results/model_A_metrics.txt

# 模型 B
python unsupervised_metrics_cal.py --dir ./results/model_B --use_gpu --save_txt ./results/model_B_metrics.txt
```

### 3. 对比不同训练 epoch 的结果
```bash
# Epoch 10
python unsupervised_metrics_cal.py --dir ./results/epoch_10 --use_gpu --save_txt ./metrics_epoch10.txt

# Epoch 20
python unsupervised_metrics_cal.py --dir ./results/epoch_20 --use_gpu --save_txt ./metrics_epoch20.txt
```

## 📦 依赖项

确保已安装以下 Python 包：

```bash
pip install numpy opencv-python torch lpips tqdm natsort scipy
```

或者使用项目的 requirements.txt：
```bash
pip install -r ../scripts/requirements.txt
```

## 💡 注意事项

1. **GPU 加速**: 强烈推荐使用 `--use_gpu` 参数，可以显著提升 LPIPS 计算速度
2. **图像格式**: 确保目录中的图像格式一致（都是 png 或都是 jpg）
3. **NIQE 参数**: 使用的是 basicsr 库中预训练的 NIQE 参数文件
4. **LPIPS 网络**: 默认使用 AlexNet，可以修改代码使用 VGG 或 SqueezeNet
5. **内存占用**: 处理大量高分辨率图像时注意内存使用

## 🔬 指标解读

### NIQE 分数参考
- **< 3.0**: 非常好的图像质量
- **3.0 - 4.0**: 良好的图像质量
- **4.0 - 5.0**: 可接受的图像质量
- **> 5.0**: 较差的图像质量

### LPIPS 分数参考
- **< 0.02**: 感知质量非常稳定
- **0.02 - 0.05**: 感知质量稳定
- **0.05 - 0.10**: 感知质量一般
- **> 0.10**: 感知质量较差

## 📚 参考文献

1. **NIQE**: Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). Making a "completely blind" image quality analyzer. *IEEE Signal processing letters*, 20(3), 209-212.

2. **LPIPS**: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *CVPR*.

## 🤝 与其他工具的关系

- **`metrics_cal.py`**: 计算有监督指标（PSNR, SSIM, LPIPS），需要参考图像
- **`unsupervised_metrics_cal.py`**: 计算无监督指标（NIQE, LPIPS），只需要增强后的图像

推荐同时使用两个工具进行全面评估：
- 有参考图像时：使用 `metrics_cal.py` 计算 PSNR/SSIM
- 无参考图像时：使用 `unsupervised_metrics_cal.py` 计算 NIQE
- 综合评估：结合两者的结果

## 📝 示例完整工作流

```bash
# 1. 使用模型进行图像增强
python test.py --opt Options/LowLight_Restormer.yml

# 2. 评估有监督指标（如果有 GT）
python metrics_cal.py \
    --dirA ./datasets/LOL-v2/Real_captured/Test/Normal \
    --dirB ./results/enhanced \
    --use_gpu \
    --save_txt supervised_metrics.txt

# 3. 评估无监督指标
python unsupervised_metrics_cal.py \
    --dir ./results/enhanced \
    --use_gpu \
    --save_txt unsupervised_metrics.txt

# 4. 分析结果
cat supervised_metrics.txt unsupervised_metrics.txt
```

## 🐛 常见问题

### Q: NIQE 计算时报错找不到参数文件
**A**: 确保运行脚本时，工作目录能够正确访问 `basicsr/metrics/niqe_pris_params.npz` 文件。建议在项目根目录运行脚本。

### Q: GPU 模式下内存不足
**A**: 可以尝试：
1. 关闭其他占用 GPU 的程序
2. 使用 CPU 模式（去掉 `--use_gpu` 参数）
3. 减小图像分辨率

### Q: LPIPS 分数都是 NaN
**A**: 可能原因：
1. 图像读取失败
2. 图像格式不正确
3. 检查图像路径和格式设置

---

**作者**: Restormer LLIE Project  
**更新日期**: 2024-11  
**版本**: 1.0
