# PSNR计算方法统一化更新

## 问题描述

训练时计算validation PSNR的方法与测试集使用`metrics_cal.py`计算PSNR的方法不一致，导致指标对比存在差异。

### 原始实现差异

1. **训练时**（`basicsr/metrics/psnr_ssim.py`）：
   - 使用自定义实现：`20. * np.log10(max_value / np.sqrt(mse))`
   - 配置文件中设置 `test_y_channel: true`，在YCbCr色彩空间的Y通道计算
   
2. **测试时**（`LLIE/metrics_cal.py`）：
   - 使用 `skimage.metrics.peak_signal_noise_ratio`
   - 直接在RGB色彩空间计算

## 解决方案

### 1. 修改 `basicsr/metrics/psnr_ssim.py`

将 `calculate_psnr` 函数中的自定义PSNR计算替换为使用 `skimage.metrics.peak_signal_noise_ratio`：

```python
# 旧代码（已删除）:
mse = np.mean((img1 - img2)**2)
if mse == 0:
    return float('inf')
max_value = 1. if img1.max() <= 1 else 255.
return 20. * np.log10(max_value / np.sqrt(mse))

# 新代码:
# 使用 skimage.metrics.peak_signal_noise_ratio，与 metrics_cal.py 保持一致
# 自动检测数据范围：如果最大值 <= 1 则为 [0, 1]，否则为 [0, 255]
max_value = 1.0 if img1.max() <= 1 else 255.0
psnr_val = skimage.metrics.peak_signal_noise_ratio(img1, img2, data_range=max_value)
return psnr_val
```

### 2. 更新配置文件

将所有训练配置文件中的 `test_y_channel` 参数从 `true` 改为 `false`，确保在RGB空间计算PSNR：

#### 修改的文件列表：

1. `LLIE/Options/LowLight_Restormer.yml`
2. `LLIE/Options/LowLight_Restormer_128_2.yml`
3. `LLIE/Options/LowLight_Restormer_128_2_60k.yml`
4. `LLIE/Options/LowLight_Restormer_pro.yml`

#### 修改内容：

```yaml
# 旧配置:
val:
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: true

# 新配置:
val:
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false  # 与 metrics_cal.py 保持一致，在RGB空间计算
```

## 验证测试

运行 `test_psnr_simple.py` 进行验证：

```bash
python test_psnr_simple.py
```

### 测试结果

所有测试通过，确认：
- skimage实现与标准PSNR公式一致
- 训练时和测试时的PSNR计算完全一致
- 支持 uint8 (0-255) 和 float (0-1) 两种图像格式

```
[SUCCESS] 所有测试通过！

关键点：
1. skimage 的实现与标准公式一致
2. 修改后的 calculate_psnr 使用 skimage 实现
3. 训练时和测试时(metrics_cal.py)的PSNR计算现在完全一致
```

## 影响说明

### 对现有模型的影响

1. **训练过程**：
   - 新训练的模型validation PSNR值将与测试集PSNR保持一致
   - 可能会看到PSNR数值略有变化（RGB vs Y通道）
   
2. **已训练模型**：
   - 不影响已训练模型的推理性能
   - 但重新评估时可能得到不同的PSNR值（因为计算方法改变）

3. **数值差异**：
   - RGB空间PSNR通常会略高于Y通道PSNR
   - 这是正常现象，因为RGB考虑了所有颜色通道

## 建议

1. **新训练**：使用更新后的配置开始新的训练
2. **已有实验**：记录PSNR计算方法的变更，避免混淆
3. **文档更新**：在论文或报告中明确说明PSNR是在RGB空间计算的

## 总结

通过这次更新，训练时和测试时的PSNR计算方法已经完全统一，使用相同的 `skimage.metrics.peak_signal_noise_ratio` 实现，在RGB色彩空间进行计算，确保了指标的一致性和可比性。

