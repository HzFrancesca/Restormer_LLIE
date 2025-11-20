# MACs vs FLOPs 测试结果总结

## 测试方法
使用简单的 Linear(1, 1) 层进行测试：
- **无偏置**: `y = w * x` → 1次乘法 = 1 MAC = 1 FLOP
- **有偏置**: `y = w * x + b` → 1次乘法 + 1次加法 = 1 MAC = 2 FLOPs

## 测试结果

| 库 | bias=False | bias=True | **实际返回** | 代码正确性 |
|---|---|---|---|---|
| **torchprofile** | 1 | 1 | **MACs** ✓ | ✓ 正确（代码中 `macs * 2 = FLOPs`） |
| **thop** | 1.0 | 1.0 | **MACs** ✓ | ❌ **错误**（代码中写的是 FLOPs） |
| **torchinfo** | 1 | **2** | **FLOPs** ✓ | ❌ **错误**（代码中 `total_mult_adds * 2`） |
| **fvcore** | 1 | 1 | **MACs** ✓ | ❌ **错误**（代码中写的是 FLOPs） |

## 关键发现

### ✓ 正确的库
1. **torchprofile** - 返回 MACs，代码处理正确
   - `profile_macs()` → MACs
   - `macs * 2` → FLOPs

### ❌ 错误的库使用

2. **thop** - 返回 MACs，但代码标记为 FLOPs
   ```python
   # 当前代码（错误）
   flops, params = profile(model, inputs=(input_tensor,))
   print(f"FLOPs: {flops}")      # ❌ 实际是 MACs
   print(f"MACs: {flops/2}")     # ❌ 错误换算
   
   # 应该改为
   macs, params = profile(model, inputs=(input_tensor,))
   print(f"MACs: {macs}")        # ✓ 正确
   print(f"FLOPs: {macs*2}")     # ✓ 正确换算
   ```

3. **torchinfo** - 返回 FLOPs，但代码标记为 MACs 并乘以2
   ```python
   # 当前代码（错误）
   total_mult_adds = model_stats.total_mult_adds  # 实际是 FLOPs
   flops = total_mult_adds * 2   # ❌ 错误！double counting
   
   # 应该改为
   total_flops = model_stats.total_mult_adds  # 实际就是 FLOPs
   flops = total_flops           # ✓ 正确
   macs = total_flops / 2        # ✓ 正确换算
   ```

4. **fvcore** - 返回 MACs，但代码标记为 FLOPs
   ```python
   # 当前代码（错误）
   flops = flops_analyzer.total()
   print(f"FLOPs: {flops}")      # ❌ 实际是 MACs
   print(f"MACs: {flops/2}")     # ❌ 错误换算
   
   # 应该改为
   macs = flops_analyzer.total()
   print(f"MACs: {macs}")        # ✓ 正确
   print(f"FLOPs: {macs*2}")     # ✓ 正确换算
   ```

## 理论基础

### MACs (Multiply-Accumulate Operations)
- 一次乘法 + 一次加法 = 1 MAC
- 硬件层面的运算单位
- 例如：`y = w * x + b` = 1 MAC

### FLOPs (Floating Point Operations)
- 每个浮点运算都算1 FLOP
- 理论上：1 MAC = 2 FLOPs（1乘 + 1加）
- 例如：`y = w * x + b` = 2 FLOPs

## 换算公式
```
FLOPs = 2 × MACs
MACs = FLOPs / 2
```

## 建议

1. **统一使用 MACs** 作为主要指标，因为：
   - 3/4 的库返回 MACs
   - 学术界常用（ResNet-50 的 "4.1G FLOPs" 实际是 4.1G MACs）
   - 更贴近硬件实现

2. **修正现有代码** 中的错误标记和换算

3. **对比基准时** 确认对方使用的是 MACs 还是 FLOPs
