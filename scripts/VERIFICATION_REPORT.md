# FLOPs 计算库验证报告

## 执行摘要

通过 `Linear(1, 1)` 层的实际测试，验证了4个库的返回值类型：

| 库 | 实际返回 | 现有代码状态 | 需要修正 |
|---|---|---|---|
| torchprofile | MACs | ✓ 正确 | 否 |
| **thop** | **MACs** | ❌ **标记为FLOPs** | **是** |
| **torchinfo** | **FLOPs** | ❌ **标记为MACs并×2** | **是** |
| **fvcore** | **MACs** | ❌ **标记为FLOPs** | **是** |

## 详细测试结果

### 测试用例
```python
# 测试1：无偏置 Linear(1, 1, bias=False)
# y = w * x
# 理论值：1 MAC = 1 FLOP

# 测试2：有偏置 Linear(1, 1, bias=True)  
# y = w * x + b
# 理论值：1 MAC = 2 FLOPs
```

### 实测数据

```
1. torchprofile
   - bias=False: 1  (符合 MACs)
   - bias=True:  1  (符合 MACs)
   → 返回 MACs ✓

2. thop
   - bias=False: 1.0  (符合 MACs)
   - bias=True:  1.0  (符合 MACs，不符合 FLOPs)
   → 返回 MACs ✓

3. torchinfo
   - bias=False: 1  (符合 MACs/FLOPs)
   - bias=True:  2  (符合 FLOPs，不符合 MACs)
   → 返回 FLOPs ✓

4. fvcore
   - bias=False: 1  (符合 MACs)
   - bias=True:  1  (符合 MACs，不符合 FLOPs)
   → 返回 MACs ✓
```

## 现有代码问题分析

### 1. calc_flops_thop.py ❌

**问题**：将 MACs 误标记为 FLOPs
```python
# 第48行 - 错误
flops, params = profile(model, inputs=(input_tensor,), verbose=False)
# 变量名使用 'flops' 但实际是 MACs

# 第56行 - 错误换算
print(f"MACs: {flops/2:,.0f}")
# MACs 应该等于 flops，不是 flops/2
```

**影响**：
- 报告的 FLOPs 实际是 MACs
- 报告的 MACs 是真实 MACs 的一半
- 导致与其他论文对比时数值偏小

**修正方案**：
```python
# 正确做法
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
```

---

### 2. calc_flops_torchinfo.py ❌

**问题**：将 FLOPs 误标记为 MACs 并乘以2
```python
# 第56行 - 误解
total_mult_adds = model_stats.total_mult_adds
# 'mult_adds' 字面意思是乘加运算，但实际返回的是 FLOPs

# 第70行 - 错误换算
flops_formatted = format_number(total_mult_adds * 2)
# 已经是 FLOPs，再乘以2会翻倍

# 第75-76行 - 标记混乱
print(f"MACs: {format_number(total_mult_adds)}")
print(f"FLOPs: {flops_formatted}")
# total_mult_adds 已经是 FLOPs
```

**影响**：
- 报告的 FLOPs 是真实值的2倍
- 报告的 MACs 实际是 FLOPs
- 与其他库的结果不一致

**修正方案**：
```python
# 正确做法
total_flops = model_stats.total_mult_adds  # 这就是 FLOPs
print(f"FLOPs: {format_number(total_flops)}")
print(f"MACs: {format_number(total_flops / 2)}")
```

---

### 3. calc_flops_fvcore.py ❌

**问题**：与 thop 相同，将 MACs 误标记为 FLOPs
```python
# 第61行 - 错误命名
flops = flops_analyzer.total()
# 实际是 MACs

# 第73行 - 错误换算  
print(f"MACs: {flops/2:,.0f}")
# MACs 应该等于 flops，不是 flops/2
```

**影响**：同 thop

**修正方案**：
```python
# 正确做法
macs = flops_analyzer.total()
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
```

---

### 4. calc_flops_torchprofile.py ✓

**状态**：正确 ✓

```python
# 第68行 - 正确
macs = profile_macs(model, input_tensor)

# 第71行 - 正确换算
flops = macs * 2

# 第78-81行 - 正确输出
print(f"MACs: {macs_formatted} ({macs:,})")
print(f"FLOPs: {flops_formatted} ({flops:,})")
```

## 统一修正建议

### 方案A：统一输出 MACs（推荐）

理由：
- 3/4 的库原生返回 MACs
- 学术界惯例（ResNet-50 的 "4.1G FLOPs" 实际是 MACs）
- 避免混淆

修改：
1. `calc_flops_thop.py`: 改为输出 MACs
2. `calc_flops_torchinfo.py`: 改为输出 MACs（除以2）
3. `calc_flops_fvcore.py`: 改为输出 MACs
4. `calc_flops_torchprofile.py`: 保持不变（已正确）

### 方案B：统一输出 FLOPs

理由：
- 符合 "FLOPs" 字面含义
- 理论定义清晰

修改：
1. `calc_flops_thop.py`: 乘以2输出 FLOPs
2. `calc_flops_torchinfo.py`: 直接输出，不乘以2
3. `calc_flops_fvcore.py`: 乘以2输出 FLOPs
4. `calc_flops_torchprofile.py`: 保持不变（已正确）

### 方案C：同时输出 MACs 和 FLOPs（最安全）

优点：
- 明确两个指标
- 避免歧义
- 与任何基准对比都方便

所有脚本统一格式：
```python
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
print(f"换算关系: 1 MAC = 2 FLOPs")
```

## 术语澄清

### MACs (Multiply-Accumulate Operations)
- **定义**：一次乘法和一次加法的复合操作
- **例子**：`y = a*x + b` 是 1 MAC
- **用途**：硬件效率评估，更接近实际计算成本
- **常见于**：CNN、Transformer 层的计算量评估

### FLOPs (Floating Point Operations)  
- **定义**：浮点运算总数
- **例子**：`y = a*x + b` 是 2 FLOPs（1乘+1加）
- **关系**：理论上 1 MAC = 2 FLOPs
- **注意**：学术论文中的 "FLOPs" 常指 MACs

### MAdds (Multiply-Adds)
- **定义**：与 MACs 相同
- **混淆点**：torchinfo 的 `total_mult_adds` 实际返回 FLOPs，不是 MACs

## 实际验证命令

```bash
# 运行完整测试
cd scripts
python test_macs_vs_flops.py

# 查看结果
type test_macs_vs_flops_results.txt
```

## 建议行动

1. ✅ **立即**: 阅读此报告，理解各库的实际行为
2. ⚠️ **重要**: 检查已发表/提交的论文中使用的数值
3. 🔧 **修正**: 根据选定方案修改4个计算脚本
4. 📊 **验证**: 重新运行所有脚本，确保数值一致
5. 📝 **文档**: 在论文中明确说明使用的是 MACs 还是 FLOPs
6. 🔍 **对比**: 重新检查与 baseline 的对比，确保使用相同标准

## 参考文献对比建议

在对比其他论文的 FLOPs/MACs 时：

1. **明确对方的定义**
   - 查看对方使用的库
   - 确认是 MACs 还是 FLOPs

2. **常见论文的实际值**
   - ResNet-50: 约 4.1G (通常指 MACs)
   - MobileNetV2: 约 300M (通常指 MACs)
   - Restormer: 需要根据本次验证重新确认

3. **换算检查**
   ```
   如果对方论文说 "X FLOPs"，可能指：
   - X MACs (最常见，学术惯例)
   - X FLOPs (严格定义，较少见)
   
   验证方法：看对方的数值是否与已知模型对齐
   ```

## 结论

**关键发现**：
- ✓ torchprofile 正确（唯一无误）
- ❌ thop 返回 MACs 但代码标记为 FLOPs
- ❌ torchinfo 返回 FLOPs 但代码标记为 MACs 并×2
- ❌ fvcore 返回 MACs 但代码标记为 FLOPs

**影响评估**：
- 中等风险：现有代码中的数值混乱
- 高风险：如果已用于论文对比，可能结论有误

**推荐方案**：
采用 **方案C**（同时输出 MACs 和 FLOPs），避免一切歧义。
