# MACs vs FLOPs 验证与结果汇总（合并版）

## 执行摘要

通过对 `Linear(1, 1)` 层的实际测试，验证了 4 个 FLOPs/MACs 计算库的返回值含义，并梳理了现有脚本中的标记与换算问题。

| 库 | 实际返回 | 现有代码状态 | 是否需要修正 |
|---|---|---|---|
| torchprofile | MACs | ✓ 正确 | 否 |
| thop | MACs | ❌ 标记为 FLOPs | 是 |
| torchinfo | MACs + 加法操作 | ❌ 误标记为 MACs 并 ×2 | 是 |
| fvcore | MACs | ❌ 标记为 FLOPs | 是 |

推荐采用“同时输出 MACs 与 FLOPs”的方式，避免歧义，并统一各脚本的输出口径与换算关系。

---

## 测试方法

使用两种测试用例验证：

**测试1：`Linear(1, 1)` 最小化测试**
- 无偏置：`y = w * x` → 1 次乘法 = 1 MAC = 1 FLOP
- 有偏置：`y = w * x + b` → 1 次乘法 + 1 次加法 = 1 MAC = 2 FLOPs

**测试2：`Linear(10, 5)` 一般性测试**（更能暴露真实行为）
- 无偏置：矩阵乘法 → 50 MACs = 100 FLOPs
- 有偏置：矩阵乘法 + bias加法 → 50 MACs + 5 加法 = 105 FLOPs

---

## 测试结果

### Linear(1, 1) 测试结果

| 库 | bias=False | bias=True | 初步判断 | 代码正确性 |
|---|---|---|---|---|
| torchprofile | 1 | 1 | MACs ✓ | ✓ 正确（`macs * 2 = FLOPs`） |
| thop | 1.0 | 1.0 | MACs ✓ | ❌ 错误（代码标记为 FLOPs） |
| torchinfo | 1 | 2 | 疑似 FLOPs | ❌ 错误（代码中 `total_mult_adds * 2`） |
| fvcore | 1 | 1 | MACs ✓ | ❌ 错误（代码标记为 FLOPs） |

⚠️ **注意**：`Linear(1,1)` 对 torchinfo 来说是特殊情况，`1 MAC + 1 加法 = 2` 恰好等于 `2 FLOPs`，容易误判！

### Linear(10, 5) 验证结果（更准确）

| 库 | bias=False | bias=True | 实际返回 | 确认结论 |
|---|---|---|---|---|
| torchinfo | 50 | **55** | MACs + 加法 ✓ | 50 MACs + 5 加法 ≠ 105 FLOPs |

实测数据（原始呈现）：
```
1. torchprofile
   - Linear(1,1) bias=False: 1  (符合 MACs)
   - Linear(1,1) bias=True:  1  (符合 MACs)
   → 返回 MACs ✓

2. thop
   - Linear(1,1) bias=False: 1.0  (符合 MACs)
   - Linear(1,1) bias=True:  1.0  (符合 MACs，不符合 FLOPs)
   → 返回 MACs ✓

3. torchinfo
   - Linear(1,1) bias=False: 1  (符合 MACs/FLOPs，无法区分)
   - Linear(1,1) bias=True:  2  (看似 FLOPs，但实为 1 MAC + 1 加法)
   - Linear(10,5) bias=False: 50  (符合 MACs)
   - Linear(10,5) bias=True:  55  (= 50 MACs + 5 加法，不符合 105 FLOPs)
   → 返回 MACs + 所有加法操作 ✓

4. fvcore
   - Linear(1,1) bias=False: 1  (符合 MACs)
   - Linear(1,1) bias=True:  1  (符合 MACs，不符合 FLOPs)
   → 返回 MACs ✓
```

---

## 现有代码问题分析

### 1) calc_flops_thop.py ❌

- 问题：将 MACs 误标记为 FLOPs；并错误换算 MACs。
```python
# 第48行 - 错误
flops, params = profile(model, inputs=(input_tensor,), verbose=False)
# 变量名使用 'flops' 但实际是 MACs

# 第56行 - 错误换算
print(f"MACs: {flops/2:,.0f}")
# MACs 应该等于 flops，不是 flops/2
```
- 影响：
  - 报告的 FLOPs 实际是 MACs
  - 报告的 MACs 是真实 MACs 的一半
  - 与他人对比时数值偏小
- 修正方案：
```python
# 正确做法
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
```

---

### 2) calc_flops_torchinfo.py ❌

- 问题：误解了 `total_mult_adds` 的含义，导致错误换算。
```python
# 第56行 - 误解
total_mult_adds = model_stats.total_mult_adds
# 'mult_adds' 字面意思是乘加(MACs)，但实际返回的是 MACs + 所有加法操作

# 第70行 - 错误换算
flops_formatted = format_number(total_mult_adds * 2)
# total_mult_adds 已包含额外加法，再×2会严重夸大

# 第75-76行 - 标记混乱
print(f"MACs: {format_number(total_mult_adds)}")
print(f"FLOPs: {flops_formatted}")
# total_mult_adds 既不是纯 MACs 也不是纯 FLOPs
```
- 影响：
  - 报告的 FLOPs 虚高（对大模型影响显著）
  - 报告的 MACs 包含了额外加法操作，偏大
  - 与其他库结果不一致
- 修正方案：
```python
# 方案1：保守处理（推荐）
# torchinfo 的返回值不适合直接用于 MACs/FLOPs 报告
# 建议改用 torchprofile 或 fvcore

# 方案2：如果必须使用 torchinfo
# 需要理解其返回值 = MACs + 额外加法操作
# 无法简单换算为标准 FLOPs，建议标注为 "Mult-Adds (含额外加法)"
total_mult_adds = model_stats.total_mult_adds
print(f"Mult-Adds (含额外加法): {format_number(total_mult_adds)}")
print(f"注意: 此值不等于标准 MACs 或 FLOPs")
```

---

### 3) calc_flops_fvcore.py ❌

- 问题：与 thop 相同，将 MACs 误标记为 FLOPs，并错误换算。
```python
# 第61行 - 错误命名
flops = flops_analyzer.total()
# 实际是 MACs

# 第73行 - 错误换算
print(f"MACs: {flops/2:,.0f}")
# MACs 应该等于 flops，不是 flops/2
```
- 修正方案：
```python
# 正确做法
macs = flops_analyzer.total()
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
```

---

### 4) calc_flops_torchprofile.py ✓

- 状态：正确 ✓
```python
# 第68行 - 正确
macs = profile_macs(model, input_tensor)

# 第71行 - 正确换算
flops = macs * 2

# 第78-81行 - 正确输出
print(f"MACs: {macs_formatted} ({macs:,})")
print(f"FLOPs: {flops_formatted} ({flops:,})")
```

---

## 统一修正建议

### 方案 A：统一输出 MACs（推荐）

- 理由：
  - 3/4 的库原生返回 MACs
  - 学术界惯例（如 ResNet-50 报“4.1G FLOPs”多为 MACs）
  - 避免混淆
- 修改：
  1. `calc_flops_thop.py`：改为输出 MACs
  2. `calc_flops_torchinfo.py`：⚠️ **不适用**，建议废弃或改用其他库
  3. `calc_flops_fvcore.py`：改为输出 MACs
  4. `calc_flops_torchprofile.py`：保持不变

### 方案 B：统一输出 FLOPs

- 理由：
  - 符合“FLOPs”字面含义
  - 理论定义清晰
- 修改：
  1. `calc_flops_thop.py`：乘以 2 输出 FLOPs
  2. `calc_flops_torchinfo.py`：直接输出，不乘以 2
  3. `calc_flops_fvcore.py`：乘以 2 输出 FLOPs
  4. `calc_flops_torchprofile.py`：保持不变

### 方案 C：同时输出 MACs 和 FLOPs（最安全）

- 优点：
  - 明确两个指标
  - 与任何基准对比都方便
  - 避免歧义
- 统一输出格式：
```python
print(f"MACs: {macs:,.0f}")
print(f"FLOPs: {macs*2:,.0f}")
print(f"换算关系: 1 MAC = 2 FLOPs")
```

---

## 术语澄清与换算公式

### MACs (Multiply-Accumulate Operations)
- 定义：一次乘法和一次加法的复合操作
- 例子：`y = a*x + b` 是 1 MAC
- 用途：更接近硬件实际计算成本

### FLOPs (Floating Point Operations)
- 定义：浮点运算总数
- 例子：`y = a*x + b` 是 2 FLOPs（1 乘 + 1 加）
- 关系：理论上 1 MAC = 2 FLOPs
- 注意：学术论文中的“FLOPs”常指 MACs

### MAdds (Multiply-Adds)
- 定义：理论上与 MACs 相同
- 混淆点：torchinfo 的 `total_mult_adds` 实际返回 **MACs + 所有加法操作**，既非纯 MACs 也非纯 FLOPs
- 实例：`Linear(10, 5, bias=True)` → torchinfo 返回 55 = 50 MACs + 5 bias加法

### 换算公式
```
FLOPs = 2 × MACs
MACs = FLOPs / 2
```

---

## 实际验证命令

```bash
# 运行完整测试（示例）
cd scripts
python test_macs_vs_flops.py

# 查看结果
type test_macs_vs_flops_results.txt
```

---

## 建议行动

1. 立即：阅读本报告，理解各库的实际行为
2. 重要：检查已发表/提交的论文中使用的数值口径
3. 修正：按选定方案修改 4 个计算脚本
4. 验证：重新运行所有脚本，确保数值一致
5. 文档：在论文/仓库中明确说明使用的是 MACs 还是 FLOPs
6. 对比：与 baseline 对比时，确保使用相同标准

---

## 参考文献对比建议

- 明确对方的定义：
  - 查看对方使用的库
  - 确认输出的是 MACs 还是 FLOPs
  - 常见论文的实际值（供对齐）：
    - ResNet-50：约 4.1G（通常指 MACs）
    - MobileNetV2：约 300M（通常指 MACs）
    - Restormer：需基于本次验证重新确认
- 换算检查：
```python
如果对方论文写“X FLOPs”，可能指：
- X MACs（最常见，学术惯例）
- X FLOPs（严格定义，较少见）

验证方法：看其数值是否与已知模型对齐
```

---

## 附录：完整测试日志（test_macs_vs_flops_results.txt）

```text
===============================================================================
测试各库计算的是 MACs 还是 FLOPs
===============================================================================

测试模型: Linear(1, 1)
输入: torch.Size([1, 1])

===============================================================================

1. torchprofile 测试
--------------------------------------------------------------------------------
Linear(1, 1, bias=False):
  profile_macs 返回: 1
  理论 MACs: 1, 理论 FLOPs: 1
  结论: torchprofile 返回 MACs

Linear(1, 1, bias=True):
  profile_macs 返回: 1
  理论 MACs: 1, 理论 FLOPs: 2
  结论: torchprofile 返回 MACs

===============================================================================

2. thop 测试
--------------------------------------------------------------------------------
Linear(1, 1, bias=False):
  thop.profile 返回: 1.0
  理论 MACs: 1, 理论 FLOPs: 1
  结论: thop 返回 MACs

Linear(1, 1, bias=True):
  thop.profile 返回: 1.0
  理论 MACs: 1, 理论 FLOPs: 2
  结论: thop 返回 MACs

===============================================================================

3. torchinfo 测试
--------------------------------------------------------------------------------
Linear(1, 1, bias=False):
  torchinfo.total_mult_adds 返回: 1
  理论 MACs: 1, 理论 FLOPs: 1
  结论: torchinfo 返回 MACs

Linear(1, 1, bias=True):
  torchinfo.total_mult_adds 返回: 2
  理论 MACs: 1, 理论 FLOPs: 2
  结论: torchinfo 返回 FLOPs

===============================================================================

4. fvcore 测试
--------------------------------------------------------------------------------
Linear(1, 1, bias=False):
  fvcore.FlopCountAnalysis 返回: 1
  理论 MACs: 1, 理论 FLOPs: 1
  结论: fvcore 返回 MACs

Linear(1, 1, bias=True):
  fvcore.FlopCountAnalysis 返回: 1
  理论 MACs: 1, 理论 FLOPs: 2
  结论: fvcore 返回 MACs

===============================================================================

总结
===============================================================================

根据测试结果判断：
- 如果 bias=False 返回 1，bias=True 返回 1 → 该库返回 MACs
- 如果 bias=False 返回 1，bias=True 返回 2 → 该库返回 FLOPs

理论基础：
- Linear(1,1, bias=False): y = w*x  → 1乘法 = 1 MAC = 1 FLOP
- Linear(1,1, bias=True):  y = w*x+b → 1乘法+1加法 = 1 MAC = 2 FLOPs
```

---

## 结论

- 关键发现：
  - ✓ torchprofile 正确（唯一无误）
  - ❌ thop 返回 MACs，但代码标记为 FLOPs
  - ❌ torchinfo 返回 **MACs + 额外加法操作**，但代码误当作纯 MACs 并 ×2（严重错误）
  - ❌ fvcore 返回 MACs，但代码标记为 FLOPs
- 影响评估：
  - 中等风险：仓库内数值口径混乱
  - 高风险：若已用于论文对比，可能影响结论
- 推荐方案：
  - 采用“方案 C（同时输出 MACs 与 FLOPs）”，并在文档中明确换算关系，以彻底避免歧义.
