# FLOPs/MACs 计算库验证测试

## 概述

本目录包含用于验证不同FLOPs/MACs计算库返回值含义的测试脚本。使用简单的线性层模型进行理论验证。

## 测试文件

### 1. `test_fvcore_macs_flops.py`
验证 **fvcore** 库返回的是MACs还是FLOPs。

**关键测试点**：
- 简单线性层 (10→5, 有/无bias)
- 对比理论MACs和理论FLOPs
- 验证bias操作是否被计入

**运行**：
```bash
python scripts/test_fvcore_macs_flops.py
```

### 2. `test_thop_macs_flops.py`
验证 **thop** 库返回的是MACs还是FLOPs。

**关键测试点**：
- 简单线性层 (10→5, 有/无bias)
- 验证thop.profile()返回的macs字段含义
- 对比有/无bias的差异

**运行**：
```bash
python scripts/test_thop_macs_flops.py
```

### 3. `test_torchprofile_macs_flops.py`
验证 **torchprofile** 库返回的是MACs还是FLOPs。

**关键测试点**：
- 简单线性层 (10→5, 有/无bias)
- 验证profile_macs()函数返回值含义
- 与thop和fvcore的对比

**运行**：
```bash
python scripts/test_torchprofile_macs_flops.py
```

### 4. `test_torchinfo_detailed.py`
详细测试 **torchinfo** 库的各种统计字段。

**关键测试点**：
- total_params（参数总数）
- total_mult_adds（计算量）
- 各字段的详细含义
- 与其他库的对比

**运行**：
```bash
python scripts/test_torchinfo_detailed.py
```

### 5. `run_all_lib_tests.ps1`
批量运行所有测试的PowerShell脚本。

**运行**：
```powershell
.\scripts\run_all_lib_tests.ps1
```

## 理论基础

### 线性层计算量

对于线性层 `y = Wx + b`：
- 输入: `(batch, in_features)`
- 权重: `(out_features, in_features)`
- 输出: `(batch, out_features)`

#### MACs（Multiply-Accumulate Operations）
- 矩阵乘法: `batch × out_features × in_features`
- 1个MAC = 1次乘法 + 1次加法

#### FLOPs（Floating Point Operations）
- 矩阵乘法: `2 × (batch × out_features × in_features)`
- bias加法: `batch × out_features`
- 总计: 矩阵乘法FLOPs + bias FLOPs

### 测试案例

**配置**：
- `in_features = 10`
- `out_features = 5`
- `batch_size = 1`

**理论值**：
- MACs (矩阵乘法): `1 × 5 × 10 = 50`
- FLOPs (矩阵乘法): `2 × 50 = 100`
- FLOPs (bias): `1 × 5 = 5`
- 总FLOPs: `105`

## 预期结果总结

| 库 | 返回值类型 | bias处理 | 备注 |
|---|---|---|---|
| **fvcore** | MACs | 通常不计入 | 仅矩阵乘法的MACs |
| **thop** | MACs | 可能计入 | 返回值名为macs |
| **torchprofile** | MACs | 通常不计入 | 函数名profile_macs，返回MACs |
| **torchinfo** | MACs + 加法 | 计入 | mult_adds包含所有运算 |

## 关键概念

### MACs vs FLOPs

**MACs (Multiply-Accumulate)**:
```
z = a * b + c  # 这是1个MAC
```
- 包含1次乘法和1次加法
- 通常用于硬件计算复杂度

**FLOPs (Floating Point Operations)**:
```
z = a * b      # 1 FLOP
z = z + c      # 1 FLOP
# 总计 2 FLOPs
```
- 每个浮点运算单独计数
- 关系: `1 MAC = 2 FLOPs`（在大多数定义中）

### 不同库的统计差异

1. **fvcore (Facebook)**
   - 通常只统计核心计算操作
   - bias等辅助操作可能不计入
   - 返回MACs

2. **thop (PyTorch-OpCounter)**
   - 返回值命名为"macs"
   - 可能包含bias等操作
   - 与fvcore可能有细微差异

3. **torchprofile**
   - 函数名为`profile_macs`
   - 专注于MACs统计
   - 通常不计入bias等纯加法操作
   - 与fvcore结果通常一致

4. **torchinfo**
   - 字段名为`mult_adds`
   - 包含MACs + 所有加法操作
   - 统计最全面，但可能与传统定义有出入

## 使用建议

1. **论文写作**：明确说明使用的工具和统计口径
2. **对比实验**：使用同一工具保持一致性
3. **转换关系**：了解MACs和FLOPs的转换关系
4. **验证方法**：对简单模型进行理论验证

## 参考资料

- [fvcore文档](https://github.com/facebookresearch/fvcore)
- [thop文档](https://github.com/Lyken17/pytorch-OpCounter)
- [torchprofile文档](https://github.com/zhijian-liu/torchprofile)
- [torchinfo文档](https://github.com/TylerYep/torchinfo)
- MACs vs FLOPs: [讨论帖](https://github.com/sovrasov/flops-counter.pytorch/issues/16)
