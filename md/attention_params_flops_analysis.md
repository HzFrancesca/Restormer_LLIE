# Restormer 不同注意力机制参数量和FLOPs分析

## 概述
本文档分析 Restormer 中五种不同注意力机制（MDTA, HTA, WTA, IRS, ICS）的参数量和计算复杂度（FLOPs）差异。

---

## 1. 注意力机制简介

### 1.1 MDTA (Multi-DConv Head Transposed Self-Attention)
- **位置**: `restormer_arch.py` 中的 `Attention` 类
- **特点**: 原始 Restormer 的注意力机制，使用深度可分离卷积和转置自注意力
- **注意力范围**: 全局注意力（H×W）

### 1.2 HTA (Height Transposed Attention)
- **位置**: `extra_attention.py` 中的 `HTA` 类
- **特点**: 横向（水平）注意力，在宽度维度上计算注意力
- **注意力范围**: W×W

### 1.3 WTA (Width Transposed Attention)
- **位置**: `extra_attention.py` 中的 `WTA` 类
- **特点**: 在每一行上计算注意力
- **注意力范围**: (C×W)×(C×W) per row

### 1.4 IRS (Intra-Row Self-Attention)
- **位置**: `extra_attention.py` 中的 `IRS` 类
- **特点**: 行内自注意力，垂直方向注意力
- **注意力范围**: W×W per channel

### 1.5 ICS (Intra-Column Self-Attention)
- **位置**: `extra_attention.py` 中的 `ICS` 类
- **特点**: 列内自注意力，水平方向注意力
- **注意力范围**: H×H per channel

---

## 2. 参数量分析

### 2.1 MDTA 参数组成
假设输入维度为 `dim`，偏置参数为 `bias`：

```python
# MDTA 参数
self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # num_heads 个参数
self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # dim * dim * 3 + (dim*3 if bias else 0)
self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, groups=dim * 3, bias=bias)  # (dim*3) * 9 + (dim*3 if bias else 0)
self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # dim * dim + (dim if bias else 0)
```

**MDTA 总参数量**（假设 bias=False）:
- Temperature: `num_heads`
- QKV Conv: `dim × dim × 3 = 3×dim²`
- QKV DWConv: `(dim × 3) × 9 = 27×dim`
- Project Out: `dim × dim = dim²`
- **总计**: `4×dim² + 27×dim + num_heads`

### 2.2 HTA 参数组成

```python
# HTA 参数
self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # num_heads
self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 3×dim²
self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=bias)  # 9×dim
self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=bias)  # 9×dim
self.mdc_v = MDC(dim, bias)  # 见下方MDC分析
self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # dim²
```

**MDC 参数量**:
```python
self.mdc1 = nn.Conv2d(dim, dim/2, groups=dim/2, kernel_size=3, bias=bias)  # (dim/2) × 9 = 4.5×dim
self.mdc2 = nn.Conv2d(dim/2, dim/2, groups=dim/2, kernel_size=3, bias=bias)  # (dim/2) × 9 = 4.5×dim
# MDC 总计: 9×dim
```

**HTA 总参数量**（假设 bias=False）:
- Temperature: `num_heads`
- QKV Conv: `3×dim²`
- MDC_Q: `9×dim`
- MDC_K: `9×dim`
- MDC_V: `9×dim`
- Project Out: `dim²`
- **总计**: `4×dim² + 27×dim + num_heads`

### 2.3 WTA 参数组成

WTA 的参数结构与 HTA **完全相同**:
- **总计**: `4×dim² + 27×dim + num_heads`

### 2.4 IRS 参数组成

```python
# IRS 参数
self.temperature = nn.Parameter(torch.ones(dim, 1, 1))  # dim（注意：不是num_heads）
self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 3×dim²
self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=bias)  # 9×dim
self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=bias)  # 9×dim
self.mdc_v = MDC(dim, bias)  # 9×dim
self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # dim²
```

**IRS 总参数量**（假设 bias=False）:
- Temperature: `dim`（**与 MDTA/HTA/WTA 不同**）
- 其他参数: `4×dim² + 27×dim`
- **总计**: `4×dim² + 28×dim`

### 2.5 ICS 参数组成

ICS 的参数结构与 IRS **完全相同**:
- **总计**: `4×dim² + 28×dim`

---

## 3. 参数量对比总结

| 注意力类型 | Temperature 参数 | 卷积参数 | 总参数量（bias=False） | 差异说明 |
|-----------|----------------|---------|---------------------|---------|
| **MDTA**  | num_heads      | 4×dim² + 27×dim | `4×dim² + 27×dim + num_heads` | 基准 |
| **HTA**   | num_heads      | 4×dim² + 27×dim | `4×dim² + 27×dim + num_heads` | **与MDTA相同** |
| **WTA**   | num_heads      | 4×dim² + 27×dim | `4×dim² + 27×dim + num_heads` | **与MDTA相同** |
| **IRS**   | **dim**        | 4×dim² + 27×dim | `4×dim² + 28×dim` | Temperature维度不同 |
| **ICS**   | **dim**        | 4×dim² + 27×dim | `4×dim² + 28×dim` | Temperature维度不同 |

### 关键发现：
1. **MDTA、HTA、WTA 参数量完全相同**
2. **IRS、ICS 参数量相同，但与前三者略有差异**
3. 差异主要在 `temperature` 参数：
   - MDTA/HTA/WTA: `num_heads` 个参数
   - IRS/ICS: `dim` 个参数
4. 由于 `dim` 通常远大于 `num_heads`，IRS/ICS 的参数量略多于 MDTA/HTA/WTA
   - 例如: dim=48, num_heads=1 时，差异为 `48 - 1 = 47` 个参数
   - 但这个差异相对于总参数量 `4×48² + 27×48 ≈ 10,512` 可以忽略不计

**结论**: **五种注意力机制的参数量基本相同**，差异可忽略（< 0.5%）。

---

## 4. FLOPs 分析

FLOPs（浮点运算次数）主要取决于：
1. 卷积操作
2. 注意力矩阵计算
3. 矩阵乘法

假设输入特征图大小为 `H×W`，通道数为 `dim`。

### 4.1 MDTA FLOPs

**卷积部分**:
```
1. QKV Conv (1×1): 2 × H × W × dim × (dim×3) = 6×H×W×dim²
2. QKV DWConv (3×3 depthwise): 2 × H × W × (dim×3) × 9 = 54×H×W×dim
3. Project Out (1×1): 2 × H × W × dim × dim = 2×H×W×dim²
卷积总计: 8×H×W×dim² + 54×H×W×dim
```

**注意力部分** (假设 num_heads=h):
```
每个头的通道数 c' = dim/h
Q, K, V shape: [B, h, c', H×W]

1. Q @ K^T: h × (c' × H×W × H×W) = h × c' × (H×W)²
2. Softmax: 可忽略（相对较小）
3. Attn @ V: h × (H×W × H×W × c') = h × c' × (H×W)²

注意力总计: 2 × h × c' × (H×W)² = 2 × dim × (H×W)²
```

**MDTA 总FLOPs**: `8×H×W×dim² + 54×H×W×dim + 2×dim×(H×W)²`

关键项: `2×dim×(H×W)²` - **全局注意力的计算量，与 (H×W)² 成正比**

### 4.2 HTA FLOPs

**卷积部分**: 与 MDTA 相同
```
8×H×W×dim² + 54×H×W×dim
```

**注意力部分**:
```
HTA 重排列: [B, h, c×h, w]
Q^T @ K: h × (W × (c×H) × (c×H)) = h × c × H × W × (c×H)
       = h × c² × H² × W
Attn @ V: h × (W × (c×H) × W) = h × c × H × W²

由于 h × c = dim:
注意力总计: dim × c × H² × W + dim × c × H × W²
          = dim × (dim/h) × H² × W + dim × (dim/h) × H × W²
          = (dim²/h) × H × W × (H + W)
```

**HTA 总FLOPs**: `8×H×W×dim² + 54×H×W×dim + (dim²/h)×H×W×(H+W)`

关键项: `(dim²/h)×H×W×(H+W)` - **复杂度与 H+W 成正比，低于全局注意力**

### 4.3 WTA FLOPs

**注意力部分**:
```
WTA 重排列: [B, h, H, c×w]
Q @ K^T: h × H × ((c×W) × (c×W)) = h × H × c² × W²
Attn @ V: h × H × ((c×W) × (c×W)) = h × H × c² × W²

注意力总计: 2 × h × H × c² × W² = 2 × (dim²/h) × H × W²
```

**WTA 总FLOPs**: `8×H×W×dim² + 54×H×W×dim + 2×(dim²/h)×H×W²`

关键项: `2×(dim²/h)×H×W²` - **复杂度与 W² 成正比**

### 4.4 IRS FLOPs

**注意力部分**:
```
IRS shape: [B, C, H, W]
Q @ K^T: dim × (W × W × H) = dim × H × W²
Attn @ V: dim × (W × W × H) = dim × H × W²

注意力总计: 2 × dim × H × W²
```

**IRS 总FLOPs**: `8×H×W×dim² + 54×H×W×dim + 2×dim×H×W²`

关键项: `2×dim×H×W²` - **复杂度与 W² 成正比**

### 4.5 ICS FLOPs

**注意力部分**:
```
ICS shape: [B, C, H, W]
Q^T @ K: dim × (H × H × W) = dim × H² × W
Attn @ V: dim × (H × H × W) = dim × H² × W

注意力总计: 2 × dim × H² × W
```

**ICS 总FLOPs**: `8×H×W×dim² + 54×H×W×dim + 2×dim×H²×W`

关键项: `2×dim×H²×W` - **复杂度与 H² 成正比**

---

## 5. FLOPs 对比总结

### 5.1 完整FLOPs公式对比

| 注意力类型 | 卷积FLOPs | 注意力FLOPs | 总FLOPs |
|-----------|----------|------------|---------|
| **MDTA**  | 8HWd² + 54HWd | 2d(HW)² | `8HWd² + 54HWd + 2d(HW)²` |
| **HTA**   | 8HWd² + 54HWd | (d²/h)HW(H+W) | `8HWd² + 54HWd + (d²/h)HW(H+W)` |
| **WTA**   | 8HWd² + 54HWd | 2(d²/h)HW² | `8HWd² + 54HWd + 2(d²/h)HW²` |
| **IRS**   | 8HWd² + 54HWd | 2dHW² | `8HWd² + 54HWd + 2dHW²` |
| **ICS**   | 8HWd² + 54HWd | 2dH²W | `8HWd² + 54HWd + 2dH²W` |

其中: `d = dim`, `h = num_heads`

### 5.2 注意力计算复杂度对比

假设 H = W = 128, dim = 48, num_heads = 1:

| 注意力类型 | 注意力FLOPs | 数值（H=W=128, d=48） | 相对MDTA |
|-----------|------------|---------------------|---------|
| **MDTA**  | 2d(HW)² | 2×48×(128×128)² ≈ **25.4G** | 100% |
| **HTA**   | (d²/h)HW(H+W) | (48²/1)×128×128×256 ≈ **1.0G** | **3.9%** |
| **WTA**   | 2(d²/h)HW² | 2×(48²/1)×128×128² ≈ **1.5G** | **5.9%** |
| **IRS**   | 2dHW² | 2×48×128×128² ≈ **0.2G** | **0.8%** |
| **ICS**   | 2dH²W | 2×48×128²×128 ≈ **0.2G** | **0.8%** |

### 5.3 关键发现

1. **卷积部分FLOPs完全相同**: 所有注意力机制的卷积操作完全一致
   - 卷积FLOPs: `8HWd² + 54HWd` ≈ 4.7M (H=W=128, d=48)
   - 这部分占总FLOPs的比例很小（< 1%）

2. **注意力部分FLOPs差异巨大**:
   - **MDTA 最高**: O((HW)²) - 全局注意力，计算量最大
   - **HTA 中等**: O(HW(H+W)) - 约为 MDTA 的 1/25
   - **WTA 较低**: O(HW²) - 约为 MDTA 的 1/17
   - **IRS/ICS 最低**: O(HW²) 或 O(H²W) - 约为 MDTA 的 1/125

3. **实际应用中的差异**:
   - 对于 128×128 分辨率，MDTA 的注意力计算约 25.4 GFLOPs
   - IRS/ICS 的注意力计算仅约 0.2 GFLOPs
   - **使用 IRS/ICS 可以减少约 99% 的注意力计算量**

4. **总FLOPs对比** (H=W=128, d=48):
   - MDTA: ≈ 25.4G FLOPs
   - HTA: ≈ 1.0G FLOPs (**减少 96%**)
   - WTA: ≈ 1.5G FLOPs (**减少 94%**)
   - IRS: ≈ 0.2G FLOPs (**减少 99.2%**)
   - ICS: ≈ 0.2G FLOPs (**减少 99.2%**)

---

## 6. 总体结论

### 6.1 参数量
**五种注意力机制的参数量几乎完全相同**（差异 < 0.5%），不会显著影响模型大小。

### 6.2 FLOPs
**五种注意力机制的FLOPs差异极大**：

1. **MDTA (全局注意力)**: 
   - FLOPs 最高
   - 复杂度: O(dim × (H×W)²)
   - 适用场景: 需要全局感受野的任务

2. **HTA (Height Transposed Attention)**:
   - FLOPs 较低（约为 MDTA 的 4%）
   - 复杂度: O((dim²/h) × H×W × (H+W))
   - 适用场景: 需要跨宽度方向建模的任务

3. **WTA (Width Transposed Attention)**:
   - FLOPs 较低（约为 MDTA 的 6%）
   - 复杂度: O((dim²/h) × H × W²)
   - 适用场景: 需要在每一行内建模的任务

4. **IRS (Intra-Row Self-Attention)**:
   - FLOPs 极低（约为 MDTA 的 0.8%）
   - 复杂度: O(dim × H × W²)
   - 适用场景: 垂直方向特征建模

5. **ICS (Intra-Column Self-Attention)**:
   - FLOPs 极低（约为 MDTA 的 0.8%）
   - 复杂度: O(dim × H² × W)
   - 适用场景: 水平方向特征建模

### 6.3 实践建议

1. **如果追求性能（精度）**: 使用 MDTA（全局注意力）
2. **如果追求效率**: 使用 IRS/ICS（可减少 99% 的计算量）
3. **如果需要平衡**: 使用 HTA/WTA（减少 94-96% 的计算量）
4. **可以混合使用**: 在不同层级使用不同的注意力机制
   - 浅层使用 IRS/ICS（低分辨率，需要局部特征）
   - 深层使用 MDTA（高分辨率，需要全局特征）

### 6.4 答案总结

**问题**: Restormer不同的注意力修改下，网络参数量和FLops是否会发生变化？

**答案**:
- **参数量**: **基本不变**（差异 < 0.5%，可忽略）
- **FLOPs**: **显著变化**（最高可相差 125 倍）
  - MDTA → IRS/ICS: FLOPs **减少 99.2%**
  - MDTA → HTA/WTA: FLOPs **减少 94-96%**

因此，**不同注意力机制主要影响计算复杂度（FLOPs），而不影响参数量**。
