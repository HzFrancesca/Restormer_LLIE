# MDTA注意力模块矩阵乘法FLOPs详细分析

## 问题
当输入为 `(1, 3, 128, 128)` 时，使用MDTA的Restormer网络中所有注意力模块的矩阵乘法 `@` 计算量是多少？

---

## 网络配置
根据配置文件 `LowLight_Restormer_128_2.yml`:

```yaml
dim: 48
num_blocks: [4, 6, 6, 8]
num_refinement_blocks: 4
heads: [1, 2, 4, 8]
```

---

## MDTA矩阵乘法计算原理

### 代码分析
从 `restormer_arch.py` 的 `Attention` 类可以看到:

```python
def forward(self, x):
    b, c, h, w = x.shape
    
    qkv = self.qkv_dwconv(self.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)
    
    # Reshape: [B, C, H, W] -> [B, num_heads, C/num_heads, H*W]
    q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
    k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
    v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
    
    # 矩阵乘法1: Q @ K^T
    attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, h, c, HW] @ [B, h, HW, c] -> [B, h, c, c]
    attn = attn.softmax(dim=-1)
    
    # 矩阵乘法2: Attn @ V
    out = attn @ v  # [B, h, c, c] @ [B, h, c, HW] -> [B, h, c, HW]
    
    out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
    out = self.project_out(out)
    return out
```

### 关键发现：MDTA是Transposed Attention
- **标准Self-Attention**: 注意力矩阵维度为 `[H×W, H×W]`（空间维度）
- **MDTA (Transposed Attention)**: 注意力矩阵维度为 `[C, C]`（通道维度）

这就是为什么叫"Transposed"的原因：它在通道维度而不是空间维度上计算注意力。

### 矩阵乘法FLOPs公式

对于单个MDTA模块，设：
- `dim`: 输入通道数
- `num_heads`: 注意力头数
- `c = dim / num_heads`: 每个头的通道数
- `H, W`: 特征图的高度和宽度
- `HW = H × W`: 空间维度大小

**两个矩阵乘法操作**：

1. **Q @ K^T**: 
   ```
   形状: [num_heads, c, HW] @ [num_heads, HW, c] -> [num_heads, c, c]
   每个头的FLOPs: c × HW × c = c² × HW
   总FLOPs: num_heads × c² × HW = (dim²/num_heads) × HW
   ```

2. **Attn @ V**:
   ```
   形状: [num_heads, c, c] @ [num_heads, c, HW] -> [num_heads, c, HW]
   每个头的FLOPs: c × c × HW = c² × HW
   总FLOPs: num_heads × c² × HW = (dim²/num_heads) × HW
   ```

**单个MDTA模块的矩阵乘法总FLOPs**:
```
Total = 2 × (dim²/num_heads) × H × W
```

---

## Restormer网络结构

Restormer采用U-Net架构，包含4个层级（Level）：

| Level | 阶段 | 分辨率 | 通道数 | 头数 | Block数 |
|-------|------|--------|--------|------|---------|
| 1 | Encoder | 128×128 | 48 | 1 | 4 |
| 2 | Encoder | 64×64 | 96 | 2 | 6 |
| 3 | Encoder | 32×32 | 192 | 4 | 6 |
| 4 | Latent | 16×16 | 384 | 8 | 8 |
| 3 | Decoder | 32×32 | 192 | 4 | 6 |
| 2 | Decoder | 64×64 | 96 | 2 | 6 |
| 1 | Decoder | 128×128 | **96** | 1 | 4 |
| 1 | Refinement | 128×128 | **96** | 1 | 4 |

**注意**: 
- Decoder Level 1 的通道数是 **96** 而不是 48（因为没有1×1卷积降维）
- Refinement 阶段使用相同的配置

**总Transformer Block数量**: 4 + 6 + 6 + 8 + 6 + 6 + 4 + 4 = **44个**

---

## 详细计算结果

### Level 1 - Encoder (4 blocks)
- **分辨率**: 128×128
- **通道数**: 48
- **头数**: 1
- **每个Block的矩阵乘法FLOPs**:
  ```
  Q @ K^T: 1 × 48² × 128 × 128 = 37,748,736 FLOPs
  Attn @ V: 1 × 48² × 128 × 128 = 37,748,736 FLOPs
  Total per block: 75,497,472 FLOPs
  ```
- **总计 (4 blocks)**: **301,989,888 FLOPs** (≈ 0.302 GFLOPs)

### Level 2 - Encoder (6 blocks)
- **分辨率**: 64×64
- **通道数**: 96
- **头数**: 2
- **每个Block的矩阵乘法FLOPs**:
  ```
  Q @ K^T: 2 × (96/2)² × 64 × 64 = 18,874,368 FLOPs
  Attn @ V: 2 × (96/2)² × 64 × 64 = 18,874,368 FLOPs
  Total per block: 37,748,736 FLOPs
  ```
- **总计 (6 blocks)**: **226,492,416 FLOPs** (≈ 0.226 GFLOPs)

### Level 3 - Encoder (6 blocks)
- **分辨率**: 32×32
- **通道数**: 192
- **头数**: 4
- **每个Block的矩阵乘法FLOPs**:
  ```
  Q @ K^T: 4 × (192/4)² × 32 × 32 = 9,437,184 FLOPs
  Attn @ V: 4 × (192/4)² × 32 × 32 = 9,437,184 FLOPs
  Total per block: 18,874,368 FLOPs
  ```
- **总计 (6 blocks)**: **113,246,208 FLOPs** (≈ 0.113 GFLOPs)

### Level 4 - Latent (8 blocks)
- **分辨率**: 16×16
- **通道数**: 384
- **头数**: 8
- **每个Block的矩阵乘法FLOPs**:
  ```
  Q @ K^T: 8 × (384/8)² × 16 × 16 = 4,718,592 FLOPs
  Attn @ V: 8 × (384/8)² × 16 × 16 = 4,718,592 FLOPs
  Total per block: 9,437,184 FLOPs
  ```
- **总计 (8 blocks)**: **75,497,472 FLOPs** (≈ 0.075 GFLOPs)

### Level 3 - Decoder (6 blocks)
- 配置与 Encoder Level 3 相同
- **总计 (6 blocks)**: **113,246,208 FLOPs** (≈ 0.113 GFLOPs)

### Level 2 - Decoder (6 blocks)
- 配置与 Encoder Level 2 相同
- **总计 (6 blocks)**: **226,492,416 FLOPs** (≈ 0.226 GFLOPs)

### Level 1 - Decoder (4 blocks)
- **分辨率**: 128×128
- **通道数**: **96** (注意：不是48)
- **头数**: 1
- **每个Block的矩阵乘法FLOPs**:
  ```
  Q @ K^T: 1 × 96² × 128 × 128 = 150,994,944 FLOPs
  Attn @ V: 1 × 96² × 128 × 128 = 150,994,944 FLOPs
  Total per block: 301,989,888 FLOPs
  ```
- **总计 (4 blocks)**: **1,207,959,552 FLOPs** (≈ 1.208 GFLOPs)

### Refinement (4 blocks)
- 配置与 Decoder Level 1 相同
- **总计 (4 blocks)**: **1,207,959,552 FLOPs** (≈ 1.208 GFLOPs)

---

## 最终结果汇总

| 阶段 | Block数 | FLOPs | 占比 |
|------|---------|-------|------|
| Encoder Level 1 | 4 | 301,989,888 | 8.7% |
| Encoder Level 2 | 6 | 226,492,416 | 6.5% |
| Encoder Level 3 | 6 | 113,246,208 | 3.3% |
| Latent Level 4 | 8 | 75,497,472 | 2.2% |
| Decoder Level 3 | 6 | 113,246,208 | 3.3% |
| Decoder Level 2 | 6 | 226,492,416 | 6.5% |
| **Decoder Level 1** | 4 | **1,207,959,552** | **34.8%** |
| **Refinement** | 4 | **1,207,959,552** | **34.8%** |
| **总计** | **44** | **3,472,883,712** | **100%** |

### 核心发现

1. **总矩阵乘法FLOPs**: **3,472,883,712 FLOPs ≈ 3.47 GFLOPs**

2. **计算量分布**:
   - **Decoder Level 1 + Refinement 占总计算量的 69.6%**
   - 原因：这两个阶段在最高分辨率(128×128)且使用96通道
   - 分辨率对计算量影响巨大：128×128 vs 16×16 = 64倍差异

3. **平均每个Block**: 78,929,175 FLOPs (≈ 79 MFLOPs)

4. **对称性观察**:
   - Encoder Level 1 与 Decoder Level 1 的Block数相同(4个)
   - 但Decoder Level 1的计算量是Encoder Level 1的 **4倍**
   - 原因：通道数不同（96 vs 48），计算量与 dim² 成正比

---

## 与文档中的分析对比

在 `attention_params_flops_analysis.md` 中提到MDTA的注意力FLOPs为 `2×dim×(H×W)²`，这是针对**标准Self-Attention**的分析。

但实际上MDTA是**Transposed Attention**，正确的公式应该是：
```
MDTA矩阵乘法FLOPs = 2 × (dim²/num_heads) × H × W
```

**关键差异**：
- **标准Attention**: 复杂度为 O(dim × (H×W)²) - 与空间维度的平方成正比
- **MDTA (Transposed)**: 复杂度为 O(dim² × H×W) - 与空间维度线性相关

这也解释了为什么MDTA在高分辨率图像上更高效！

---

## 验证与结论

### 验证计算
以Encoder Level 1为例验证：
```python
dim = 48, num_heads = 1, H = W = 128
每个Block FLOPs = 2 × (48²/1) × 128 × 128
                = 2 × 2304 × 16384
                = 75,497,472 ✓ (与计算结果一致)
```

### 结论

对于输入 `(1, 3, 128, 128)` 的MDTA-Restormer网络：

- **所有注意力模块的矩阵乘法总计算量**: **3.47 GFLOPs**
- **44个Transformer Block，平均每个**: **79 MFLOPs**
- **计算瓶颈**: Decoder Level 1 和 Refinement 阶段（占69.6%）

---

## 计算脚本
详细计算代码见: `scripts/calculate_mdta_matmul_flops.py`

运行命令：
```bash
python scripts/calculate_mdta_matmul_flops.py
```
