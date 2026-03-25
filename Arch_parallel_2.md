```python

##########################################################################
## Dual-Branch Spatial & Frequency Attention
class DualAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(DualAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # --- 合并生成所有组件 (q_s, k_s, v_s, q_f, k_f) ---
        self.qkv_all = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_all_dwconv = nn.Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias,
        )

        # --- 空间分支投影 ---
        self.proj_spatial = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # --- 特征融合投影 ---
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # ==========================================
        # 1. 空间域分支交互
        # ==========================================
        qkv_all = self.qkv_all_dwconv(self.qkv_all(x))
        q_s, k_s, v_s, q_f, k_f = qkv_all.chunk(5, dim=1)

        q_s = rearrange(q_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k_s = rearrange(k_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v_s = rearrange(v_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q_s = F.normalize(q_s, dim=-1)
        k_s = F.normalize(k_s, dim=-1)

        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)

        out_s = attn_s @ v_s
        out_s = rearrange(
            out_s, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out_s = self.proj_spatial(out_s)

        # ==========================================
        # 2. DCT 频域分支交互
        # ==========================================
        # 应用 2D DCT 变换
        q_f_dct = dct_2d(q_f)
        k_f_dct = dct_2d(k_f)

        # 变换形状用于多头注意力计算 [b, head, c, hw]
        q_f_dct = rearrange(
            q_f_dct, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        k_f_dct = rearrange(
            k_f_dct, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )

        # MDTA 机制的归一化
        q_f_dct = F.normalize(q_f_dct, dim=-1)
        k_f_dct = F.normalize(k_f_dct, dim=-1)

        # 计算通道维度的注意力矩阵 [b, head, c, c]
        attn_f = (q_f_dct @ k_f_dct.transpose(-2, -1)) * self.temperature

        # 对注意力矩阵应用 IDCT 和 Softmax
        attn_f = idct_2d(attn_f)
        attn_f = attn_f.softmax(dim=-1)

        # ==========================================
        # 3. 融合: 与空间分支 out_s 进行矩阵乘法
        # ==========================================
        # 准备 out_s 的形状 [b, head, c, hw]
        out_s_reshaped = rearrange(
            out_s, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )

        # 矩阵乘法融合
        out = attn_f @ out_s_reshaped

        # 还原空间维度并投影输出
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)

        return out
```

# DualAttention (DCT 版) 模块架构详解

## 1. 设计初衷与背景

本模块是针对 **低光图像增强 (LLIE)** 任务设计的全新双分支注意力机制。它结合了 Restormer 的空间建模能力与 **离散余弦变换 (DCT)** 的频域分析能力。相比于之前的 Patch-based FFT 版本，该版本在频域建模上更加紧凑，且引入了跨域矩阵乘法融合。

### 核心特性

* **空间分支**：标准的转置注意力 (Transposed Attention)，捕捉全局空间通道关系。
* **频率分支**：将特征图映射到 DCT 域，利用 **MDTA 机制** 在频域计算通道间的注意力。
* **IDCT 调制**：将频域计算得到的注意力矩阵通过 **IDCT** 还原到准空间域，实现更精准的权重校准。
* **矩阵乘法融合**：频率注意力矩阵 $Attn_f$ 直接作用于空间分支输出 $Out_s$（$Attn_f \times Out_s$），实现深层跨域交互。

---

## 2. 代码实现关键流程

### A. 统一投影优化 (`dim * 5`)

将 $Q_s, K_s, V_s, Q_f, K_f$ 统一在一次大卷积中生成，显著提高计算并行度：

* **Query/Key 对数**：空间域 ($Q_s/K_s$) 与频域 ($Q_f/K_f$) 共用一个输入投影，减少 Kernel 调度开销。
* **深度卷积**：后续接 3x3 DWConv 注入局部上下文。

### B. DCT 频域 MDTA

不同于传统的空间局部 Patch 处理，本模块在**全局频率分量**上执行注意力计算：

* **DCT-II 变换**：将 $Q_f, K_f$ 转换到频域。由于 DCT 具有极高的能量集中特性，模型能更敏锐地识别噪声分量。
* **转置注意力**：在频域计算 $C \times C$ 的相关性。这有助于模型理解不同频道在频域上的协同补光规律。

### C. IDCT 与 Softmax 调制

流程图中点睛之笔：

1. **IDCT 还原**：将频域生成的 $C \times C$ 矩阵视为一种频域滤波器，通过 IDCT 将其转换回能与空间特征直接对齐的权重分布。
2. **Softmax 归一化**：确保各通道权重的概率分布稳定，防止产生过大数值干扰训练。

---

## 3. 实现与应用注意事项

> [!IMPORTANT]
> **全局建模 vs 局部建模**
> 相比于 FFT-Patch 版本，DCT 版本是基于全图尺寸的（Global Domain）。对于大分辨率图像，通过 FFT 实现的快速 DCT 算法（$O(N \log N)$）能有效控制计算开销。

> [!TIP]
> **矩阵乘法融合**
> 原本的逐元素点乘（$Out_s \odot Attn_f$）更像是一个 Gate 机制；现在的矩阵乘法（$Attn_f \times Out_s$）则更像是一个 **域间混合器 (Domain Mixer)**，允许频率特征对空间特征进行更深入的重构。

---

## 4. DualAttention (DCT 版) 流程示意图

```text
           Input X (B, C, H, W)
                 |
        [ QKV_All Projection ]
       (Conv 1x1 + DWConv 3x3)
                 |
    +-----------+-----------+
    |           |           |
 [Qs, Ks, Vs]  [V_s]      [Qf, Kf]
    |           |           |
    |           |       [ DCT_2D ]
    |           |           |
    |           |     [ MDTA Mechanism ]
    |           |   (Qf_dct @ Kf_dct^T)
    |           |           |
    |           |       [ IDCT_2D ]
    |           |           |
    |           |       [ SoftMax ]
    |           |           |
    |           |        (Attn_f)
    |           |           |
 [ Spatial Attention ]      |
 (MDTA: Qs * Ks^T * Vs)     |
    |                       |
 [ Proj_spatial ]           |
    |                       |
 (Out_s) ------------------[ @ ] (Matrix Multiplication)
                            |
                     [ Proj_out ]
                            |
                      Output (B, C, H, W)
```

### 流程阶段详解

1. **统一投影层 (Merged Projection)**：一次生成五个通道分量，确保计算流程极致紧凑。
2. **空间分支 (Spatial Branch)**：计算空间域通道间的 $Attn_s$，并将结果作用于 $V_s$ 得到初步增强特征 $Out\_s$。
3. **频率分支 (Frequency Branch)**：
    * 将生成的 $Q_f, K_f$ 投射至 DCT 频域。
    * 在频域执行转置注意力计算，识别核心频率响应。
    * 通过 **IDCT** 将频域关系映射回空间语义维度。
4. **跨域矩阵融合 (Domain Matrix Fusion)**：这是该架构的核心创新。频率注意力矩阵 $Attn\_f$ (形状 $C \times C$) 与空间特征 $Out\_s$ (形状 $C \times HW$) 进行矩阵乘法。这打破了空间和频率的壁垒，实现了真正的多域联合增强。
