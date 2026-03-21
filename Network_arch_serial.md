# 串行式网络架构分析 (Serial Network Architecture)

## 1. 核心改进：串行 TransformerBlock

区别于时空与频域分支相互平行的双轨设计，当前的 `TransformerBlock` 将特征处理流程重构为**基于独立开关控制的串行管线 (Serial Pipeline)**。各阶段按严格的先后顺序执行，并各自配有独立的 `LayerNorm` 与残差连接（Residual Connection）。

在一个配置了全部模块的 Block （如 Decoder 阶段）中，其内部完整执行流程如下：

1. **空间注意力阶段 (Spatial Attn Stage)**

   ```python
   x = x + self.attn_spatial(self.norm_spatial_attn(x))
   ```

   *作用*：通过通道维度的注意力运算（MDTA），在空域建立特征的全局关系。

2. **频率注意力阶段 (Freq Attn Stage)**

   ```python
   x = x + self.attn_freq(self.norm_freq_attn(x))
   ```

   *作用*：紧接空域注意力后，通过 `FSAS` 模块转换到频域计算由 Patch (8x8) 分块的注意力矩阵，增强并补充高频/低频响应特性。

3. **空间前馈网络阶段 (Spatial FFN Stage)**

   ```python
   x = x + self.ffn_spatial(self.norm_spatial_ffn(x))
   ```

   *作用*：基于 `GDFN` 机制，利用 1x1 和 3x3 DwConv 进行空域局部信息的提取和特征门控（Gated）非线性映射。

4. **频率前馈网络阶段 (Freq FFN Stage)**

   ```python
   x = x + self.ffn_freq(self.norm_freq_ffn(x))
   ```

   *作用*：在输出之前，串联进入 `DFFN`。在常规 FFN 增加基于傅里叶变换的频域滤波器，过滤出特定频段核心特征，得到最终结果。

结合上述四个阶段，串行 `TransformerBlock` 的完整流程图如下：

```text
       ┌───────────────┐
       │Input Feature x│
       └───────┬───────┘
               │
┌──────────────▼──────────────┐
│  Spatial Attention Stage    │
│  x = x + Spatial_Attn(LN(x))│
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Frequency Attention Stage  │
│  x = x + Freq_Attn(LN(x))   │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Spatial FFN Stage          │
│  x = x + Spatial_FFN(LN(x)) │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Frequency FFN Stage        │
│  x = x + Freq_FFN(LN(x))    │
└──────────────┬──────────────┘
               │
       ┌───────▼───────┐
       │ Output Feature│
       └───────────────┘
```

这种深度串行的策略（Spatial -> Freq -> Spatial -> Freq）大大拉长了各个 Transformer Block 的梯度路径，使得模型对单一特征的解构更加彻底。

---

## 2. 局部网络组件剖析 (FFN 与 Attention)

### 2.1 Forward 逻辑模块 (FFN)

FFN 设计包含两个变种模块：纯空域的 `GDFN` 及加入频域过滤的 `DFFN`。值得注意的是，`DFFN` 内部虽然名为 "Dual-domain" (双域)，实则采取了**串行频空结合**的执行策略。

* **GDFN**:
  * **流程**：输入特征 -> `1x1 Conv` 升维 (x2) -> `3x3 Depthwise Conv` 聚合局部 -> 将通道对半切分 (Chunk=2) -> 左半经过 GELU 后与右半相乘 (门控作用) -> `1x1 Conv` 降维还原。
* **DFFN**:
  * **流程**：在升维阶段 (`1x1 Conv`) 之后，**串行插入**了一组频域操作：先重排为 `8x8` Patch，执行 `rfft2`，乘以可学习的频域感知权重矩阵 (`fft_weight`)，再 `irfft2` 回归空域。随后的特征流**完全复用 GDFN 的门控网络** (`3x3 DwConv -> Chunk -> GELU Gating -> 1x1 Conv`)。

**GDFN 与 DFFN 流程对比如下：**

```text
      【GDFN (纯空域)】                  【DFFN (频域+空域)】
                                       
        Input X                             Input X
           │                                   │
           ▼                                   ▼
    1x1 Conv (升维 x2)                  1x1 Conv (升维 x2)
           │                                   │
           │                                   ▼
           │                            Reshape (8x8 Patch)
           │                                   │
           │                                   ▼
           │                           rfft2 (转频域)
           │                                   │
           │                                   ▼
           │                          (*) fft_weight (频域感知权重)
           │                                   │
           │                                   ▼
           │                           irfft2 (转回空域)
           │                                   │
           ▼                                   ▼
      3x3 DWConv                          3x3 DWConv
           │                                   │
           ▼                                   ▼
   Chunk (通道对半切分)                 Chunk (通道对半切分)
     ┌─────┴─────┐                       ┌─────┴─────┐
     ▼           ▼                       ▼           ▼
   GELU         (保持)                 GELU         (保持)
     │           │                       │           │
     └──► (x) ◄──┘                       └──► (x) ◄──┘
   Element-wise 乘法                   Element-wise 乘法
           │                                   │
           ▼                                   ▼
    1x1 Conv (降维还原)                 1x1 Conv (降维还原)
           │                                   │
           ▼                                   ▼
        Output                              Output
```

### 2.2 特征映射注意力 (Attention)

* **Attention (空间多头转置注意力 / MDTA)**：通过 `1x1` 和 `3x3` DwConv 得到 Q, K, V，在**通道维度**上进行自注意力相乘 `(Q @ K.T)`。通道维度规避了空域计算随着分辨率剧增的 O(N^2) 复杂度问题。
* **FSAS (Frequency Spatial Attention System)**：频域对位相乘机制。Q 和 K 进入 8x8 块级别傅里叶变换 (`rfft2`)。在频域中将两组矩阵硬复用复数乘积（结合空间卷积定理，等价于空域特定的卷积相关特性），再通过 `irfft2` 进行反变换输出。

---

## 3. 总网络流程与模块分布 (Network Workflow)

Restormer 本体利用 U-Net 层级结构。不同网络深度由于语义层次不同，对应 `TransformerBlock` 内各模块的启用策略（`True`/`False`）也经过了特定调整。

### 输入层与降采样管线 (Encoder)

1. **输入与 Patch Embedding**: 输入图像通过 `OverlapPatchEmbed` 做最初的高维映射。
2. **Encoder Level 1 & 2**: 高分辨率特征图区域。极度关注空间域。
    * **配置**: `空间 Attn` 以及 `空间 FFN` （均未开启频域处理）。
3. **Encoder Level 3**: 较低的分辨率。开始渗透频域分析。
    * **配置**: `空间 Attn` + `空间 FFN` + `频率 FFN` (仅仅未开启频域 Attention)。
4. **潜变量层 (Latent Layer / Level 4)**: 最小特征尺度，处于瓶颈层部位。
    * **配置**: 与 Level 3 相同（即不引入频率衰减极大的 `FSAS` 频率 Attention）。

### 升采样管线重组与多模态加持 (Decoder & Refinement)

整个 Decoder 管线（包含跳跃特征拼接）以及 Refinement 均将所有组件拉满，执行大面积串行增强恢复。

1. **Decoder Level 3 / Level 2 / Level 1**: 各层依序承接并对拼接了 Encoder 侧特征 (Skip Connection) 以及深层上采样的特征做处理。
    * **配置全开**: `空间 Attn` + `频率 Attn` + `空间 FFN` + `频率 FFN`
2. **Refinement 模块**: 以原图同尺寸持续对 Decoder 输出端特征进行加工修复。
    * **配置全开**: `空间 Attn` + `频率 Attn` + `空间 FFN` + `频率 FFN`
3. **最终残差输出**: 利用 3x3 输出卷积对映射的通道进行压制，叠加原始图像输入 (Global Residual Image)，达成亮度/色彩等信息的基准维持。

**全局网络结构与特征流向图如下：**

```text
       Input Image
            │
            ▼
   ┌─────────────────┐
   │OverlapPatchEmbed│
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Encoder Level 1 │ (配置: 空间 Attn + 空间 FFN)
   └────────┬────────┘
            │
            ▼ [Downsample]
   ┌─────────────────┐
   │ Encoder Level 2 │ (配置: 空间 Attn + 空间 FFN)
   └────────┬────────┘
            │
            ▼ [Downsample]
   ┌─────────────────┐
   │ Encoder Level 3 │ (配置: 空间 Attn + 空间 FFN + 频率 FFN)
   └────────┬────────┘
            │
            ▼ [Downsample]
   ┌─────────────────┐
   │ Latent / Level 4│ (配置: 空间 Attn + 空间 FFN + 频率 FFN)
   └────────┬────────┘
            │
            ▼ [Upsample]  ◄────── (+) Skip Connection from Level 3
   ┌─────────────────┐
   │ Decoder Level 3 │ (配置全开: SpAttn + FrAttn + SpFFN + FrFFN)
   └────────┬────────┘
            │
            ▼ [Upsample]  ◄────── (+) Skip Connection from Level 2
   ┌─────────────────┐
   │ Decoder Level 2 │ (配置全开: SpAttn + FrAttn + SpFFN + FrFFN)
   └────────┬────────┘
            │
            ▼ [Upsample]  ◄────── (+) Skip Connection from Level 1
   ┌─────────────────┐
   │ Decoder Level 1 │ (配置全开: SpAttn + FrAttn + SpFFN + FrFFN)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   Refinement    │ (配置全开: SpAttn + FrAttn + SpFFN + FrFFN)
   └────────┬────────┘
            │
            ▼
       [3x3 Conv] ─────────────┐
                               │
            ┌─────────────────(+)─── Input Image (Global Residual)
            │                  
            ▼                  
      Restored Image           
```

## 总结

通过代码逻辑推导说明，目前的架构在空间和频域上实施了**逐步解耦、串联递进**的方式。其整体特点为：
> 先空间打底、再辅以频率剥离；在深层语义和上采样重构恢复层中，强行灌入双重注意力机制和双重通道提取网络，大幅提升了对极低频色块和高频结构纹理边界的敏锐度。
