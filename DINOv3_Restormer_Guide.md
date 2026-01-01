# DINOv3 + Restormer 低光照图像增强集成指南

## 目录

1. [背景介绍](#1-背景介绍)
2. [架构设计原理](#2-架构设计原理)
3. [集成方案（由浅入深）](#3-集成方案由浅入深)
4. [关键技术细节](#4-关键技术细节)
5. [训练策略](#5-训练策略)
6. [完整代码实现](#6-完整代码实现)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 背景介绍

### 1.1 DINOv3 简介

DINOv3 是 Meta AI 于 2025 年发布的最新自监督视觉基础模型：

- **参数规模**：从 21M (ViT-S) 到 6.7B (ViT-7B)
- **训练数据**：17 亿张 Instagram 图像（LVD-1689M）
- **核心创新**：Gram Anchoring 技术解决长训练中特征退化问题
- **特点**：无需标注即可学习高质量的全局和局部视觉特征

### 1.2 Restormer 简介

Restormer 是用于高分辨率图像恢复的高效 Transformer：

- **架构**：U-Net 风格的 Encoder-Decoder
- **核心模块**：MDTA (Multi-DConv Head Transposed Attention)
- **特点**：通道维度注意力，适合高分辨率图像处理

### 1.3 为什么要结合？

| 模型 | 优势 | 局限 |
|------|------|------|
| DINOv3 | 强大的语义理解、预训练特征 | Token 输出，非像素级 |
| Restormer | 像素级恢复、高分辨率处理 | 缺乏高级语义信息 |

**结合目标**：利用 DINOv3 的语义先验指导 Restormer 的像素级恢复。

---

## 2. 架构设计原理

### 2.1 核心挑战

低光照增强 (LLIE) 是像素级任务，而 ViT 输出的是 Token 序列：

```
输入图像 [B, 3, H, W]
    ↓ DINOv3
Token 序列 [B, N_patches + 1, Dim]  # N_patches = (H/16) × (W/16)
    ↓ 需要转换
空间特征图 [B, Dim, H/16, W/16]
    ↓ 上采样
像素级输出 [B, 3, H, W]
```

### 2.2 设计原则

1. **不要只用最后一层**：浅层保留纹理细节（去噪），深层提供语义信息（色彩恢复）
2. **DINO 作为引导而非主干**：保留 CNN/Restormer 的空间细节处理能力
3. **处理 Patch 边界**：避免 ViT 固有的块效应
4. **适配输入分布**：低光照图像需要预处理才能匹配 DINO 训练分布

### 2.3 DINO 特征的两种使用方式（重要概念）

在将 DINO 特征融入网络时，有两种根本不同的设计模式：

| 方式 | 描述 | 代码示例 | 特点 |
|------|------|----------|------|
| **直接融合** | DINO 特征与主干特征直接相加/拼接 | `out = feat + dino_feat` | 简单，但可能引入 patch 噪声 |
| **引导条件** | DINO 生成注意力权重，调制主干特征 | `out = feat * attn(dino_feat)` | 精细，保留空间细节 |

**直接融合示例**：
```python
# DINO 特征直接参与重建
latent = restormer_latent + self.proj(dino_feat)  # 直接相加
# 或
latent = torch.cat([restormer_latent, dino_feat], dim=1)  # 拼接
```

**引导条件示例**：
```python
# DINO 特征只提供 "在哪里增强、增强多少" 的指导
spatial_weight = torch.sigmoid(self.to_attention(dino_feat))  # [B, 1, H, W]
channel_weight = torch.sigmoid(self.to_channel(dino_feat))    # [B, C, 1, 1]
latent = restormer_latent * spatial_weight * channel_weight   # 调制而非替换
```

**为什么 LLIE 任务推荐引导条件方式？**

1. **保留空间细节**：低光照增强需要精确的纹理和边缘信息，这些来自 CNN/Restormer，不应被 DINO 的 patch 特征覆盖
2. **语义指导**：DINO 擅长理解 "这是什么区域"（天空、人脸、物体），用这个信息指导 "哪里需要增强" 更合理
3. **避免块效应**：引导方式不直接使用 DINO 的空间特征，减少 patch 边界问题

**重要说明**：下面的三个集成方案（方案一、二、三）描述的是 DINO 特征**注入的位置和时机**，而"直接融合"vs"引导条件"是**注入的方式**。两者是正交的概念，可以自由组合：

```
┌─────────────────────────────────────────────────────────────┐
│                    集成方案 × 使用方式                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│                 │   直接融合       │      引导条件            │
├─────────────────┼─────────────────┼─────────────────────────┤
│ 方案一(Latent层) │ latent += dino  │ latent *= attn(dino)   │
│ 方案二(多尺度)   │ 各层 concat     │ 各层用 dino 生成 attn   │
│ 方案三(DINO主干) │ decoder(dino)   │ dino 引导 CNN 分支      │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 3. 集成方案（由浅入深）

### 3.1 方案一：DINO 语义引导（推荐起步）

**思路**：保留原有 Restormer，将 DINO 特征作为 Attention 引导注入。

**适用场景**：
- 小数据集（如 LOL 仅几百张图）
- 快速验证 DINO 特征是否有帮助
- 显存有限

**架构图**：
```
输入图像 ──┬──→ Restormer Encoder ──→ Latent ──→ Decoder ──→ 输出
           │                            ↑
           └──→ DINOv3 (冻结) ──→ 特征对齐 ──┘
```

**代码实现**：
```python
class DINOGuidedRestormer(nn.Module):
    def __init__(self, restormer_config, dino_model='dinov3_vitb16'):
        super().__init__()
        self.restormer = Restormer(**restormer_config)
        
        # 冻结的 DINO
        self.dino = torch.hub.load('facebookresearch/dinov3', dino_model)
        for p in self.dino.parameters():
            p.requires_grad = False
        
        # 特征对齐：DINO dim (768) -> Restormer latent dim (384)
        self.dino_proj = nn.Conv2d(768, 384, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # DINO 特征
        with torch.no_grad():
            dino_out = self.dino.forward_features(self._preprocess(x))
            h, w = H // 16, W // 16
            dino_feat = dino_out[:, 1:].transpose(1, 2).reshape(B, 768, h, w)
        
        # Restormer 编码
        # ... 正常编码流程 ...
        latent = self.restormer.latent(inp_enc_level4)
        
        # 融合 DINO 特征
        dino_aligned = self.dino_proj(dino_feat)
        latent = latent + dino_aligned
        
        # 解码
        # ... 正常解码流程 ...
        return output
```

---

### 3.2 方案二：多尺度特征金字塔 (FPN)

**思路**：从 DINO 不同层提取特征，构建 Feature Pyramid 融合浅层和深层信息。

**适用场景**：
- 需要同时处理去噪（浅层）和色彩恢复（深层）
- 中等规模数据集

**架构图**：
```
DINOv3 Layer 4  ──→ 投影 ──→ 融合到 Encoder Level 1
DINOv3 Layer 8  ──→ 投影 ──→ 融合到 Encoder Level 2
DINOv3 Layer 12 ──→ 投影 ──→ 融合到 Encoder Level 3
```

**代码实现**：
```python
class MultiScaleDINOEncoder(nn.Module):
    def __init__(self, dino_model='dinov3_vitb16', extract_layers=[4, 8, 12]):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov3', dino_model)
        for p in self.dino.parameters():
            p.requires_grad = False
        
        self.extract_layers = extract_layers
        self.dino_dim = 768
        
        # 各层特征投影
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(self.dino_dim, 96, 1),   # 浅层
            nn.Conv2d(self.dino_dim, 192, 1),  # 中层
            nn.Conv2d(self.dino_dim, 384, 1),  # 深层
        ])
    
    def forward(self, x):
        B, C, H, W = x.shape
        h, w = H // 16, W // 16
        
        features = []
        with torch.no_grad():
            x_dino = self.dino.prepare_tokens(x)
            for i, blk in enumerate(self.dino.blocks):
                x_dino = blk(x_dino)
                if i + 1 in self.extract_layers:
                    feat = x_dino[:, 1:].transpose(1, 2).reshape(B, self.dino_dim, h, w)
                    features.append(feat)
        
        return [proj(feat) for feat, proj in zip(features, self.proj_layers)]
```

---

### 3.3 方案三：DINO Encoder + 轻量 Decoder

**思路**：完全用 DINOv3 作为 Encoder，配合专门设计的 Decoder。

**适用场景**：
- 数据充足
- 追求最佳效果
- 可以使用 LoRA 微调

**架构图**：
```
输入图像 ──→ DINOv3 (LoRA) ──→ 多尺度特征 ──→ FPN Decoder ──→ 输出
                                              (PixelShuffle)
```

---

## 4. 关键技术细节

### 4.1 块效应 (Grid Artifacts) 处理

#### 问题根源

ViT 将图像切成 16×16 的 patch，每个 patch 独立编码。上采样时 patch 边界会产生不连续：

```
原图 256×256 → DINO → 16×16 tokens → 直接上采样 → 块状边界 ❌
```

#### 解决方案

**方案 A：PixelShuffle + 卷积平滑**

```python
class SmoothUpsampler(nn.Module):
    """避免块效应的上采样模块"""
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            # 关键：3x3 卷积平滑 patch 边界
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.upsample(x)
```

**方案 B：大卷积核跨 Patch 混合**

```python
class OverlapPatchUpsample(nn.Module):
    """重叠卷积消除边界效应"""
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
        # 5x5 卷积核覆盖多个 patch
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return self.refine(x)
```

**方案 C：专用边界混合模块**

```python
class PatchBoundaryBlender(nn.Module):
    """专门处理 patch 边界的模块"""
    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        # Depthwise conv 跨越 patch 边界
        self.blend = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        # 可学习的边界权重
        self.edge_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        
    def forward(self, x):
        blended = self.blend(x)
        return x + self.edge_weight * (blended - x)
```

---

### 4.2 引导条件方式的具体实现

> **注意**：本节是对 2.3 节中"引导条件"使用方式的详细实现。这不是一个独立的集成方案，而是一种可以应用到方案一、二、三中的特征融合技术。

#### 核心思想

不让 DINO 特征直接参与重建，而是作为 "在哪里增强、增强多少" 的语义指导。DINO 特征生成注意力权重，用于调制 CNN/Restormer 的特征，而非替换它们。

#### 实现方式 A：空间 + 通道注意力引导（推荐）

适用于方案一、二，计算开销小：

```python
class DINOGuidedAttention(nn.Module):
    """DINO 特征作为 Attention 引导"""
    def __init__(self, cnn_dim, dino_dim=768):
        super().__init__()
        # 空间注意力：哪些区域需要增强
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dino_dim, dino_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dino_dim // 4, 1, 1),
            nn.Sigmoid(),
        )
        # 通道注意力：增强哪些特征通道
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dino_dim, cnn_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, cnn_feat, dino_feat):
        # 上采样 DINO 特征到 CNN 尺寸
        dino_up = F.interpolate(dino_feat, size=cnn_feat.shape[-2:], mode='bilinear')
        
        # 计算注意力
        spatial_weight = self.spatial_attn(dino_up)   # [B, 1, H, W]
        channel_weight = self.channel_attn(dino_feat) # [B, C, 1, 1]
        
        # 应用引导（调制而非替换）
        guided_feat = cnn_feat * spatial_weight * channel_weight
        return cnn_feat + guided_feat  # 残差连接保留原始特征
```

#### 实现方式 B：Cross-Attention 引导

适用于需要更精细语义交互的场景，计算开销较大：

```python
class DINOCrossAttentionGuide(nn.Module):
    """用 Cross-Attention 让 CNN 特征查询 DINO 语义"""
    def __init__(self, cnn_dim, dino_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cnn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # CNN 特征生成 Query（主干网络主导）
        self.q_proj = nn.Conv2d(cnn_dim, cnn_dim, 1)
        # DINO 特征生成 Key, Value（提供语义参考）
        self.k_proj = nn.Conv2d(dino_dim, cnn_dim, 1)
        self.v_proj = nn.Conv2d(dino_dim, cnn_dim, 1)
        self.out_proj = nn.Conv2d(cnn_dim, cnn_dim, 1)
        
    def forward(self, cnn_feat, dino_feat):
        B, C, H, W = cnn_feat.shape
        dino_up = F.interpolate(dino_feat, size=(H, W), mode='bilinear')
        
        q = self.q_proj(cnn_feat)   # CNN 特征作为 Query
        k = self.k_proj(dino_up)    # DINO 特征作为 Key
        v = self.v_proj(dino_up)    # DINO 特征作为 Value
        
        # Reshape for attention
        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=H, w=W)
        return cnn_feat + self.out_proj(out)  # 残差连接
```

#### 如何在各方案中应用

**方案一 + 引导条件**：
```python
# 在 Latent 层使用引导
latent = self.restormer.latent(inp_enc_level4)
latent = self.dino_guide(latent, dino_feat)  # 引导而非直接融合
```

**方案二 + 引导条件**：
```python
# 多尺度引导
out_enc_level2 = self.dino_guide_l2(out_enc_level2, dino_feats[1])
out_enc_level3 = self.dino_guide_l3(out_enc_level3, dino_feats[2])
```

**方案三 + 引导条件**：
```python
# DINO 引导独立的 CNN 分支
cnn_feat = self.cnn_encoder(x)
guided_feat = self.dino_guide(cnn_feat, dino_feat)
output = self.decoder(guided_feat)
```

---

### 4.3 低光照输入预处理

DINOv3 在自然图像（ImageNet 分布）上训练，低光照图像数值接近 0，分布差异大。

```python
def preprocess_for_dino(x, gamma=0.4):
    """低光照图像预处理，使其更接近 DINO 训练分布"""
    # 1. Gamma 校正提升暗部
    x_corrected = torch.pow(x.clamp(min=1e-8), gamma)
    
    # 2. ImageNet 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    
    return (x_corrected - mean) / std
```

**实验建议**：
- 尝试不同 gamma 值 (0.3 ~ 0.5)
- 对比直接传入暗图 vs gamma 校正后传入
- 观察 DINO 特征图的激活模式

---

## 5. 训练策略

### 5.1 策略对比

| 策略 | 可训练参数 | 显存占用 | 适用场景 |
|------|-----------|---------|---------|
| 冻结 DINO + 训练 Decoder | ~10M | 低 | 小数据集，快速验证 |
| 冻结 DINO + 训练 Restormer | ~26M | 中 | 中等数据集 |
| LoRA 微调 DINO | ~1% DINO 参数 | 中 | 需要适配暗光特性 |
| 全量微调 | 全部 | 极高 | 大数据集，不推荐 |

### 5.2 策略 A：冻结 Backbone + 训练 Decoder（推荐起步）

```python
# 冻结 DINO
for param in model.dino.parameters():
    param.requires_grad = False

# 只训练融合层和解码器
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
```

### 5.3 策略 B：LoRA 微调

当冻结效果不够好时，使用 LoRA 让 DINO 适应暗光特性：

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # 低秩维度
    lora_alpha=32,           # 缩放因子
    target_modules=["qkv"],  # 只在 Attention 的 QKV 上加
    lora_dropout=0.1,
)

# 应用 LoRA
model.dino = get_peft_model(model.dino, lora_config)

# 此时只训练 ~1% 的参数
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

### 5.4 损失函数建议

```python
class LLIELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()  # VGG-based
        
    def forward(self, pred, target):
        # 像素级损失
        l1_loss = self.l1(pred, target)
        
        # 感知损失（保持语义一致性）
        perceptual_loss = self.perceptual(pred, target)
        
        # 可选：DINO 特征一致性损失
        # dino_loss = self.dino_consistency(pred, target)
        
        return l1_loss + 0.1 * perceptual_loss
```

### 5.5 学习率设置

```python
# 分层学习率
param_groups = [
    {'params': model.dino.parameters(), 'lr': 1e-5},      # DINO (如果用 LoRA)
    {'params': model.fusion.parameters(), 'lr': 2e-4},    # 融合层
    {'params': model.decoder.parameters(), 'lr': 2e-4},   # 解码器
]
optimizer = torch.optim.AdamW(param_groups)
```

---

## 6. 完整代码实现

### 6.1 完整模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SmoothUpsampler(nn.Module):
    """避免块效应的上采样模块"""
    def __init__(self, in_dim, out_dim, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.upsample(x)


class PatchBoundaryBlender(nn.Module):
    """处理 patch 边界的模块"""
    def __init__(self, dim):
        super().__init__()
        self.blend = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.edge_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        
    def forward(self, x):
        blended = self.blend(x)
        return x + self.edge_weight * (blended - x)


class DINOGuidedAttention(nn.Module):
    """DINO 特征引导模块"""
    def __init__(self, cnn_dim, dino_dim=768):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dino_dim, dino_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dino_dim // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dino_dim, cnn_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, cnn_feat, dino_feat):
        dino_up = F.interpolate(dino_feat, size=cnn_feat.shape[-2:], mode='bilinear')
        spatial_weight = self.spatial_attn(dino_up)
        channel_weight = self.channel_attn(dino_feat)
        guided_feat = cnn_feat * spatial_weight * channel_weight
        return cnn_feat + guided_feat


class DINOGuidedLLIENet(nn.Module):
    """完整方案：CNN 主干 + DINO 引导 + 无块效应上采样"""
    def __init__(self, dim=48, dino_model='dinov3_vitb16'):
        super().__init__()
        self.dino_dim = 768
        
        # ===== DINO 分支（冻结）=====
        self.dino = torch.hub.load('facebookresearch/dinov3', dino_model)
        for p in self.dino.parameters():
            p.requires_grad = False
            
        # ===== CNN 主干 =====
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1),
        )
        
        # ===== DINO 引导模块 =====
        self.dino_guide = DINOGuidedAttention(cnn_dim=dim * 8, dino_dim=self.dino_dim)
        
        # ===== Decoder =====
        self.decoder = nn.ModuleList([
            nn.Sequential(SmoothUpsampler(dim * 8, dim * 4), PatchBoundaryBlender(dim * 4)),
            nn.Sequential(SmoothUpsampler(dim * 4, dim * 2), PatchBoundaryBlender(dim * 2)),
            nn.Sequential(SmoothUpsampler(dim * 2, dim), PatchBoundaryBlender(dim)),
        ])
        
        self.output = nn.Conv2d(dim, 3, 3, padding=1)
        
    def _preprocess_for_dino(self, x, gamma=0.4):
        x = torch.pow(x.clamp(min=1e-8), gamma)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
        
    def extract_dino_features(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            x_norm = self._preprocess_for_dino(x)
            dino_out = self.dino.forward_features(x_norm)
            h, w = H // 16, W // 16
            dino_feat = dino_out[:, 1:].transpose(1, 2).reshape(B, self.dino_dim, h, w)
        return dino_feat
        
    def forward(self, x):
        dino_feat = self.extract_dino_features(x)
        cnn_feat = self.cnn_encoder(x)
        
        dino_up = F.interpolate(dino_feat, size=cnn_feat.shape[-2:], mode='bilinear')
        guided_feat = self.dino_guide(cnn_feat, dino_up)
        
        feat = guided_feat
        for dec_block in self.decoder:
            feat = dec_block(feat)
        
        return self.output(feat) + x
```

---

### 6.2 与现有 Restormer 集成

```python
# 在 basicsr/models/archs/ 下创建新文件
# dino_restormer_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .restormer_arch import Restormer, LayerNorm, TransformerBlock


class DINOFeatureExtractor(nn.Module):
    """DINO 特征提取器（支持多尺度）"""
    def __init__(self, model_name='dinov3_vitb16', extract_layers=None):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov3', model_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        
        self.extract_layers = extract_layers or [12]  # 默认只取最后一层
        self.dino_dim = 768  # vitb16
        
    def forward(self, x, gamma=0.4):
        B, C, H, W = x.shape
        h, w = H // 16, W // 16
        
        # 预处理
        x_proc = torch.pow(x.clamp(min=1e-8), gamma)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (x_proc - mean) / std
        
        features = []
        with torch.no_grad():
            x_dino = self.dino.prepare_tokens(x_norm)
            for i, blk in enumerate(self.dino.blocks):
                x_dino = blk(x_dino)
                if i + 1 in self.extract_layers:
                    feat = x_dino[:, 1:].transpose(1, 2).reshape(B, self.dino_dim, h, w)
                    features.append(feat)
        
        return features if len(features) > 1 else features[0]


class DINOGuidedRestormer(nn.Module):
    """DINO 引导的 Restormer"""
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dino_model='dinov3_vitb16',
        dino_inject_level='latent',  # 'latent', 'all', 'decoder'
    ):
        super().__init__()
        
        # 原始 Restormer
        self.restormer = Restormer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        
        # DINO 特征提取器
        self.dino_extractor = DINOFeatureExtractor(dino_model)
        self.dino_dim = 768
        self.dino_inject_level = dino_inject_level
        
        # 特征融合层
        if dino_inject_level == 'latent':
            self.dino_fusion = nn.Sequential(
                nn.Conv2d(self.dino_dim, int(dim * 2**3), 1),
                nn.GELU(),
            )
        elif dino_inject_level == 'all':
            self.dino_fusions = nn.ModuleDict({
                'level2': nn.Conv2d(self.dino_dim, int(dim * 2**1), 1),
                'level3': nn.Conv2d(self.dino_dim, int(dim * 2**2), 1),
                'latent': nn.Conv2d(self.dino_dim, int(dim * 2**3), 1),
            })
        
        # DINO 引导注意力
        self.dino_guide = DINOGuidedAttention(
            cnn_dim=int(dim * 2**3), 
            dino_dim=self.dino_dim
        )
        
    def forward(self, inp_img):
        # 提取 DINO 特征
        dino_feat = self.dino_extractor(inp_img)
        
        # Restormer 编码
        inp_enc_level1 = self.restormer.patch_embed(inp_img)
        out_enc_level1 = self.restormer.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.restormer.down1_2(out_enc_level1)
        out_enc_level2 = self.restormer.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.restormer.down2_3(out_enc_level2)
        out_enc_level3 = self.restormer.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.restormer.down3_4(out_enc_level3)
        latent = self.restormer.latent(inp_enc_level4)
        
        # DINO 特征注入
        if self.dino_inject_level in ['latent', 'all']:
            dino_latent = F.interpolate(dino_feat, size=latent.shape[-2:], mode='bilinear')
            latent = self.dino_guide(latent, dino_latent)

        # Restormer 解码
        inp_dec_level3 = self.restormer.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.restormer.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.restormer.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.restormer.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.restormer.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.restormer.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.restormer.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.restormer.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.restormer.refinement(out_dec_level1)
        out_dec_level1 = self.restormer.output(out_dec_level1) + inp_img

        return out_dec_level1
```

---

## 7. 常见问题与解决方案

### 7.1 显存不足

**问题**：DINOv3 模型较大，显存占用高。

**解决方案**：
```python
# 1. 使用更小的 DINO 模型
dino_model = 'dinov3_vits16'  # 21M 参数，而非 vitb16 的 86M

# 2. 使用 gradient checkpointing
from torch.utils.checkpoint import checkpoint
latent = checkpoint(self.restormer.latent, inp_enc_level4)

# 3. 混合精度训练
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
```

### 7.2 DINO 特征与任务不匹配

**问题**：DINO 在自然图像上训练，低光照图像分布差异大。

**解决方案**：
```python
# 1. 调整 gamma 校正参数
for gamma in [0.3, 0.4, 0.5]:
    feat = extract_dino_features(x, gamma=gamma)
    # 评估特征质量

# 2. 使用 LoRA 微调
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"])
model.dino = get_peft_model(model.dino, lora_config)

# 3. 添加特征适配层
self.dino_adapter = nn.Sequential(
    nn.Conv2d(768, 768, 1),
    nn.GELU(),
    nn.Conv2d(768, 768, 1),
)
```

### 7.3 输出有块效应

**问题**：ViT patch 边界导致输出不连续。

**解决方案**：
```python
# 1. 使用 SmoothUpsampler 替代直接上采样
# 2. 添加 PatchBoundaryBlender
# 3. 在最后输出前添加平滑卷积
self.smooth_output = nn.Sequential(
    nn.Conv2d(dim, dim, 5, padding=2),
    nn.GELU(),
    nn.Conv2d(dim, 3, 3, padding=1),
)
```

### 7.4 训练不稳定

**问题**：DINO 特征与 Restormer 特征尺度不匹配。

**解决方案**：
```python
# 1. 特征归一化
dino_feat = F.layer_norm(dino_feat, dino_feat.shape[1:])

# 2. 使用可学习的融合权重
self.fusion_weight = nn.Parameter(torch.tensor(0.1))
fused = restormer_feat + self.fusion_weight * dino_feat

# 3. 渐进式训练
# 第一阶段：冻结 DINO，只训练融合层
# 第二阶段：解冻 LoRA，联合训练
```

---

## 8. 推荐实施路径

| 阶段 | 方案 | 训练策略 | 预期效果 |
|------|------|----------|----------|
| 1 | 方案一（语义引导） | 冻结 DINO，训练融合层 | 快速验证，基线效果 |
| 2 | 方案二（多尺度 FPN） | 冻结 DINO，训练 FPN | 提升细节恢复 |
| 3 | 方案三 + LoRA | LoRA 微调 DINO | 最佳效果 |

---

## 9. 参考资源

- [DINOv3 论文](https://arxiv.org/abs/2508.10104)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Restormer 论文](https://arxiv.org/abs/2111.09881)
- [HuggingFace DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [PEFT/LoRA 库](https://github.com/huggingface/peft)

---

## 10. 安装依赖

```bash
# PyTorch (CUDA 支持)
pip install torch torchvision

# DINOv3 (通过 timm 或 transformers)
pip install timm>=1.0.20
# 或
pip install transformers>=4.56.0

# LoRA 支持
pip install peft

# 其他依赖
pip install einops
```
