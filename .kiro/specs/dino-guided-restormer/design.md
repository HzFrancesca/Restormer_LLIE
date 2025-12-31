# Design Document: DINOv3 Guided Restormer (方案一)

## Overview

本设计实现 DINOv3 语义引导 Restormer 的集成方案，用于低光照图像增强任务。核心思路是保留原有 Restormer 架构，将冻结的 DINOv3 特征作为"引导条件"注入到 Latent 层，利用 DINO 的高级语义信息指导像素级恢复，而非直接替换 Restormer 的空间特征。

### 设计原则

1. **DINO 作为引导而非主干**：保留 Restormer 的空间细节处理能力
2. **引导条件方式**：DINO 生成注意力权重调制特征，而非直接融合
3. **冻结 DINO**：减少训练参数，适合小数据集（如 LOL）
4. **最小侵入性**：尽量复用现有 Restormer 代码

## Architecture

```
输入图像 [B, 3, H, W]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
Restormer Encoder                    DINOv3 (冻结)
    │                                          │
    ├─ Level 1: [B, 48, H, W]                  │
    ├─ Level 2: [B, 96, H/2, W/2]              │
    ├─ Level 3: [B, 192, H/4, W/4]             │
    └─ Level 4: [B, 384, H/8, W/8]             │
         │                                     │
         ▼                                     ▼
    Latent Layer ◄─── DINO Guided ◄─── DINO Features
    [B, 384, H/8, W/8]   Attention      [B, 768, H/16, W/16]
         │                                     
         ▼                                     
Restormer Decoder                              
    │                                          
    └─► 输出图像 [B, 3, H, W]                   
```

## Components and Interfaces

### Component 1: DINOFeatureExtractor

负责从冻结的 DINOv3 模型提取语义特征。

```python
class DINOFeatureExtractor(nn.Module):
    """
    从 DINOv3 提取语义特征
    
    Attributes:
        dino: 冻结的 DINOv3 模型
        dino_dim: DINO 输出维度 (384 for vits14, 768 for vitb14, 1024 for vitl14)
        gamma: 低光照预处理的 gamma 值
    
    Methods:
        forward(x): 提取 DINO 特征
        _preprocess(x): 低光照图像预处理
    """
    
    def __init__(self, model_name: str = 'dinov2_vitb14', gamma: float = 0.4):
        ...
    
    def _preprocess(self, x: Tensor, gamma: float) -> Tensor:
        """
        低光照图像预处理
        
        Args:
            x: 输入图像 [B, 3, H, W], 值域 [0, 1]
            gamma: gamma 校正值
        
        Returns:
            预处理后的图像，已应用 gamma 校正和 ImageNet 归一化
        """
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        """
        提取 DINO 特征
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            DINO 特征图 [B, dino_dim, H/14, W/14]
        """
        ...
```

### Component 2: DINOGuidedAttention

使用 DINO 特征生成注意力权重来调制 Restormer 特征。

```python
class DINOGuidedAttention(nn.Module):
    """
    DINO 引导注意力模块
    
    使用 DINO 特征生成空间和通道注意力权重，
    调制 Restormer 特征而非直接替换。
    
    Attributes:
        spatial_attn: 空间注意力生成器
        channel_attn: 通道注意力生成器
    """
    
    def __init__(self, cnn_dim: int, dino_dim: int = 768):
        ...
    
    def forward(self, cnn_feat: Tensor, dino_feat: Tensor) -> Tensor:
        """
        应用 DINO 引导注意力
        
        Args:
            cnn_feat: Restormer 特征 [B, C, H, W]
            dino_feat: DINO 特征 [B, dino_dim, h, w]
        
        Returns:
            引导后的特征 [B, C, H, W]，包含残差连接
        """
        ...
```

### Component 3: DINOGuidedRestormer

集成 DINO 引导的完整 Restormer 模型。

```python
class DINOGuidedRestormer(nn.Module):
    """
    DINOv3 引导的 Restormer
    
    在 Latent 层注入 DINO 语义引导，保留原始 Restormer 架构。
    
    Attributes:
        restormer: 原始 Restormer 模块（复用现有代码）
        dino_extractor: DINO 特征提取器
        dino_proj: DINO 特征投影层
        dino_guide: DINO 引导注意力模块
        use_dino_guidance: 是否启用 DINO 引导
    """
    
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: List[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        dino_model: str = 'dinov2_vitb14',
        dino_gamma: float = 0.4,
        use_dino_guidance: bool = True,
    ):
        ...
    
    def forward(self, inp_img: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            inp_img: 输入图像 [B, 3, H, W]
        
        Returns:
            增强后的图像 [B, 3, H, W]
        """
        ...
```

## Data Models

### DINO 模型配置

| Model Name | Patch Size | Embedding Dim | Params |
|------------|------------|---------------|--------|
| dinov2_vits14 | 14 | 384 | 21M |
| dinov2_vitb14 | 14 | 768 | 86M |
| dinov2_vitl14 | 14 | 1024 | 300M |
| dinov2_vitg14 | 14 | 1536 | 1.1B |

### 特征维度映射

```python
DINO_DIM_MAP = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

# Restormer Latent 层维度 = dim * 2^3 = 48 * 8 = 384
RESTORMER_LATENT_DIM = lambda dim: int(dim * 2**3)
```

### 配置文件格式

```yaml
# YAML 配置示例
network_g:
  type: DINOGuidedRestormer
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: false
  LayerNorm_type: WithBias
  # DINO 相关配置
  dino_model: dinov2_vitb14
  dino_gamma: 0.4
  use_dino_guidance: true
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: DINO Feature Shape Invariant

*For any* input image of shape [B, 3, H, W] where H and W are divisible by 14, the DINO feature extractor SHALL produce output of shape [B, dino_dim, H/14, W/14].

**Validates: Requirements 1.1, 1.3**

### Property 2: Attention Weights Range Invariant

*For any* pair of CNN features and DINO features, the guided attention module SHALL produce spatial attention weights in shape [B, 1, H, W] and channel attention weights in shape [B, C, 1, 1], with all values in range [0, 1].

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 3: Preprocessing Transformation

*For any* input image with values in [0, 1], the preprocessor SHALL:
- Apply gamma correction: output = input^gamma
- Clamp minimum to 1e-8 before gamma correction
- Apply ImageNet normalization after gamma correction

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 4: Input/Output Format Consistency

*For any* input image of shape [B, 3, H, W], the DINOGuidedRestormer SHALL produce output of the same shape [B, 3, H, W].

**Validates: Requirements 3.5, 6.2**

### Property 5: DINO Parameters Frozen

*For all* parameters in the DINO model, requires_grad SHALL be False after initialization.

**Validates: Requirements 1.4**

### Property 6: Residual Connection Preservation

*For any* input to the guided attention module, the output SHALL equal input + guided_features, preserving the original features through residual connection.

**Validates: Requirements 2.4**

### Property 7: Feature Projection Dimension

*For any* DINO feature of dimension dino_dim, the projection layer SHALL produce output matching Restormer latent dimension (dim * 8).

**Validates: Requirements 3.3**

### Property 8: Disabled Guidance Equivalence

*For any* input image, when use_dino_guidance=False, the DINOGuidedRestormer output SHALL be identical to original Restormer output.

**Validates: Requirements 5.5**

## Error Handling

### DINO 模型加载失败

```python
try:
    self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load DINO model '{model_name}': {e}")
```

### 输入尺寸不兼容

```python
def _check_input_size(self, x: Tensor):
    _, _, H, W = x.shape
    patch_size = 14  # DINOv2 uses 14x14 patches
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"Input size ({H}, {W}) must be divisible by patch size {patch_size}"
        )
```

### 权重加载兼容性

```python
def load_pretrained_restormer(self, checkpoint_path: str, strict: bool = False):
    """
    加载预训练 Restormer 权重，忽略 DINO 相关的缺失键
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # 过滤掉 DINO 相关的键
    restormer_keys = {k: v for k, v in state_dict.items() 
                      if not k.startswith('dino')}
    self.load_state_dict(restormer_keys, strict=False)
```

## Testing Strategy

### 单元测试

1. **DINOFeatureExtractor 测试**
   - 测试特征提取输出形状
   - 测试预处理函数的 gamma 校正
   - 测试参数冻结状态

2. **DINOGuidedAttention 测试**
   - 测试注意力权重范围 [0, 1]
   - 测试残差连接
   - 测试不同尺寸输入的上采样

3. **DINOGuidedRestormer 测试**
   - 测试前向传播输出形状
   - 测试禁用 DINO 引导时的行为
   - 测试权重加载兼容性

### 属性测试

使用 `hypothesis` 库进行属性测试，每个属性至少运行 100 次迭代。

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    height=st.sampled_from([112, 224, 336]),  # 必须是 14 的倍数
    width=st.sampled_from([112, 224, 336]),
)
def test_dino_feature_shape_invariant(batch_size, height, width):
    """Property 1: DINO Feature Shape Invariant"""
    # Feature: dino-guided-restormer, Property 1: DINO Feature Shape Invariant
    ...
```

### 集成测试

1. 测试完整训练流程（1 个 epoch）
2. 测试推理流程
3. 测试配置文件解析
