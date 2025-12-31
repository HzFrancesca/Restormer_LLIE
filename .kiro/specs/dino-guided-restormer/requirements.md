# Requirements Document

## Introduction

本规范定义了 DINOv3 语义引导 Restormer 的集成方案（方案一），用于低光照图像增强（LLIE）任务。该方案保留原有 Restormer 架构，将冻结的 DINOv3 特征作为语义引导注入到 Restormer 的 Latent 层，利用 DINO 的高级语义信息指导像素级恢复。

## Glossary

- **DINOv3**: Meta AI 发布的自监督视觉基础模型，能够学习高质量的全局和局部视觉特征
- **Restormer**: 用于高分辨率图像恢复的高效 Transformer 架构
- **LLIE**: Low-Light Image Enhancement，低光照图像增强
- **Latent_Layer**: Restormer 编码器最深层的特征表示层
- **DINO_Feature**: DINOv3 模型提取的 Token 序列特征
- **Guided_Attention**: 使用 DINO 特征生成注意力权重来调制 Restormer 特征的模块
- **Gamma_Correction**: 用于调整低光照图像亮度分布的预处理技术
- **Patch_Boundary**: ViT 将图像切分为 16×16 patch 时产生的边界

## Requirements

### Requirement 1: DINO 特征提取器

**User Story:** As a developer, I want to extract semantic features from DINOv3 model, so that I can use them to guide the Restormer restoration process.

#### Acceptance Criteria

1. WHEN an input image is provided, THE DINO_Feature_Extractor SHALL extract features from the frozen DINOv3 model
2. WHEN extracting features, THE DINO_Feature_Extractor SHALL apply Gamma_Correction preprocessing to adapt low-light images to DINO's training distribution
3. WHEN the DINOv3 model outputs token sequences, THE DINO_Feature_Extractor SHALL reshape them from [B, N_patches+1, Dim] to spatial feature maps [B, Dim, H/16, W/16]
4. THE DINO_Feature_Extractor SHALL keep all DINOv3 parameters frozen (requires_grad=False)
5. WHEN gamma parameter is not specified, THE DINO_Feature_Extractor SHALL use a default gamma value of 0.4

### Requirement 2: DINO 引导注意力模块

**User Story:** As a developer, I want DINO features to guide Restormer features through attention mechanism, so that semantic information can enhance pixel-level restoration without replacing spatial details.

#### Acceptance Criteria

1. WHEN DINO features and Restormer features are provided, THE Guided_Attention SHALL generate spatial attention weights indicating which regions need enhancement
2. WHEN DINO features and Restormer features are provided, THE Guided_Attention SHALL generate channel attention weights indicating which feature channels to enhance
3. THE Guided_Attention SHALL use sigmoid activation to produce attention weights in range [0, 1]
4. THE Guided_Attention SHALL apply residual connection to preserve original Restormer features
5. WHEN DINO feature spatial size differs from Restormer feature size, THE Guided_Attention SHALL upsample DINO features using bilinear interpolation

### Requirement 3: DINOv3 引导的 Restormer 集成

**User Story:** As a developer, I want to integrate DINO guidance into Restormer architecture, so that I can leverage semantic priors for better low-light image enhancement.

#### Acceptance Criteria

1. THE DINOGuidedRestormer SHALL preserve the original Restormer encoder-decoder architecture
2. WHEN processing an image, THE DINOGuidedRestormer SHALL inject DINO guidance at the Latent_Layer level
3. THE DINOGuidedRestormer SHALL project DINO features (768-dim for ViT-B) to match Restormer latent dimension
4. THE DINOGuidedRestormer SHALL support configurable DINO model variants (vits14, vitb14, vitl14, vitg14)
5. WHEN forward pass completes, THE DINOGuidedRestormer SHALL output enhanced image with residual connection to input

### Requirement 4: 低光照图像预处理

**User Story:** As a developer, I want to preprocess low-light images before feeding to DINOv3, so that the features extracted are more meaningful for the enhancement task.

#### Acceptance Criteria

1. WHEN preprocessing low-light images, THE Preprocessor SHALL apply gamma correction to boost dark regions
2. WHEN preprocessing, THE Preprocessor SHALL apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. WHEN input values are near zero, THE Preprocessor SHALL clamp minimum values to 1e-8 to avoid numerical issues
4. THE Preprocessor SHALL support configurable gamma values between 0.3 and 0.5

### Requirement 5: 配置文件支持

**User Story:** As a user, I want to configure the DINO-guided Restormer through YAML config files, so that I can easily experiment with different settings.

#### Acceptance Criteria

1. THE Configuration SHALL support specifying DINO model variant (vits14, vitb14, vitl14, vitg14)
2. THE Configuration SHALL support specifying gamma correction value for preprocessing
3. THE Configuration SHALL support enabling/disabling DINO guidance
4. THE Configuration SHALL be compatible with existing Restormer training pipeline
5. WHEN DINO guidance is disabled, THE DINOGuidedRestormer SHALL behave identically to original Restormer

### Requirement 6: 模型兼容性

**User Story:** As a developer, I want the new architecture to be compatible with existing training infrastructure, so that I can reuse the current training pipeline.

#### Acceptance Criteria

1. THE DINOGuidedRestormer SHALL be registered in the architecture module for dynamic instantiation
2. THE DINOGuidedRestormer SHALL accept the same input/output format as original Restormer
3. THE DINOGuidedRestormer SHALL support loading pretrained Restormer weights for the base architecture
4. WHEN loading pretrained weights, THE DINOGuidedRestormer SHALL handle missing DINO-related keys gracefully
