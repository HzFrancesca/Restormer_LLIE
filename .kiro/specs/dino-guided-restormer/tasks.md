# Implementation Plan: DINOv3 Guided Restormer

## Overview

本实现计划将 DINOv3 语义引导集成到 Restormer 架构中，用于低光照图像增强。实现采用增量方式，先构建基础组件，再逐步集成。

## Tasks

- [x] 1. 创建 DINO 特征提取器模块
  - [x] 1.1 实现 DINOFeatureExtractor 类
    - 创建 `basicsr/models/archs/dino_guided_restormer_arch.py` 文件
    - 实现 DINO 模型加载（使用 torch.hub）
    - 实现参数冻结逻辑
    - 实现 `_preprocess` 方法（gamma 校正 + ImageNet 归一化）
    - 实现 `forward` 方法（token 序列转空间特征图）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 1.2 编写 DINOFeatureExtractor 属性测试
    - **Property 1: DINO Feature Shape Invariant**
    - **Property 3: Preprocessing Transformation**
    - **Property 5: DINO Parameters Frozen**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3**

- [x] 2. 创建 DINO 引导注意力模块
  - [x] 2.1 实现 DINOGuidedAttention 类
    - 实现空间注意力生成器（Conv -> GELU -> Conv -> Sigmoid）
    - 实现通道注意力生成器（AdaptiveAvgPool -> Conv -> Sigmoid）
    - 实现 `forward` 方法（上采样 + 注意力调制 + 残差连接）
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 2.2 编写 DINOGuidedAttention 属性测试
    - **Property 2: Attention Weights Range Invariant**
    - **Property 6: Residual Connection Preservation**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. 实现 DINOGuidedRestormer 集成模型
  - [x] 3.1 实现 DINOGuidedRestormer 类
    - 复用现有 Restormer 组件（patch_embed, encoder, decoder 等）
    - 集成 DINOFeatureExtractor
    - 实现 DINO 特征投影层（dino_dim -> latent_dim）
    - 集成 DINOGuidedAttention 到 Latent 层
    - 实现 `use_dino_guidance` 开关
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x] 3.2 编写 DINOGuidedRestormer 属性测试
    - **Property 4: Input/Output Format Consistency**
    - **Property 7: Feature Projection Dimension**
    - **Property 8: Disabled Guidance Equivalence**
    - **Validates: Requirements 3.3, 3.5, 5.5, 6.2**

- [x] 4. Checkpoint - 验证核心功能
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. 实现配置和兼容性支持
  - [x] 5.1 更新架构模块注册
    - 在 `basicsr/models/archs/__init__.py` 中确保新架构可被动态加载
    - _Requirements: 6.1_
  - [x] 5.2 实现预训练权重加载
    - 实现 `load_pretrained_restormer` 方法
    - 处理缺失的 DINO 相关键
    - _Requirements: 6.3, 6.4_
  - [x] 5.3 创建配置文件模板
    - 创建 `LLIE/Options/LowLight_DINORestormer.yml` 配置文件
    - 包含 DINO 相关配置项
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Final Checkpoint - 完整验证
  - 确保所有测试通过，如有问题请询问用户

## Notes

- All tasks are required including property tests
- 使用 `hypothesis` 库进行属性测试
- DINO 模型通过 `torch.hub` 加载，首次运行需要下载
- 默认使用 `dinov2_vitb14` 模型（768 维特征）
