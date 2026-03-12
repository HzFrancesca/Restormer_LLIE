# Restormer LLIE 单卡分布式等效训练优化报告 (RTX 4090 24GB)

本报告总结了针对单张 RTX 4090 (24GB 显存) 优化 Restormer 模型训练的配置策略、显存测试结果及其背后的设计逻辑。

## 1. 核心挑战与策略

**原论文环境**：8 张 A100/V100。
**目标**：在单卡 4090 上实现与 8 卡集群等效（甚至更优）的训练质量，同时兼顾训练速度。

### 核心技术栈

* **Gradient Checkpointing (梯度检查点)**：在 `network_g` 中开启，通过空间换时间，将大分辨率下的显存占用降低 60% 以上。
* **Gradient Accumulation (梯度累积)**：通过物理 Batch Size 与累积步数的乘积，模拟大规模分布式训练的总 Batch Size（目标 BS=32~64）。
* **Progressive Learning (渐进式学习)**：随训练阶段动态调整 Patch Size、Batch Size 和 Accumulation Steps。

---

## 2. 核心概念与逻辑关系

在分布式等效训练中，理解以下四个参数的数学关系至关重要：

### 2.1 核心公式

* **等效总 Batch Size (Effective Batch)** = 物理 Batch Size $\times$ 梯度累积步数 (Accumulation Steps)
* **总样本曝光量 (Total Samples Seen)** = Iterations (参数更新次数) $\times$ 等效总 Batch Size

### 2.2 概念解析

* **物理 Batch Size**：显卡单次向前传播（Forward）处理的图片数，严格受物理显存（24G）限制。
* **梯度累积步数**：在执行 `optimizer.step()` 更新权重前，连续计算并累加梯度的次数。这是单卡模拟大集群的关键（相当于“嚼多口再咽下”）。
* **参数更新次数 (Iterations)**：代码中 `optimizer.step()` 实际执行的次数。
  * **注意**：累计步数越高，完成一次 Iteration 所需的物理时间就越长。例如 Accum = 8 时，每一步 Iteration 的时间是 Accum = 1 的 8 倍。

---

## 3. 显存压力测试结果

基于显存压力测试脚本，我们对物理 Batch Size (BS) 在不同 Patch Size (GT) 下的占用进行了实测：

| 阶段 | Patch Size | 物理 Batch Size | 显存占用 (实测) | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | 128 | 48 | **21.1 GB** | 安全 (留有 2G+ 缓冲) |
| **Stage 2** | 160 | 32 | **21.8 GB** | 安全 |
| **Stage 3** | 192 | 20 | **20.0 GB** | 安全 |
| **Stage 4** | 256 | 12 | **21.0 GB** | 安全 |
| **Stage 5** | 320 | 8 | **22.0 GB** | **极限临界** |
| **Stage 6** | 384 | 4 | **16.2 GB** | 宽松 |

**设计准则**：物理 Batch Size 优先选择 **8 的倍数**（或 4 的倍数）以触发 NVIDIA Tensor Cores 的硬件加速。

---

## 4. 训练方案设计：方案 A (平衡与速度优先)

为了避免后期（GT 320/384）更新步长太慢，我们采用了 **方案 A**。

### 4.1 动态步长对齐表

我们将总 Batch Size 稳定在 **32~48** 之间，并据此重新分配了迭代次数：

| 阶段 | Patch Size | 物理 BS | 累积步数 (Accum) | 等效总 BS | 建议 Iterations | 样本曝光量 (vs 原版) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 128 | 48 | 1 | **48** | 120,000 | 1.0x (原64) |
| 2 | 160 | 32 | 1 | **32** | 70,000 | 0.8x |
| 3 | 192 | 20 | 2 | **40** | 40,000 | 1.25x |
| 4 | 256 | 12 | 3 | **36** | 30,000 | 2.25x |
| 5 | 320 | 8 | 4 | **32** | 25,000 | 4.0x |
| 6 | 384 | 4 | 8 | **32** | 15,000 | 4.0x |

### 4.2 学习率周期设计 (Cosine Annealing)

* **Total Iterations**: 300,000 / 100,000
* **Periods**: 划分为两个阶段。
  * **前期 (40%)**: 固定高学习率 (`3e-4`)，对应小 Patch 基础特征学习。
  * **后期 (60%)**: 随 Patch 增大进行余弦退火降噪。

---

## 5. 关键配置项 (YAML 参数)

为确保代码正确执行单卡优化，必须检查以下配置：

1. **`batch_size_per_gpu`**: 必须设为 **48** (或大于等于 `mini_batch_sizes` 的第一个值)。
2. **`mini_accumulation_steps`**: 列表长度必须与 `mini_batch_sizes` 一致。
3. **`use_checkpoint: True`**: 必须在 `network_g` 中开启。
4. **`total_iter` & `periods`**: 必须计算 `iters` 列表的总和以保持对齐。

---

## 6. 文件清单与用途

| 文件名 | 适用场景 | 训练强度 |
| :--- | :--- | :--- |
| `Restormer_LOLv2_100k_1gpu.yml` | **快速实验**，Thesis 早期验证 | 低 (10w 步) |
| `LowLight_Restormer_pro_1gpu.yml` | 进阶版模型，针对 4090 深度优化 | 高 (30w 步) |

---

## 7. 维护建议

* **OOM 处理**：若训练过程中因系统显存占用波动导致 Out of Memory，请将对应阶段的 `mini_batch_sizes` **减 4**，同时保持 `batch_size_per_gpu` 的同步修改。
* **收敛观察**：由于使用了梯度累积，`Iteration` 时长会随阶段增加。建议通过 Tensorboard 观察 `current_lr` 与 `l_pix` 的对应关系。
