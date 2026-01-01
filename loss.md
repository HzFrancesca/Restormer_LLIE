针对 **Restormer + DINOv3 H+** 的组合，损失函数的设计需要兼顾三个层面：**像素级还原**、**人眼视觉感知**以及**语义一致性**。

由于你引入了 DINOv3，最核心的创新点在于**语义一致性损失（Semantic Consistency Loss）**，这能防止模型在提亮暗部时产生“幻觉”或破坏物体结构。

以下是为你定制的组合损失函数方案：

---

### 1. 总体损失公式

建议采用 **复合损失函数 (Composite Loss)**：

**推荐权重 () 初始化：**

*  (基础还原)
*  (结构保持)
*  (视觉纹理)
*  (语义约束，数值较小是因为 Feature 数值通常较大)

---

### 2. 各部分详解与实现

#### A. 重建损失 (Reconstruction Loss) - 推荐 Charbonnier Loss

不要使用标准的 MSE (L2) 或 L1。Restormer 原论文使用的是 **Charbonnier Loss** (L1 的平滑变体)。它比 L2 更能保留边缘，比 L1 收敛更稳定。

```python
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

```

#### B. 结构相似性损失 (SSIM Loss)

低光照增强容易导致局部对比度失真。SSIM Loss 强制模型关注局部的亮度、对比度和结构，而不仅仅是像素差值。

```python
# 可以直接使用 pytorch_msssim 库
# pip install pytorch-msssim
from pytorch_msssim import SSIM

# 初始化
criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
# 使用: loss_ssim = 1 - criterion_ssim(output, target)

```

#### C. 感知损失 (Perceptual Loss / VGG Loss)

利用预训练的 VGG-19 网络提取特征，比较生成图和 GT 图在“特征空间”的距离。这有助于恢复更符合人眼习惯的**纹理细节**。

#### D. 语义一致性损失 (DINO Identity Loss) —— **关键创新**

既然你已经加载了 DINOv3，**必须**利用它来计算 Loss。
**原理：** 增强后的图像（Output）在 DINOv3 眼中，应该和 Ground Truth（GT）看起来是一样的物体。这能有效抑制伪影和怪异的色彩斑块。

*注意：这里不需要梯度回传给 DINO，DINO 仅作为“裁判”。*

---

### 3. 完整的 Loss Module 代码

你可以直接复制这个模块到你的训练代码中：

```python
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_msssim import SSIM

class DINORestormerLoss(nn.Module):
    def __init__(self, dino_model, device, lambda_rec=1.0, lambda_ssim=1.0, lambda_per=0.1, lambda_sem=0.05):
        super().__init__()
        self.device = device
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_per = lambda_per
        self.lambda_sem = lambda_sem

        # 1. Charbonnier Loss
        self.char_loss = CharbonnierLoss()

        # 2. SSIM Loss
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

        # 3. Perceptual Loss (VGG19)
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
        # 4. DINO Model (引用传入的 DINO 模型实例)
        self.dino = dino_model
        # 确保 DINO 处于 eval 模式且不计算梯度
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, pred, gt):
        # --- 1. Pixel Level ---
        l_rec = self.char_loss(pred, gt)
        l_ssim = 1 - self.ssim_loss(pred, gt)

        # --- 2. Perceptual Level (VGG) ---
        # 归一化到 VGG 需要的区间 (假设输入是 0-1)
        # 简单的做法是直接传，或者做标准的 ImageNet 归一化
        pred_vgg = self.vgg_layers(pred)
        gt_vgg = self.vgg_layers(gt)
        l_per = nn.functional.l1_loss(pred_vgg, gt_vgg)

        # --- 3. Semantic Level (DINO) ---
        # 提取 DINO 特征进行对比
        # 技巧：使用 Cosine Similarity 而不是 L1
        # 因为光照变化可能会影响特征向量的模长(Magnitude)，但语义主要由方向(Direction)决定
        with torch.no_grad():
            # 获取 GT 的 DINO 特征 (作为 Target)
            gt_dino_out = self.dino.forward_features(gt)
            gt_tokens = gt_dino_out['x_norm_patchtokens'] # [B, N, D]

        # 获取 Pred 的 DINO 特征 (需要梯度)
        pred_dino_out = self.dino.forward_features(pred)
        pred_tokens = pred_dino_out['x_norm_patchtokens'] # [B, N, D]

        # 计算 Cosine Embedding Loss (目标是让它们方向一致，即 label=1)
        # Flatten 为 (B*N, D) 以便计算
        target = torch.ones(pred_tokens.shape[0] * pred_tokens.shape[1]).to(self.device)
        l_sem = nn.functional.cosine_embedding_loss(
            pred_tokens.flatten(0, 1), 
            gt_tokens.flatten(0, 1), 
            target
        )

        # --- Total Loss ---
        loss_total = (self.lambda_rec * l_rec) + \
                     (self.lambda_ssim * l_ssim) + \
                     (self.lambda_per * l_per) + \
                     (self.lambda_sem * l_sem)

        return loss_total, {
            "rec": l_rec.item(),
            "ssim": l_ssim.item(),
            "per": l_per.item(),
            "sem": l_sem.item()
        }

```

### 4. 避坑指南

1. **显存爆炸：** DINOv3 H+ 和 VGG19 同时跑可能会吃满显存。
* *解决：* 如果显存不够，可以去掉 VGG Loss (`lambda_per=0`)，因为 DINO Loss 某种程度上已经包含了高层特征对比。


2. **DINO 输入尺寸：** 计算 DINO Loss 时，如果你为了节省显存把图片 Resize 小了，记得 `pred` 和 `gt` 都要 Resize 到一样的大小再喂给 DINO。
3. **Color Loss (可选)：** 如果你发现训练出来的图颜色偏灰或饱和度不够，可以额外加一个 **Color Loss**（计算 RGB 向量夹角或直方图距离），但在 DINO 介入的情况下，通常语义对齐了，颜色也就不会偏太远。

### 总结

使用 **Charbonnier (保底)** + **SSIM (结构)** + **DINO Cosine (语义)** 是你这个架构最强的组合。特别是 DINO Cosine Loss，它利用了 DINO "忽略光照、只看物体" 的特性，非常适合低光增强任务。