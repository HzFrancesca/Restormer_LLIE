"""
可视化 Restormer Encoder Level 1 Block 0 的完整流程
支持 MDTA, HTA, WTA 三种注意力机制
可视化: Patch Embedding, Block Input, Q, K, V, Attention Map, Attention*V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision import transforms
import os
import sys
import argparse
import numbers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============== Layer Norm ==============

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ============== Patch Embed ==============

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ============== 带中间输出的注意力模块 ==============

class MDTAWithOutputs(nn.Module):
    """MDTA with Q, K, V, Attention Map, and Attention*V outputs"""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 保存中间结果
        self.q_before_rearrange = None  # Q before rearrange: [B, C, H, W]
        self.k_before_rearrange = None  # K before rearrange: [B, C, H, W]
        self.v_before_rearrange = None  # V before rearrange: [B, C, H, W]
        self.q_after_rearrange = None   # Q after rearrange: [B, head, ...]
        self.k_after_rearrange = None   # K after rearrange: [B, head, ...]
        self.v_after_rearrange = None   # V after rearrange: [B, head, ...]
        self.attn_map = None
        self.attn_v_before_rearrange = None  # Attention*V BEFORE rearrange BEFORE project_out
        self.attn_v_after_rearrange = None   # Attention*V AFTER rearrange BEFORE project_out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # 保存 rearrange 之前的 Q, K, V (保留空间结构 [B, C, H, W])
        self.q_before_rearrange = q.clone().detach()
        self.k_before_rearrange = k.clone().detach()
        self.v_before_rearrange = v.clone().detach()
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        
        # 保存 rearrange 之后的 Q, K, V
        self.q_after_rearrange = q.clone().detach()
        self.k_after_rearrange = k.clone().detach()
        self.v_after_rearrange = v.clone().detach()
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        
        # 保存中间结果
        self.attn_map = attn.detach()
        self.attn_v_before_rearrange = out.detach()  # BEFORE rearrange BEFORE project_out
        
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        self.attn_v_after_rearrange = out.detach()  # AFTER rearrange BEFORE project_out
        
        return self.project_out(out)


class HTAWithOutputs(nn.Module):
    """HTA with Q, K, V, Attention Map, and Attention*V outputs"""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 保存中间结果
        self.q_before_rearrange = None  # Q before rearrange: [B, C, H, W]
        self.k_before_rearrange = None  # K before rearrange: [B, C, H, W]
        self.v_before_rearrange = None  # V before rearrange: [B, C, H, W]
        self.q_after_rearrange = None   # Q after rearrange: [B, head, ...]
        self.k_after_rearrange = None   # K after rearrange: [B, head, ...]
        self.v_after_rearrange = None   # V after rearrange: [B, head, ...]
        self.attn_map = None
        self.attn_v_before_rearrange = None  # Attention*V BEFORE rearrange BEFORE project_out
        self.attn_v_after_rearrange = None   # Attention*V AFTER rearrange BEFORE project_out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # 保存 rearrange 之前的 Q, K, V (保留空间结构 [B, C, H, W])
        self.q_before_rearrange = q.clone().detach()
        self.k_before_rearrange = k.clone().detach()
        self.v_before_rearrange = v.clone().detach()
        
        q = rearrange(q, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        
        # 保存 rearrange 之后的 Q, K, V
        self.q_after_rearrange = q.clone().detach()
        self.k_after_rearrange = k.clone().detach()
        self.v_after_rearrange = v.clone().detach()
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        
        self.attn_map = attn.detach()
        self.attn_v_before_rearrange = out.detach()  # BEFORE rearrange BEFORE project_out
        
        out = rearrange(out, "b head w (c h) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        self.attn_v_after_rearrange = out.detach()  # AFTER rearrange BEFORE project_out
        
        return self.project_out(out)


class WTAWithOutputs(nn.Module):
    """WTA with Q, K, V, Attention Map, and Attention*V outputs"""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 保存中间结果
        self.q_before_rearrange = None  # Q before rearrange: [B, C, H, W]
        self.k_before_rearrange = None  # K before rearrange: [B, C, H, W]
        self.v_before_rearrange = None  # V before rearrange: [B, C, H, W]
        self.q_after_rearrange = None   # Q after rearrange: [B, head, ...]
        self.k_after_rearrange = None   # K after rearrange: [B, head, ...]
        self.v_after_rearrange = None   # V after rearrange: [B, head, ...]
        self.attn_map = None
        self.attn_v_before_rearrange = None  # Attention*V BEFORE rearrange BEFORE project_out
        self.attn_v_after_rearrange = None   # Attention*V AFTER rearrange BEFORE project_out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # 保存 rearrange 之前的 Q, K, V (保留空间结构 [B, C, H, W])
        self.q_before_rearrange = q.clone().detach()
        self.k_before_rearrange = k.clone().detach()
        self.v_before_rearrange = v.clone().detach()
        
        q = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        
        # 保存 rearrange 之后的 Q, K, V
        self.q_after_rearrange = q.clone().detach()
        self.k_after_rearrange = k.clone().detach()
        self.v_after_rearrange = v.clone().detach()
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        
        self.attn_map = attn.detach()
        self.attn_v_before_rearrange = out.detach()  # BEFORE rearrange BEFORE project_out
        
        out = rearrange(out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        self.attn_v_after_rearrange = out.detach()  # AFTER rearrange BEFORE project_out
        
        return self.project_out(out)


ATTENTION_CLASSES = {
    "MDTA": MDTAWithOutputs,
    "HTA": HTAWithOutputs,
    "WTA": WTAWithOutputs,
}


# ============== 简化的 Encoder Level 1 Block 0 模型 ==============

class FirstBlockVisualizer(nn.Module):
    """只包含 Patch Embedding 和 Encoder Level 1 的第一个 Block"""
    def __init__(self, attn_type="MDTA", bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.attn_type = attn_type
        
        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=48, bias=bias)
        
        # Encoder Level 1 Block 0
        self.norm1 = LayerNorm(48, LayerNorm_type)
        self.attn = ATTENTION_CLASSES[attn_type](48, 1, bias)
        
    def forward(self, x):
        # Patch Embedding
        patch_feat = self.patch_embed(x)
        
        # Block 0 Input (before LayerNorm)
        block_input = patch_feat.clone()
        
        # LayerNorm + Attention
        normed = self.norm1(patch_feat)
        attn_out = self.attn(normed)
        
        result = {
            "patch_feat": patch_feat,
            "block_input": block_input,
            "q_before_rearrange": self.attn.q_before_rearrange,
            "k_before_rearrange": self.attn.k_before_rearrange,
            "v_before_rearrange": self.attn.v_before_rearrange,
            "q_after_rearrange": self.attn.q_after_rearrange,
            "k_after_rearrange": self.attn.k_after_rearrange,
            "v_after_rearrange": self.attn.v_after_rearrange,
            "attn_map": self.attn.attn_map,
            "attn_v_before_rearrange": self.attn.attn_v_before_rearrange,  # BEFORE rearrange BEFORE project_out
            "attn_v_after_rearrange": self.attn.attn_v_after_rearrange,    # AFTER rearrange BEFORE project_out
            "attn_output": attn_out,
        }
        
        return result
    
    def load_from_checkpoint(self, checkpoint_path, device):
        """从 Restormer checkpoint 加载权重"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # 加载 patch_embed
        patch_dict = {k.replace("patch_embed.", ""): v 
                      for k, v in state_dict.items() if k.startswith("patch_embed.")}
        if patch_dict:
            self.patch_embed.load_state_dict(patch_dict, strict=False)
            print(f"Loaded: patch_embed")
        
        # 加载 encoder_level1.0 (第一个 block)
        block_prefix = "encoder_level1.0."
        
        # LayerNorm
        norm1_dict = {k.replace(f"{block_prefix}norm1.", ""): v 
                      for k, v in state_dict.items() if k.startswith(f"{block_prefix}norm1.")}
        if norm1_dict:
            self.norm1.load_state_dict(norm1_dict, strict=False)
            print(f"Loaded: encoder_level1.0.norm1")
        
        # Attention
        attn_dict = {k.replace(f"{block_prefix}attn.", ""): v 
                     for k, v in state_dict.items() if k.startswith(f"{block_prefix}attn.")}
        if attn_dict:
            self.attn.load_state_dict(attn_dict, strict=False)
            print(f"Loaded: encoder_level1.0.attn ({self.attn_type})")
        
        return True


# ============== 可视化函数 ==============

def load_image(image_path, no_resize=True):
    img = Image.open(image_path).convert("RGB")
    if no_resize:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    return transform(img).unsqueeze(0)


def compute_l2_norm(tensor):
    """计算 L2 Norm: sqrt(sum(x^2)) across channel dimension"""
    if tensor.dim() == 4:  # [B, C, H, W]
        return torch.sqrt((tensor[0] ** 2).sum(dim=0)).cpu().numpy()
    elif tensor.dim() == 3:  # [C, H, W]
        return torch.sqrt((tensor ** 2).sum(dim=0)).cpu().numpy()
    else:
        return tensor.cpu().numpy()


def visualize_all(input_img, outputs, attn_type, save_dir):
    """可视化所有中间结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    patch_feat = outputs["patch_feat"]
    block_input = outputs["block_input"]
    q_before = outputs["q_before_rearrange"]
    k_before = outputs["k_before_rearrange"]
    v_before = outputs["v_before_rearrange"]
    q_after = outputs["q_after_rearrange"]
    k_after = outputs["k_after_rearrange"]
    v_after = outputs["v_after_rearrange"]
    attn_map = outputs["attn_map"]
    attn_v_before_rearrange = outputs["attn_v_before_rearrange"]
    attn_v_after_rearrange = outputs["attn_v_after_rearrange"]
    attn_output = outputs["attn_output"]
    
    # 创建大图: 2行4列
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Input Image, Patch Embedding, Block Input, Q
    ax1 = plt.subplot(2, 4, 1)
    img_np = np.clip(input_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
    ax1.imshow(img_np)
    ax1.set_title(f"Input Image\n{list(input_img.shape)}", fontsize=10)
    ax1.axis("off")
    
    ax2 = plt.subplot(2, 4, 2)
    patch_l2 = compute_l2_norm(patch_feat)
    im2 = ax2.imshow(patch_l2, cmap="viridis")
    ax2.set_title(f"Patch Embedding (L2 Norm)\n{list(patch_feat.shape)}", fontsize=10)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(2, 4, 3)
    block_l2 = compute_l2_norm(block_input)
    im3 = ax3.imshow(block_l2, cmap="viridis")
    ax3.set_title(f"Block 0 Input (L2 Norm)\n{list(block_input.shape)}", fontsize=10)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = plt.subplot(2, 4, 4)
    q_np = q_after[0, 0].cpu().numpy()  # rearrange 后的 Q
    im4 = ax4.imshow(q_np, cmap="plasma", aspect="auto")
    ax4.set_title(f"Query (after rearrange)\n{list(q_after.shape)}", fontsize=10)
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Row 2: K, V, Attention Map, Attention*V
    ax5 = plt.subplot(2, 4, 5)
    k_np = k_after[0, 0].cpu().numpy()  # rearrange 后的 K
    im5 = ax5.imshow(k_np, cmap="plasma", aspect="auto")
    ax5.set_title(f"Key (after rearrange)\n{list(k_after.shape)}", fontsize=10)
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = plt.subplot(2, 4, 6)
    v_np = v_after[0, 0].cpu().numpy()  # rearrange 后的 V
    im6 = ax6.imshow(v_np, cmap="plasma", aspect="auto")
    ax6.set_title(f"Value (after rearrange)\n{list(v_after.shape)}", fontsize=10)
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    ax7 = plt.subplot(2, 4, 7)
    attn_np = attn_map[0, 0].cpu().numpy()
    im7 = ax7.imshow(attn_np, cmap="inferno", aspect="auto")
    ax7.set_title(f"Attention Map\n{list(attn_map.shape)}", fontsize=10)
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = plt.subplot(2, 4, 8)
    # Attention*V (AFTER rearrange BEFORE project_out) - 已经是 [B, C, H, W] 格式
    attn_v_l2 = compute_l2_norm(attn_v_after_rearrange)
    im8 = ax8.imshow(attn_v_l2, cmap="viridis")
    ax8.set_title(f"Attention*V (L2 Norm)\nAFTER rearrange BEFORE project_out\n{list(attn_v_after_rearrange.shape)}", fontsize=10)
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{attn_type} - Encoder Level 1 Block 0 Visualization", fontsize=14, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{attn_type}_complete_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_first_block", "visualization_first_block_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(img_np)
    ax1.axis("off")
    
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(patch_l2, cmap="viridis")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.imshow(block_l2, cmap="viridis")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(q_np, cmap="plasma", aspect="auto")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(k_np, cmap="plasma", aspect="auto")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = plt.subplot(2, 4, 6)
    im6 = ax6.imshow(v_np, cmap="plasma", aspect="auto")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    ax7 = plt.subplot(2, 4, 7)
    im7 = ax7.imshow(attn_np, cmap="inferno", aspect="auto")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = plt.subplot(2, 4, 8)
    im8 = ax8.imshow(attn_v_l2, cmap="viridis")
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_complete_visualization.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")
    
    # 额外保存 Attention*V BEFORE rearrange BEFORE project_out
    fig, ax = plt.subplots(figsize=(10, 8))
    attn_v_before = attn_v_before_rearrange[0, 0].cpu().numpy()
    im = ax.imshow(attn_v_before, cmap="plasma", aspect="auto")
    ax.set_title(f"{attn_type} - Attention*V\nBEFORE rearrange BEFORE project_out\n{list(attn_v_before_rearrange.shape)}", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{attn_type}_attn_v_before_rearrange.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_v_before, cmap="plasma", aspect="auto")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_attn_v_before_rearrange.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")
    
    # 额外保存 Attention*V AFTER rearrange BEFORE project_out
    fig, ax = plt.subplots(figsize=(8, 6))
    attn_v_after_l2 = compute_l2_norm(attn_v_after_rearrange)
    im = ax.imshow(attn_v_after_l2, cmap="viridis")
    ax.set_title(f"{attn_type} - Attention*V (L2 Norm)\nAFTER rearrange BEFORE project_out\n{list(attn_v_after_rearrange.shape)}", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{attn_type}_attn_v_after_rearrange.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_v_after_l2, cmap="viridis")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_attn_v_after_rearrange.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")
    
    # 额外保存 Attention Output (AFTER project_out)
    fig, ax = plt.subplots(figsize=(8, 6))
    attn_out_l2 = compute_l2_norm(attn_output)
    im = ax.imshow(attn_out_l2, cmap="viridis")
    ax.set_title(f"{attn_type} - Attention Output (L2 Norm)\nAFTER project_out\n{list(attn_output.shape)}", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{attn_type}_attn_output_after_proj.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_out_l2, cmap="viridis")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_attn_output_after_proj.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_individual_maps(outputs, attn_type, save_dir):
    """单独保存每个 map 的高清版本"""
    os.makedirs(save_dir, exist_ok=True)
    save_dir_simple = save_dir.replace("visualization_first_block", "visualization_first_block_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    # 保存 Q, K, V (before rearrange - 保留空间结构)
    maps_before = {
        "q_before_rearrange": outputs["q_before_rearrange"][0].cpu(),  # [C, H, W]
        "k_before_rearrange": outputs["k_before_rearrange"][0].cpu(),
        "v_before_rearrange": outputs["v_before_rearrange"][0].cpu(),
    }
    
    for name, tensor in maps_before.items():
        # 计算 L2 Norm
        data = compute_l2_norm(tensor)
        
        # 完整版本
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(f"{attn_type} - {name.upper()}\n(L2 Norm, Spatial Structure)\nShape: {list(tensor.shape)}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"{attn_type}_{name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        
        # 简化版本
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap="viridis")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_{name}.png")
        plt.savefig(save_path_simple, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved (simple): {save_path_simple}")
    
    # 保存 Q, K, V, Attention Map (after rearrange - 用于计算注意力)
    maps_after = {
        "q_after_rearrange": outputs["q_after_rearrange"][0, 0].cpu().numpy(),
        "k_after_rearrange": outputs["k_after_rearrange"][0, 0].cpu().numpy(),
        "v_after_rearrange": outputs["v_after_rearrange"][0, 0].cpu().numpy(),
        "attn_map": outputs["attn_map"][0, 0].cpu().numpy(),
    }
    
    for name, data in maps_after.items():
        # 完整版本
        fig, ax = plt.subplots(figsize=(10, 8))
        if name == "attn_map":
            im = ax.imshow(data, cmap="inferno", aspect="auto")
        else:
            im = ax.imshow(data, cmap="plasma", aspect="auto")
        ax.set_title(f"{attn_type} - {name.upper()}\nShape: {data.shape}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"{attn_type}_{name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
        
        # 简化版本
        fig, ax = plt.subplots(figsize=(10, 8))
        if name == "attn_map":
            im = ax.imshow(data, cmap="inferno", aspect="auto")
        else:
            im = ax.imshow(data, cmap="plasma", aspect="auto")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path_simple = os.path.join(save_dir_simple, f"{attn_type}_{name}.png")
        plt.savefig(save_path_simple, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved (simple): {save_path_simple}")


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="Visualize First Block of Restormer Encoder Level 1")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--attn_type", type=str, required=True, 
                        choices=["MDTA", "HTA", "WTA"], help="Attention type")
    parser.add_argument("--no_resize", action="store_true", help="Don't resize image (use original size)")
    parser.add_argument("--output_dir", type=str, default="visualization_first_block", 
                        help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Attention: {args.attn_type}")
    print(f"Checkpoint: {args.checkpoint}")

    # 创建模型
    model = FirstBlockVisualizer(attn_type=args.attn_type).to(device)
    
    if not model.load_from_checkpoint(args.checkpoint, device):
        return
    model.eval()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    img_tensor = load_image(args.image, no_resize=args.no_resize).to(device)
    print(f"Image: {args.image} -> {img_tensor.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"\nOutputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {list(val.shape)}")

    # 保存结果
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(args.output_dir, image_name, args.attn_type)
    
    visualize_all(img_tensor, outputs, args.attn_type, save_dir)
    save_individual_maps(outputs, args.attn_type, save_dir)
    
    print(f"\nDone! Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
