"""
Restormer 注意力机制可视化脚本
支持 Encoder Level 1-4 (Latent)，串行执行到指定 level 和 block
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


# ============== 配置 ==============

LEVEL_CONFIG = {
    1: {"dim": 48,  "heads": 1, "num_blocks": 4, "prefix": "encoder_level1"},
    2: {"dim": 96,  "heads": 2, "num_blocks": 6, "prefix": "encoder_level2"},
    3: {"dim": 192, "heads": 4, "num_blocks": 6, "prefix": "encoder_level3"},
    4: {"dim": 384, "heads": 8, "num_blocks": 8, "prefix": "latent"},  # Latent/Bottleneck
}

ATTENTION_TITLES = {
    "MDTA": "MDTA (Channel: CxC)",
    "HTA": "HTA (Column: WxW)",
    "WTA": "WTA (Row: HxH)",
}


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


# ============== Feed Forward & Patch Embed & Downsample ==============

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ============== 带 Attention Map 输出的注意力模块 ==============

class MDTAWithMap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_map = None
        self.value_map = None
        self.attn_v_map = None  # 新增：保存 Attention*V

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)  # [b, head, c_per_head, h*w]
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)  # [b, head, c_per_head, h*w]
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)  # [b, head, c_per_head, h*w]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, head, c_per_head, c_per_head]
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach()
        self.value_map = v.detach()  # 保存 V
        out = attn @ v  # [b, head, c_per_head, h*w]
        self.attn_v_map = out.detach()  # 新增：保存 Attention*V (before project_out)
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class HTAWithMap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_map = None
        self.value_map = None
        self.attn_v_map = None  # 新增：保存 Attention*V

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head w (c h)", head=self.num_heads)  # [b, head, w, D]
        k = rearrange(k, "b (head c) h w -> b head w (c h)", head=self.num_heads)  # [b, head, w, D]
        v = rearrange(v, "b (head c) h w -> b head w (c h)", head=self.num_heads)  # [b, head, w, D]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, head, w, w]
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach()
        self.value_map = v.detach()  # 保存 V
        out = attn @ v  # [b, head, w, D]
        self.attn_v_map = out.detach()  # 新增：保存 Attention*V (before project_out)
        out = rearrange(out, "b head w (c h) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class WTAWithMap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_map = None
        self.value_map = None
        self.attn_v_map = None  # 新增：保存 Attention*V

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach()
        self.value_map = v.detach()  # 保存 V
        out = attn @ v
        self.attn_v_map = out.detach()  # 新增：保存 Attention*V (before project_out)
        out = rearrange(out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


ATTENTION_CLASSES = {
    "MDTA": MDTAWithMap, "HTA": HTAWithMap, "WTA": WTAWithMap,
}


# ============== TransformerBlock ==============

class TransformerBlockWithMap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type="WithBias", attn_type="MDTA"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ATTENTION_CLASSES[attn_type](dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.attn_output = None  # 纯 Attention 输出

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        self.attn_output = attn_out.detach()  # 保存纯 Attention 输出
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def get_attn_map(self):
        return self.attn.attn_map
    
    def get_attn_output(self):
        return self.attn_output
    
    def get_value_map(self):
        return self.attn.value_map
    
    def get_attn_v_map(self):
        """获取 Attention*V (before project_out)"""
        return self.attn.attn_v_map


# ============== 完整 Encoder 可视化模型 ==============

class RestormerEncoderVisualizer(nn.Module):
    """
    Restormer Encoder 可视化模型
    支持 Level 1-4，串行执行到指定 level 和 block
    """
    def __init__(self, attn_type="MDTA", bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.attn_type = attn_type
        
        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=48, bias=bias)
        
        # Encoder Level 1: dim=48, heads=1, 4 blocks
        self.encoder_level1 = nn.ModuleList([
            TransformerBlockWithMap(48, 1, 2.66, bias, LayerNorm_type, attn_type)
            for _ in range(4)
        ])
        
        # Downsample 1->2
        self.down1_2 = Downsample(48)
        
        # Encoder Level 2: dim=96, heads=2, 6 blocks
        self.encoder_level2 = nn.ModuleList([
            TransformerBlockWithMap(96, 2, 2.66, bias, LayerNorm_type, attn_type)
            for _ in range(6)
        ])
        
        # Downsample 2->3
        self.down2_3 = Downsample(96)
        
        # Encoder Level 3: dim=192, heads=4, 6 blocks
        self.encoder_level3 = nn.ModuleList([
            TransformerBlockWithMap(192, 4, 2.66, bias, LayerNorm_type, attn_type)
            for _ in range(6)
        ])
        
        # Downsample 3->4
        self.down3_4 = Downsample(192)
        
        # Latent (Level 4): dim=384, heads=8, 8 blocks
        self.latent = nn.ModuleList([
            TransformerBlockWithMap(384, 8, 2.66, bias, LayerNorm_type, attn_type)
            for _ in range(8)
        ])
        
        self.levels = {
            1: self.encoder_level1,
            2: self.encoder_level2,
            3: self.encoder_level3,
            4: self.latent,
        }
        self.downsamples = {
            1: self.down1_2,
            2: self.down2_3,
            3: self.down3_4,
        }

    def forward(self, x, target_level=1, target_block=0):
        """
        串行执行到 target_level 的 target_block
        返回: patch_feat, level_input, block_input, block_output, attn_map, attn_output, value_map, attn_v_map
        """
        # Patch Embedding
        x = self.patch_embed(x)
        patch_feat = x.clone()
        
        level_input = None
        block_input = None
        block_output = None
        attn_map = None
        attn_output = None
        value_map = None
        attn_v_map = None  # 新增
        
        for level in range(1, target_level + 1):
            # 记录当前 level 的输入
            if level == target_level:
                level_input = x.clone()
            
            blocks = self.levels[level]
            num_blocks = len(blocks)
            
            # 确定要执行多少个 block
            if level == target_level:
                blocks_to_run = target_block + 1
            else:
                blocks_to_run = num_blocks
            
            for i in range(blocks_to_run):
                if level == target_level and i == target_block:
                    block_input = x.clone()
                
                x = blocks[i](x)
                
                if level == target_level and i == target_block:
                    block_output = x.clone()
                    attn_map = blocks[i].get_attn_map()
                    attn_output = blocks[i].get_attn_output()
                    value_map = blocks[i].get_value_map()
                    attn_v_map = blocks[i].get_attn_v_map()  # 新增
            
            # 如果不是目标 level，执行下采样
            if level < target_level and level in self.downsamples:
                x = self.downsamples[level](x)
        
        return patch_feat, level_input, block_input, block_output, attn_map, attn_output, value_map, attn_v_map

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
        self._load_module(state_dict, "patch_embed.", self.patch_embed, "patch_embed")
        
        # 加载 downsamples
        self._load_module(state_dict, "down1_2.", self.down1_2, "down1_2")
        self._load_module(state_dict, "down2_3.", self.down2_3, "down2_3")
        self._load_module(state_dict, "down3_4.", self.down3_4, "down3_4")
        
        # 加载各 level 的 blocks
        level_prefixes = {
            1: "encoder_level1",
            2: "encoder_level2",
            3: "encoder_level3",
            4: "latent",
        }
        
        for level, prefix in level_prefixes.items():
            blocks = self.levels[level]
            for i, block in enumerate(blocks):
                block_prefix = f"{prefix}.{i}."
                block_dict = {k.replace(block_prefix, ""): v
                              for k, v in state_dict.items() if k.startswith(block_prefix)}
                if block_dict:
                    block.load_state_dict(block_dict, strict=False)
            print(f"Loaded: {prefix} ({len(blocks)} blocks)")
        
        return True
    
    def _load_module(self, state_dict, prefix, module, name):
        module_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        if module_dict:
            module.load_state_dict(module_dict, strict=False)
            print(f"Loaded: {name}")


# ============== 可视化函数 ==============

def load_image(image_path, size=256, no_resize=True):
    img = Image.open(image_path).convert("RGB")
    if no_resize:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
    return transform(img).unsqueeze(0)


def save_attention_map(attn_type, attn_map, save_dir, level, block):
    os.makedirs(save_dir, exist_ok=True)
    
    if attn_map.dim() == 4:
        attn_np = attn_map[0, 0].cpu().numpy()
    else:
        attn_np = attn_map[0, 0, 0].cpu().numpy()

    # 完整版本（带坐标轴）
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_np, cmap="inferno", aspect="auto")
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    ax.set_title(f"{ATTENTION_TITLES[attn_type]}\n{level_name} Block {block} | Shape: {attn_np.shape}", fontsize=11)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "attention_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_np, cmap="inferno", aspect="auto")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "attention_map.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_feature_map(attn_type, input_img, patch_feat, level_input, block_input, block_output, 
                     save_dir, level, block):
    os.makedirs(save_dir, exist_ok=True)
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input -> Patch Embed -> Level Input
    img_np = np.clip(input_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Image", fontsize=10)
    axes[0, 0].axis("off")
    
    im = axes[0, 1].imshow(patch_feat[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[0, 1].set_title(f"Patch Embed\n{list(patch_feat.shape)}", fontsize=10)
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im = axes[0, 2].imshow(level_input[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[0, 2].set_title(f"{level_name} Input\n{list(level_input.shape)}", fontsize=10)
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Block Input -> Block Output -> Diff
    im = axes[1, 0].imshow(block_input[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[1, 0].set_title(f"Block {block} Input", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im = axes[1, 1].imshow(block_output[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[1, 1].set_title(f"Block {block} Output", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    diff = (block_output[0] - block_input[0]).abs().cpu().mean(dim=0).numpy()
    im = axes[1, 2].imshow(diff, cmap="magma")
    axes[1, 2].set_title(f"Block {block} |Output - Input|", fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "feature_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_value_map(attn_type, value_map, save_dir, level, block):
    """保存 Value (V) 的可视化 - 使用 L2 Norm"""
    os.makedirs(save_dir, exist_ok=True)
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    # value_map shape: [b, head, c_per_head, spatial_dim] or [b, head, spatial_dim, feature_dim]
    # 取第一个 batch 和第一个 head
    if value_map.dim() == 4:
        v_np = value_map[0, 0].cpu().numpy()  # [c_per_head, spatial_dim] or [spatial_dim, feature_dim]
    else:
        v_np = value_map[0, 0, 0].cpu().numpy()
    
    # 完整版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(v_np, cmap="plasma", aspect="auto")
    
    if attn_type == "MDTA":
        ax.set_title(f"Value Matrix (V)\nShape: {v_np.shape} [channels × spatial]", fontsize=11)
        ax.set_xlabel("Spatial Dimension (H×W)")
        ax.set_ylabel("Channel Dimension")
    elif attn_type == "HTA":
        ax.set_title(f"Value Matrix (V)\nShape: {v_np.shape} [width × features]", fontsize=11)
        ax.set_xlabel("Feature Dimension (C×H)")
        ax.set_ylabel("Width (W)")
    else:  # WTA
        ax.set_title(f"Value Matrix (V)\nShape: {v_np.shape} [height × features]", fontsize=11)
        ax.set_xlabel("Feature Dimension (C×W)")
        ax.set_ylabel("Height (H)")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "value_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(v_np, cmap="plasma", aspect="auto")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "value_map.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_attn_v_map(attn_type, attn_v_map, block_input, save_dir, level, block):
    """保存 Attention Output (BEFORE project_out) 的可视化 - 使用 L2 Norm"""
    os.makedirs(save_dir, exist_ok=True)
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    # 获取 num_heads
    num_heads = 1 if level == 1 else (2 if level == 2 else (4 if level == 3 else 8))
    
    # 转回 [B, C, H, W] 格式计算 L2 Norm - 修复 HTA 和 WTA 的重建方式
    if attn_type == "MDTA":
        # MDTA: [b, head, c, (h w)] -> [b, (head c), h, w]
        attn_v_4d = rearrange(attn_v_map, "b head c (h w) -> b (head c) h w", 
                              head=num_heads,
                              h=block_input.shape[2], w=block_input.shape[3])
    elif attn_type == "HTA":
        # HTA: [b, head, w, (c h)] -> 先恢复 [b, head, w, c, h] -> [b, (head c), h, w]
        b, head, w, ch = attn_v_map.shape
        h = block_input.shape[2]
        c_per_head = ch // h
        attn_v_reshaped = attn_v_map.reshape(b, head, w, c_per_head, h)  # [b, head, w, c, h]
        attn_v_4d = rearrange(attn_v_reshaped, "b head w c h -> b (head c) h w")
    else:  # WTA
        # WTA: [b, head, h, (c w)] -> 先恢复 [b, head, h, c, w] -> [b, (head c), h, w]
        b, head, h, cw = attn_v_map.shape
        w = block_input.shape[3]
        c_per_head = cw // w
        attn_v_reshaped = attn_v_map.reshape(b, head, h, c_per_head, w)  # [b, head, h, c, w]
        attn_v_4d = rearrange(attn_v_reshaped, "b head h c w -> b (head c) h w")
    
    # 计算 L2 Norm
    feat = attn_v_4d[0].cpu()  # [C, H, W]
    l2_norm = torch.sqrt((feat ** 2).sum(dim=0)).numpy()
    
    # 完整版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.set_title(f"Attention Output (L2 Norm)\nBEFORE project_out (Fixed Reshape)", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "attn_output_before_proj.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "attn_output_before_proj.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_attention_output(attn_output, save_dir, level, block):
    """保存 Attention Output (AFTER project_out，不含残差和FFN) - 使用 L2 Norm"""
    os.makedirs(save_dir, exist_ok=True)
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    feat = attn_output[0].cpu()  # [C, H, W]
    
    # 计算 L2 Norm: sqrt(sum(x^2)) across channel dimension
    l2_norm = torch.sqrt((feat ** 2).sum(dim=0)).numpy()
    
    # 完整版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.set_title("L2 Norm across channels", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{level_name} Block {block} Attention Output\n(AFTER project_out, no residual/FFN)", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "attn_output_after_proj.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "attn_output_after_proj.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_block_output_feature(block_output, save_dir, level, block):
    """单独保存 block 输出的特征图 - 使用 L2 Norm"""
    os.makedirs(save_dir, exist_ok=True)
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    feat = block_output[0].cpu()  # [C, H, W]
    
    # 计算 L2 Norm: sqrt(sum(x^2)) across channel dimension
    l2_norm = torch.sqrt((feat ** 2).sum(dim=0)).numpy()
    
    # 完整版本
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.set_title("L2 Norm across channels", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{level_name} Block {block} Output Feature (FFN Output)\nShape: {list(block_output.shape)}", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "block_output_feature.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 同时保存原始 tensor 以便后续分析
    tensor_path = os.path.join(save_dir, "block_output_feature.pt")
    torch.save(block_output.cpu(), tensor_path)
    print(f"Saved: {tensor_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(l2_norm, cmap="viridis")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "block_output_feature.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


def save_channel_attention_maps(attn_type, attn_map, save_dir, level, block, num_channels=8):
    if attn_map.dim() != 5:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    c = min(num_channels, attn_map.shape[2])
    level_name = "Latent" if level == 4 else f"Encoder L{level}"
    
    # 完整版本
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(c):
        attn_np = attn_map[0, 0, i].cpu().numpy()
        axes[i].imshow(attn_np, cmap="inferno", aspect="auto")
        axes[i].set_title(f"Channel {i}", fontsize=10)
        axes[i].axis("off")
    
    for i in range(c, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f"{attn_type} {level_name} Block {block} - Per Channel", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "channel_attention_maps.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    # 简化版本（去掉所有标题和标签，用于论文）
    save_dir_simple = save_dir.replace("visualization_encoder", "visualization_encoder_simple")
    os.makedirs(save_dir_simple, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(c):
        attn_np = attn_map[0, 0, i].cpu().numpy()
        axes[i].imshow(attn_np, cmap="inferno", aspect="auto")
        axes[i].axis("off")
    
    for i in range(c, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    save_path_simple = os.path.join(save_dir_simple, "channel_attention_maps.png")
    plt.savefig(save_path_simple, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (simple): {save_path_simple}")


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="Visualize Restormer Attention (Multi-Level)")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--attn_type", type=str, required=True,
                        choices=["MDTA", "HTA", "WTA"], help="Attention type")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Encoder level (1-3) or Latent (4)")
    parser.add_argument("--block", type=int, default=-1, help="Block index (-1 for last block)")
    parser.add_argument("--size", type=int, default=128, help="Image size (only used when --resize is set)")
    parser.add_argument("--resize", action="store_true", help="Resize image to --size (default: no resize)")
    parser.add_argument("--output_dir", type=str, default="visualization", help="Output directory")
    args = parser.parse_args()

    # 处理 block 索引（-1 表示最后一个）
    max_blocks = LEVEL_CONFIG[args.level]["num_blocks"]
    if args.block == -1:
        args.block = max_blocks - 1
    elif args.block >= max_blocks:
        print(f"Error: Level {args.level} only has {max_blocks} blocks (0-{max_blocks-1})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_name = "latent" if args.level == 4 else f"encoder{args.level}"
    
    print(f"Device: {device}")
    print(f"Attention: {args.attn_type}")
    print(f"Level: {args.level} ({level_name}), Block: {args.block}")
    print(f"Checkpoint: {args.checkpoint}")

    # 创建模型
    model = RestormerEncoderVisualizer(attn_type=args.attn_type).to(device)
    
    if not model.load_from_checkpoint(args.checkpoint, device):
        return
    model.eval()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    img_tensor = load_image(args.image, args.size, no_resize=not args.resize).to(device)
    print(f"Image: {args.image} -> {img_tensor.shape}")
    if args.resize:
        print(f"Resized to: {args.size}x{args.size}")
    else:
        print(f"No resize (original size)")

    # 前向传播
    with torch.no_grad():
        patch_feat, level_input, block_input, block_output, attn_map, attn_output, value_map, attn_v_map = model(
            img_tensor, target_level=args.level, target_block=args.block
        )
    
    print(f"\nPatch feat: {patch_feat.shape}")
    print(f"Level {args.level} input: {level_input.shape}")
    print(f"Block {args.block} input: {block_input.shape}")
    print(f"Block {args.block} output: {block_output.shape}")
    print(f"Attention map: {attn_map.shape}")
    print(f"Attention output: {attn_output.shape}")
    print(f"Value map: {value_map.shape}")
    print(f"Attention*V map: {attn_v_map.shape}")

    # 保存结果 - 目录结构: visualization/图片文件名/encoder1/MDTA/block0
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(args.output_dir, image_name, level_name, args.attn_type, f"block{args.block}")
    
    save_attention_map(args.attn_type, attn_map, save_dir, args.level, args.block)
    save_value_map(args.attn_type, value_map, save_dir, args.level, args.block)
    save_attn_v_map(args.attn_type, attn_v_map, block_input, save_dir, args.level, args.block)  # 新增
    save_attention_output(attn_output, save_dir, args.level, args.block)
    save_feature_map(args.attn_type, img_tensor, patch_feat, level_input, block_input, 
                     block_output, save_dir, args.level, args.block)
    save_block_output_feature(block_output, save_dir, args.level, args.block)
    save_channel_attention_maps(args.attn_type, attn_map, save_dir, args.level, args.block)

    print(f"\nDone! Results: {save_dir}/")


if __name__ == "__main__":
    main()
