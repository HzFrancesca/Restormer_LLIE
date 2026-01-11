"""
DINOv3 Guided Restormer Architecture

This module implements the DINOv3-guided Restormer for low-light image enhancement.
The core idea is to use frozen DINOv3 features as semantic guidance to enhance
Restormer's pixel-level restoration capability.

Design Principles:
1. DINO as guidance, not backbone - preserve Restormer's spatial detail processing
2. Guided attention - DINO generates attention weights to modulate features
3. Frozen DINO - reduce trainable parameters for small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from typing import List, Optional

from .restormer_arch import (
    Restormer,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
    TransformerBlock,
    LayerNorm,
)


# DINOv3 model dimension mapping (16x16 patches, HuggingFace transformers)
DINO_DIM_MAP = {
    'dinov3_vits16': 384,
    'dinov3_vitsplus16': 384,   # ViT-S+/16 (distilled from ViT-7B)
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vithplus16': 1280,  # ViT-H+/16 hidden_size=1280 (from HuggingFace config)
}

# HuggingFace model name mapping for DINOv3
DINO_HF_MODEL_MAP = {
    'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'dinov3_vitsplus16': 'facebook/dinov3-vitsplus16-pretrain-lvd1689m',
    'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'dinov3_vithplus16': 'facebook/dinov3-vithplus16-pretrain-lvd1689m',
}

# DINOv3 patch size mapping (all use 16x16 patches)
DINO_PATCH_SIZE_MAP = {
    'dinov3_vits16': 16,
    'dinov3_vitsplus16': 16,
    'dinov3_vitb16': 16,
    'dinov3_vitl16': 16,
    'dinov3_vithplus16': 16,
}


# =============================================================================
# LoRA Configuration
# =============================================================================

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) fine-tuning.
    
    LoRA enables efficient fine-tuning by adding low-rank decomposition matrices
    to the attention layers while keeping the original weights frozen.
    
    Args:
        r: Low-rank dimension (default: 16)
        lora_alpha: Scaling factor for LoRA weights (default: 32)
        lora_dropout: Dropout probability for LoRA layers (default: 0.1)
        target_modules: List of module names to apply LoRA (default: ["qkv"])
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])


class DINOFeatureExtractor(nn.Module):
    """
    Extract semantic features from frozen DINOv3 model.
    
    This module handles:
    1. Loading and freezing DINO model
    2. Preprocessing low-light images (gamma correction + ImageNet normalization)
    3. Extracting and reshaping DINO features to spatial feature maps
    4. Optional LoRA fine-tuning support
    
    Args:
        model_name: DINO model variant ('dinov3_vits16', 'dinov3_vitb16', etc.)
        gamma: Gamma correction value for low-light preprocessing (default: 0.4)
        local_model_path: Optional path to local DINO model weights
        extract_layers: List of layer indices to extract features from (for multi-scale)
                       If None, only extracts from the last layer
        use_lora: Whether to apply LoRA adapters for fine-tuning (default: False)
        lora_config: LoRA configuration (optional, uses defaults if None)
    """
    
    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        gamma: float = 0.4,
        local_model_path: Optional[str] = None,
        extract_layers: Optional[List[int]] = None,
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.gamma = gamma
        self.dino_dim = DINO_DIM_MAP.get(model_name, 768)
        self.patch_size = DINO_PATCH_SIZE_MAP.get(model_name, 16)  # DINOv3 uses 16x16 patches
        self.extract_layers = extract_layers  # None means only last layer
        self.use_lora = use_lora
        
        # Load DINO model
        self.dino = self._load_dino_model(model_name, local_model_path)
        
        # Apply LoRA if enabled
        if use_lora:
            self._apply_lora(lora_config or LoRAConfig())
        else:
            # Freeze all DINO parameters if not using LoRA
            self._freeze_dino()
        
        # ImageNet normalization constants
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _load_dino_model(
        self, 
        model_name: str, 
        local_path: Optional[str] = None
    ) -> nn.Module:
        """
        Load DINOv3 model from HuggingFace or local path.
        
        Supports:
        - DINOv3 models via HuggingFace transformers (dinov3_vits16, dinov3_vitb16, 
          dinov3_vitl16, dinov3_vithplus16)
        - Local model paths
        """
        try:
            if local_path is not None:
                # Load from local path using transformers
                from transformers import AutoModel
                import os
                is_local = os.path.isdir(local_path)
                dino = AutoModel.from_pretrained(
                    local_path, 
                    trust_remote_code=True,
                    local_files_only=is_local
                )
            elif model_name in DINO_HF_MODEL_MAP:
                # Load DINOv3 from HuggingFace
                from transformers import AutoModel
                hf_model_name = DINO_HF_MODEL_MAP[model_name]
                dino = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
            else:
                raise ValueError(f"Unknown DINOv3 model: {model_name}. "
                               f"Available models: {list(DINO_HF_MODEL_MAP.keys())}")
            return dino
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO model '{model_name}': {e}")
    
    def _freeze_dino(self):
        """Freeze all DINO parameters."""
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
    
    def _apply_lora(self, lora_config: LoRAConfig) -> None:
        """
        Apply LoRA adapters to DINO model for efficient fine-tuning.
        
        This method uses the PEFT library to add low-rank adaptation layers
        to the specified target modules (typically QKV projections).
        
        Args:
            lora_config: LoRA configuration with r, alpha, dropout, and target_modules
            
        Raises:
            ImportError: If PEFT library is not installed
        """
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA support. "
                "Install it with: pip install peft>=0.6.0"
            )
        
        # Create PEFT LoRA config
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none",
        )
        
        # Apply LoRA to DINO model
        self.dino = get_peft_model(self.dino, peft_config)
        
        # Store config for later reference
        self.lora_config = lora_config
    
    def save_lora_weights(self, path: str) -> None:
        """
        Save only LoRA weights to a file.
        
        Args:
            path: Path to save the LoRA weights
            
        Raises:
            RuntimeError: If LoRA is not enabled
        """
        if not self.use_lora:
            raise RuntimeError("LoRA is not enabled. Cannot save LoRA weights.")
        
        # Get LoRA state dict (only trainable parameters)
        lora_state_dict = {}
        for name, param in self.dino.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.data.clone()
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'lora_config': self.lora_config,
            'model_name': self.model_name,
        }, path)
    
    def load_lora_weights(self, path: str) -> None:
        """
        Load LoRA weights from a file.
        
        Args:
            path: Path to the saved LoRA weights
            
        Raises:
            RuntimeError: If LoRA is not enabled or model mismatch
        """
        if not self.use_lora:
            raise RuntimeError("LoRA is not enabled. Cannot load LoRA weights.")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Verify model compatibility
        if checkpoint.get('model_name') != self.model_name:
            raise RuntimeError(
                f"Model mismatch: checkpoint is for {checkpoint.get('model_name')}, "
                f"but current model is {self.model_name}"
            )
        
        # Load LoRA weights
        lora_state_dict = checkpoint['lora_state_dict']
        current_state = self.dino.state_dict()
        
        for name, param in lora_state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)
        
        self.dino.load_state_dict(current_state)
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess low-light images for DINO.
        
        Applies:
        1. Clamping minimum to 1e-8 to avoid numerical issues
        2. Gamma correction to boost dark regions
        3. ImageNet normalization
        
        Args:
            x: Input image [B, 3, H, W] with values in [0, 1]
            
        Returns:
            Preprocessed image ready for DINO
        """
        # Clamp minimum to avoid numerical issues with gamma correction
        x = x.clamp(min=1e-8)
        
        # Apply gamma correction to boost dark regions
        x = torch.pow(x, self.gamma)
        
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO features from input image.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            If extract_layers is None: DINO feature map [B, dino_dim, H/16, W/16]
            If extract_layers is set: List of feature maps from specified layers
        """
        B, C, H, W = x.shape
        
        # Pad input if not divisible by patch size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W
        
        # Preprocess for DINO
        x_preprocessed = self._preprocess(x)
        
        # Calculate spatial dimensions
        h, w = H_padded // self.patch_size, W_padded // self.patch_size
        feat_h = H // self.patch_size if (pad_h > 0 or pad_w > 0) else h
        feat_w = W // self.patch_size if (pad_h > 0 or pad_w > 0) else w
        
        # Extract features (no gradient computation)
        with torch.no_grad():
            # HuggingFace transformers model (DINOv3)
            outputs = self.dino(x_preprocessed, output_hidden_states=True)
            num_registers = getattr(self.dino.config, 'num_register_tokens', 4)
            
            if self.extract_layers is not None:
                # Multi-scale extraction from specified layers
                hidden_states = outputs.hidden_states  # Tuple of [B, seq_len, dim]
                features_list = []
                
                for layer_idx in self.extract_layers:
                    # Layer index is 1-based in hidden_states (0 is embedding)
                    if layer_idx < len(hidden_states):
                        layer_hidden = hidden_states[layer_idx]
                        # Remove CLS and register tokens
                        patch_tokens = layer_hidden[:, 1 + num_registers:]
                        # Reshape to spatial
                        feat = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=h, w=w)
                        # Remove padding if needed
                        if pad_h > 0 or pad_w > 0:
                            feat = feat[:, :, :feat_h, :feat_w]
                        features_list.append(feat)
                
                return features_list
            else:
                # Single layer extraction (last layer)
                last_hidden = outputs.last_hidden_state
                patch_tokens = last_hidden[:, 1 + num_registers:]
                features = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=h, w=w)
                
                # Remove padding from feature map if needed
                if pad_h > 0 or pad_w > 0:
                    features = features[:, :, :feat_h, :feat_w]
                
                return features


# =============================================================================
# Grid Artifacts Handling Modules (块效应处理模块)
# =============================================================================

class SmoothUpsampler(nn.Module):
    """
    PixelShuffle-based upsampling with boundary smoothing.
    
    This module eliminates grid artifacts by using PixelShuffle followed by
    a 3×3 convolution to blend patch boundaries.
    
    Architecture:
        Conv2d(in_dim, out_dim * scale^2, 3x3) → PixelShuffle → Conv2d(3x3) → GELU
    
    Args:
        in_dim: Input channel dimension
        out_dim: Output channel dimension
        scale_factor: Upsampling factor (default: 2)
    """
    
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.upsample = nn.Sequential(
            # Expand channels for PixelShuffle
            nn.Conv2d(in_dim, out_dim * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            # 3×3 conv to smooth patch boundaries
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input tensor with boundary smoothing.
        
        Args:
            x: Input tensor [B, in_dim, H, W]
            
        Returns:
            Upsampled tensor [B, out_dim, H*scale, W*scale]
        """
        return self.upsample(x)


class PatchBoundaryBlender(nn.Module):
    """
    Blend features across ViT patch boundaries using depthwise convolution.
    
    This module uses a 5×5 depthwise convolution to mix features across
    patch boundaries, with a learnable blending weight.
    
    Architecture:
        DWConv(5x5) → Conv(1x1) → GELU
        Output = x + edge_weight * (blended - x)
    
    Args:
        dim: Feature dimension
        patch_size: ViT patch size (default: 16, for reference only)
    """
    
    def __init__(self, dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        
        # 5×5 depthwise conv to blend across patch boundaries
        self.blend = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),  # Depthwise
            nn.Conv2d(dim, dim, kernel_size=1),  # Pointwise
            nn.GELU(),
        )
        
        # Learnable edge weight initialized to 0.5
        self.edge_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual-style boundary blending.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Blended tensor [B, C, H, W]
            Formula: output = x + edge_weight * (blend(x) - x)
        """
        blended = self.blend(x)
        return x + self.edge_weight * (blended - x)


class OverlapPatchUpsample(nn.Module):
    """
    Bilinear upsampling with overlapping convolution for smooth boundaries.
    
    This module uses bilinear interpolation followed by a 5×5 convolution
    to cover multiple patches and eliminate boundary artifacts.
    
    Architecture:
        Bilinear Upsample → Conv(5x5) → Conv(3x3) → GELU → Conv(3x3)
    
    Args:
        in_dim: Input channel dimension
        out_dim: Output channel dimension
        scale_factor: Upsampling factor (default: 2)
    """
    
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 5×5 conv to cover multiple patches
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=2)
        
        # Refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample with overlapping convolution.
        
        Args:
            x: Input tensor [B, in_dim, H, W]
            
        Returns:
            Upsampled tensor [B, out_dim, H*scale, W*scale]
        """
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        x = self.conv(x)
        return self.refine(x)


# =============================================================================
# Feature Fusion Modules
# =============================================================================

class SFTFusion(nn.Module):
    """
    Spatial Feature Transform (SFT) Fusion Module.
    
    Implements affine modulation similar to StyleGAN/Stable Diffusion:
        F_fused = γ(F_dino) ⊙ F_res + β(F_dino)
    
    This allows DINO semantic features to modulate Restormer features
    through learned scale (γ) and shift (β) parameters.
    
    Args:
        cnn_dim: Dimension of CNN/Restormer features
        dino_dim: Dimension of DINO features (default: 768 for ViT-B)
    """
    
    def __init__(self, cnn_dim: int, dino_dim: int = 768):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.dino_dim = dino_dim
        
        # Shared feature compression
        self.dino_compress = nn.Sequential(
            nn.Conv2d(dino_dim, cnn_dim, kernel_size=1),
            nn.GELU(),
        )
        
        # Scale (γ) generator
        self.gamma_conv = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
        )
        
        # Shift (β) generator
        self.beta_conv = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
        )
    
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        dino_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply SFT modulation to CNN features.
        
        Args:
            cnn_feat: Restormer features [B, C, H, W]
            dino_feat: DINO features [B, dino_dim, h, w]
            
        Returns:
            Modulated features [B, C, H, W]
        """
        # Upsample DINO features to match CNN feature size
        if dino_feat.shape[-2:] != cnn_feat.shape[-2:]:
            dino_up = F.interpolate(
                dino_feat, 
                size=cnn_feat.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            dino_up = dino_feat
        
        # Compress DINO channels to match CNN channels
        dino_compressed = self.dino_compress(dino_up)  # [B, C, H, W]
        
        # Generate scale and shift
        gamma = self.gamma_conv(dino_compressed)  # [B, C, H, W]
        beta = self.beta_conv(dino_compressed)    # [B, C, H, W]
        
        # Apply SFT: F_fused = γ ⊙ F_res + β
        # Add 1 to gamma for residual-style modulation (identity when gamma=0)
        fused = (1 + gamma) * cnn_feat + beta
        
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Transposed Cross-Attention Fusion Module (Channel-wise).
    
    Similar to Restormer's MDTA, but cross-modal:
    - Q from Restormer features
    - K, V from DINO features
    - Attention computed in channel dimension: [C × C] instead of [HW × HW]
    
    This design:
    1. Matches Restormer's transposed attention philosophy
    2. Reduces complexity from O((HW)²) to O(C²)
    3. Enables channel-wise semantic guidance from DINO
    
    Computation:
        Q: [B, heads, C/heads, HW] from Restormer
        K: [B, heads, C/heads, HW] from DINO
        V: [B, heads, C/heads, HW] from DINO
        Attn = softmax(Q @ K^T / τ)  -> [B, heads, C/heads, C/heads]
        Out = Attn @ V -> [B, heads, C/heads, HW]
    
    Args:
        cnn_dim: Dimension of CNN/Restormer features
        dino_dim: Dimension of DINO features (default: 768 for ViT-B)
        num_heads: Number of attention heads (default: 8)
    """
    
    def __init__(self, cnn_dim: int, dino_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.dino_dim = dino_dim
        self.num_heads = num_heads
        self.head_dim = cnn_dim // num_heads
        
        # Learnable temperature (like MDTA)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Query projection with DWConv (from Restormer features)
        self.q_conv = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=1)
        self.q_dwconv = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1, groups=cnn_dim)
        
        # Key and Value projections with DWConv (from DINO features)
        self.kv_conv = nn.Conv2d(dino_dim, cnn_dim * 2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(cnn_dim * 2, cnn_dim * 2, kernel_size=3, padding=1, groups=cnn_dim * 2)
        
        # Output projection
        self.out_proj = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=1)
    
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        dino_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply transposed cross-attention fusion (channel-wise).
        
        Args:
            cnn_feat: Restormer features [B, C, H, W]
            dino_feat: DINO features [B, dino_dim, h, w]
            
        Returns:
            Fused features [B, C, H, W] with residual connection
        """
        B, C, H, W = cnn_feat.shape
        
        # Upsample DINO features to match CNN spatial size
        if dino_feat.shape[-2:] != (H, W):
            dino_up = F.interpolate(
                dino_feat, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            dino_up = dino_feat
        
        # Project Q from Restormer features
        q = self.q_dwconv(self.q_conv(cnn_feat))  # [B, C, H, W]
        
        # Project K, V from DINO features
        kv = self.kv_dwconv(self.kv_conv(dino_up))  # [B, 2C, H, W]
        k, v = kv.chunk(2, dim=1)  # [B, C, H, W] each
        
        # Reshape for transposed multi-head attention
        # [B, C, H, W] -> [B, heads, C/heads, HW]
        q = rearrange(q, 'b (nh hd) h w -> b nh hd (h w)', nh=self.num_heads, hd=self.head_dim)
        k = rearrange(k, 'b (nh hd) h w -> b nh hd (h w)', nh=self.num_heads, hd=self.head_dim)
        v = rearrange(v, 'b (nh hd) h w -> b nh hd (h w)', nh=self.num_heads, hd=self.head_dim)
        
        # Normalize Q and K (like MDTA)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Transposed attention: [B, heads, C/heads, C/heads]
        # Q @ K^T computes channel-to-channel attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values: [B, heads, C/heads, HW]
        out = attn @ v
        
        # Reshape back: [B, heads, C/heads, HW] -> [B, C, H, W]
        out = rearrange(out, 'b nh hd (h w) -> b (nh hd) h w', h=H, w=W)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection
        return cnn_feat + out


class DINOGuidedAttention(nn.Module):
    """
    DINO-guided attention module with multiple fusion strategies.
    
    Supports two fusion modes:
    1. SFT (Spatial Feature Transform): Efficient affine modulation
       F_fused = γ(F_dino) ⊙ F_res + β(F_dino)
    
    2. Cross-Attention (Transposed/Channel-wise): Like MDTA but cross-modal
       Q from Restormer, K/V from DINO
       Attn = softmax(Q @ K^T / τ) in channel dimension [C × C]
       Out = Attn @ V + residual
    
    Args:
        cnn_dim: Dimension of CNN/Restormer features
        dino_dim: Dimension of DINO features (default: 768 for ViT-B)
        fusion_type: 'sft' or 'cross_attention' (default: 'sft')
        num_heads: Number of attention heads for cross-attention (default: 8)
    """
    
    def __init__(
        self, 
        cnn_dim: int, 
        dino_dim: int = 768,
        fusion_type: str = 'sft',
        num_heads: int = 8,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.dino_dim = dino_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'sft':
            self.fusion = SFTFusion(cnn_dim, dino_dim)
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(cnn_dim, dino_dim, num_heads)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'sft' or 'cross_attention'.")
    
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        dino_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply DINO-guided fusion to CNN features.
        
        Args:
            cnn_feat: Restormer features [B, C, H, W]
            dino_feat: DINO features [B, dino_dim, h, w]
            
        Returns:
            Fused features [B, C, H, W]
        """
        return self.fusion(cnn_feat, dino_feat)


# =============================================================================
# Multi-Scale FPN Components (方案二)
# =============================================================================

class MultiScaleDINOFusion(nn.Module):
    """
    Multi-scale DINO feature fusion module for FPN-style integration.
    
    This module handles fusion at a single encoder level, with features
    from a specific DINO layer. Supports both SFT and Cross-Attention.
    
    Args:
        cnn_dim: Dimension of CNN/Restormer features at this level
        dino_dim: Dimension of DINO features (same for all layers)
        fusion_type: 'sft' or 'cross_attention'
        num_heads: Number of attention heads for cross-attention
    """
    
    def __init__(
        self,
        cnn_dim: int,
        dino_dim: int = 768,
        fusion_type: str = 'sft',
        num_heads: int = 4,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.dino_dim = dino_dim
        
        # Use the same fusion strategies as single-scale
        if fusion_type == 'sft':
            self.fusion = SFTFusion(cnn_dim, dino_dim)
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(cnn_dim, dino_dim, num_heads)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(
        self,
        cnn_feat: torch.Tensor,
        dino_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse DINO features into CNN features at this level.
        
        Args:
            cnn_feat: Restormer features [B, C, H, W]
            dino_feat: DINO features [B, dino_dim, h, w] (will be upsampled)
            
        Returns:
            Fused features [B, C, H, W]
        """
        return self.fusion(cnn_feat, dino_feat)


class MultiScaleDINOGuidedRestormer(nn.Module):
    """
    Multi-Scale FPN DINOv3-guided Restormer (方案二).
    
    This architecture extracts features from multiple DINO layers and
    injects them at corresponding Restormer encoder levels:
    
    - DINO Layer 4 (shallow) → Encoder Level 2 (texture/denoising)
    - DINO Layer 8 (middle)  → Encoder Level 3 (structure)
    - DINO Layer 12 (deep)   → Latent Layer (semantics/color)
    
    Design Rationale:
    - Shallow DINO layers preserve low-level texture details (good for denoising)
    - Deep DINO layers capture high-level semantics (good for color restoration)
    - FPN-style multi-scale fusion combines both benefits
    
    Architecture:
    ```
    Input ──┬──→ Restormer Encoder L1 ──→ L2 ──→ L3 ──→ Latent ──→ Decoder ──→ Output
            │                             ↑      ↑       ↑
            └──→ DINOv3 ──→ Layer4 ───────┘      │       │
                         ──→ Layer8 ─────────────┘       │
                         ──→ Layer12 ────────────────────┘
    ```
    
    Args:
        inp_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 3)
        dim: Base feature dimension (default: 48)
        num_blocks: Number of transformer blocks at each level
        num_refinement_blocks: Number of refinement blocks
        heads: Number of attention heads at each level
        ffn_expansion_factor: FFN expansion factor
        bias: Whether to use bias in convolutions
        LayerNorm_type: Type of layer normalization
        attn_types: Attention types for each level
        dino_model: DINO model variant
        dino_gamma: Gamma correction for DINO preprocessing
        dino_local_path: Optional local path to DINO model
        use_dino_guidance: Whether to enable DINO guidance
        fusion_type: Fusion strategy - 'sft' or 'cross_attention'
        fusion_num_heads: Number of attention heads for cross-attention
        dino_extract_layers: DINO layers to extract [shallow, middle, deep]
        inject_levels: Which encoder levels to inject DINO features
                      'all' = [level2, level3, latent]
                      'latent_only' = [latent] (same as DINOGuidedRestormer)
                      'encoder_only' = [level2, level3]
        use_checkpoint: Whether to use gradient checkpointing (default: False)
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
        attn_types: List[str] = ["MDTA", "MDTA", "MDTA", "MDTA"],
        dino_model: str = 'dinov3_vitb16',
        dino_gamma: float = 0.4,
        dino_local_path: Optional[str] = None,
        use_dino_guidance: bool = True,
        fusion_type: str = 'sft',
        fusion_num_heads: int = 4,
        dino_extract_layers: List[int] = [4, 8, 12],
        inject_levels: str = 'all',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.use_dino_guidance = use_dino_guidance
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.inject_levels = inject_levels
        self.dino_extract_layers = dino_extract_layers
        
        # Feature dimensions at each level
        self.level2_dim = int(dim * 2**1)  # 96 for dim=48
        self.level3_dim = int(dim * 2**2)  # 192 for dim=48
        self.latent_dim = int(dim * 2**3)  # 384 for dim=48
        
        # ===== Restormer Components =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_blocks[0])
            ]
        )
        
        # Encoder Level 2
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level2_dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                )
                for _ in range(num_blocks[1])
            ]
        )
        
        # Encoder Level 3
        self.down2_3 = Downsample(self.level2_dim)
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level3_dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                )
                for _ in range(num_blocks[2])
            ]
        )
        
        # Latent Layer (Level 4)
        self.down3_4 = Downsample(self.level3_dim)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.latent_dim,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[3],
                )
                for _ in range(num_blocks[3])
            ]
        )
        
        # Decoder Level 3
        self.up4_3 = Upsample(self.latent_dim)
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), self.level3_dim, kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level3_dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                )
                for _ in range(num_blocks[2])
            ]
        )
        
        # Decoder Level 2
        self.up3_2 = Upsample(self.level3_dim)
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), self.level2_dim, kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level2_dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                )
                for _ in range(num_blocks[1])
            ]
        )
        
        # Decoder Level 1
        self.up2_1 = Upsample(self.level2_dim)
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level2_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_blocks[0])
            ]
        )
        
        # Refinement
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.level2_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        
        # Output
        self.output = nn.Conv2d(
            self.level2_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        
        # ===== Multi-Scale DINO Guidance Components =====
        if use_dino_guidance:
            dino_dim = DINO_DIM_MAP.get(dino_model, 768)
            
            # DINO feature extractor with multi-scale extraction
            self.dino_extractor = DINOFeatureExtractor(
                model_name=dino_model,
                gamma=dino_gamma,
                local_model_path=dino_local_path,
                extract_layers=dino_extract_layers,
            )
            
            # Multi-scale fusion modules
            # Determine which levels to inject based on config
            self.inject_level2 = inject_levels in ['all', 'encoder_only']
            self.inject_level3 = inject_levels in ['all', 'encoder_only']
            self.inject_latent = inject_levels in ['all', 'latent_only']
            
            if self.inject_level2:
                # Shallow DINO (Layer 4) → Encoder Level 2
                self.dino_fusion_level2 = MultiScaleDINOFusion(
                    cnn_dim=self.level2_dim,
                    dino_dim=dino_dim,
                    fusion_type=fusion_type,
                    num_heads=max(1, fusion_num_heads // 2),
                )
            
            if self.inject_level3:
                # Middle DINO (Layer 8) → Encoder Level 3
                self.dino_fusion_level3 = MultiScaleDINOFusion(
                    cnn_dim=self.level3_dim,
                    dino_dim=dino_dim,
                    fusion_type=fusion_type,
                    num_heads=fusion_num_heads,
                )
            
            if self.inject_latent:
                # Deep DINO (Layer 12) → Latent Layer
                self.dino_fusion_latent = MultiScaleDINOFusion(
                    cnn_dim=self.latent_dim,
                    dino_dim=dino_dim,
                    fusion_type=fusion_type,
                    num_heads=fusion_num_heads,
                )
    
    def _forward_with_checkpoint(
        self, 
        x: torch.Tensor, 
        module: nn.Module
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
    
    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale DINO guidance and optional checkpointing.
        
        Args:
            inp_img: Input image [B, 3, H, W]
            
        Returns:
            Enhanced image [B, 3, H, W]
        """
        # ===== Extract Multi-Scale DINO Features =====
        dino_feats = None
        if self.use_dino_guidance:
            dino_feats = self.dino_extractor(inp_img)
            # dino_feats is a list: [layer4_feat, layer8_feat, layer12_feat]
        
        # ===== Restormer Encoder (with optional checkpointing) =====
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self._forward_with_checkpoint(inp_enc_level1, self.encoder_level1)
        
        # Level 2 with optional DINO fusion (shallow features for texture)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._forward_with_checkpoint(inp_enc_level2, self.encoder_level2)
        if self.use_dino_guidance and self.inject_level2 and dino_feats is not None:
            out_enc_level2 = self.dino_fusion_level2(out_enc_level2, dino_feats[0])
        
        # Level 3 with optional DINO fusion (middle features for structure)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._forward_with_checkpoint(inp_enc_level3, self.encoder_level3)
        if self.use_dino_guidance and self.inject_level3 and dino_feats is not None:
            out_enc_level3 = self.dino_fusion_level3(out_enc_level3, dino_feats[1])
        
        # Latent with optional DINO fusion (deep features for semantics)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._forward_with_checkpoint(inp_enc_level4, self.latent)
        if self.use_dino_guidance and self.inject_latent and dino_feats is not None:
            latent = self.dino_fusion_latent(latent, dino_feats[2])
        
        # ===== Restormer Decoder (with optional checkpointing) =====
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self._forward_with_checkpoint(inp_dec_level3, self.decoder_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self._forward_with_checkpoint(inp_dec_level2, self.decoder_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self._forward_with_checkpoint(inp_dec_level1, self.decoder_level1)
        
        out_dec_level1 = self._forward_with_checkpoint(out_dec_level1, self.refinement)
        
        # Output with residual connection
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1
    
    def load_pretrained_restormer(
        self,
        checkpoint_path: str,
        strict: bool = False
    ) -> None:
        """
        Load pretrained Restormer weights, handling missing DINO-related keys.
        
        Args:
            checkpoint_path: Path to Restormer checkpoint
            strict: Whether to strictly enforce key matching
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle nested state dict
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out DINO-related keys
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('dino_'):
                continue
            filtered_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys (expected for DINO components): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")


class DINOGuidedRestormer(nn.Module):
    """
    DINOv3-guided Restormer for low-light image enhancement.
    
    This architecture injects DINO semantic guidance at the Latent layer
    while preserving the original Restormer encoder-decoder structure.
    
    Design follows the three-step approach:
    1. Feature Extraction: Frozen DINOv3 extracts semantic features
    2. Alignment: Bilinear upsampling + 1x1 conv for resolution/channel matching
    3. Fusion: SFT (affine modulation) or Cross-Attention
    
    Args:
        inp_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 3)
        dim: Base feature dimension (default: 48)
        num_blocks: Number of transformer blocks at each level
        num_refinement_blocks: Number of refinement blocks
        heads: Number of attention heads at each level
        ffn_expansion_factor: FFN expansion factor
        bias: Whether to use bias in convolutions
        LayerNorm_type: Type of layer normalization
        attn_types: Attention types for each level
        dino_model: DINO model variant
        dino_gamma: Gamma correction for DINO preprocessing
        dino_local_path: Optional local path to DINO model
        use_dino_guidance: Whether to enable DINO guidance
        fusion_type: Fusion strategy - 'sft' (efficient) or 'cross_attention' (memory-intensive)
        fusion_num_heads: Number of attention heads for cross-attention fusion
        use_checkpoint: Whether to use gradient checkpointing (default: False)
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
        attn_types: List[str] = ["MDTA", "MDTA", "MDTA", "MDTA"],
        dino_model: str = 'dinov3_vitb16',
        dino_gamma: float = 0.4,
        dino_local_path: Optional[str] = None,
        use_dino_guidance: bool = True,
        fusion_type: str = 'sft',
        fusion_num_heads: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.use_dino_guidance = use_dino_guidance
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.latent_dim = int(dim * 2**3)  # 384 for dim=48
        
        # ===== Restormer Components (reuse existing architecture) =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_blocks[0])
            ]
        )
        
        # Encoder Level 2
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                )
                for _ in range(num_blocks[1])
            ]
        )
        
        # Encoder Level 3
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                )
                for _ in range(num_blocks[2])
            ]
        )
        
        # Latent Layer (Level 4)
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=self.latent_dim,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[3],
                )
                for _ in range(num_blocks[3])
            ]
        )
        
        # Decoder Level 3
        self.up4_3 = Upsample(self.latent_dim)
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                )
                for _ in range(num_blocks[2])
            ]
        )
        
        # Decoder Level 2
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                )
                for _ in range(num_blocks[1])
            ]
        )
        
        # Decoder Level 1
        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_blocks[0])
            ]
        )
        
        # Refinement
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        
        # Output
        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        
        # ===== DINO Guidance Components =====
        if use_dino_guidance:
            # DINO feature extractor
            self.dino_extractor = DINOFeatureExtractor(
                model_name=dino_model,
                gamma=dino_gamma,
                local_model_path=dino_local_path,
            )
            
            # DINO guided fusion (handles alignment internally)
            dino_dim = DINO_DIM_MAP.get(dino_model, 768)
            self.dino_guide = DINOGuidedAttention(
                cnn_dim=self.latent_dim,
                dino_dim=dino_dim,
                fusion_type=fusion_type,
                num_heads=fusion_num_heads,
            )
    
    def _forward_with_checkpoint(
        self, 
        x: torch.Tensor, 
        module: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input tensor
            module: Module to apply
            
        Returns:
            Output tensor
        """
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
    
    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional DINO guidance and gradient checkpointing.
        
        Args:
            inp_img: Input image [B, 3, H, W]
            
        Returns:
            Enhanced image [B, 3, H, W]
        """
        # ===== Extract DINO features (if enabled) =====
        dino_feat = None
        if self.use_dino_guidance:
            dino_feat = self.dino_extractor(inp_img)
        
        # ===== Restormer Encoder (with optional checkpointing) =====
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self._forward_with_checkpoint(inp_enc_level1, self.encoder_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self._forward_with_checkpoint(inp_enc_level2, self.encoder_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self._forward_with_checkpoint(inp_enc_level3, self.encoder_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self._forward_with_checkpoint(inp_enc_level4, self.latent)
        
        # ===== Apply DINO Guidance at Latent Layer =====
        if self.use_dino_guidance and dino_feat is not None:
            latent = self.dino_guide(latent, dino_feat)
        
        # ===== Restormer Decoder (with optional checkpointing) =====
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self._forward_with_checkpoint(inp_dec_level3, self.decoder_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self._forward_with_checkpoint(inp_dec_level2, self.decoder_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self._forward_with_checkpoint(inp_dec_level1, self.decoder_level1)
        
        out_dec_level1 = self._forward_with_checkpoint(out_dec_level1, self.refinement)
        
        # Output with residual connection
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1
    
    def load_pretrained_restormer(
        self, 
        checkpoint_path: str, 
        strict: bool = False
    ) -> None:
        """
        Load pretrained Restormer weights, handling missing DINO-related keys.
        
        Args:
            checkpoint_path: Path to Restormer checkpoint
            strict: Whether to strictly enforce key matching
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle nested state dict (e.g., from training checkpoint)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out DINO-related keys if loading from original Restormer
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # Skip DINO-related keys
            if k.startswith('dino_'):
                continue
            filtered_state_dict[k] = v
        
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys (expected for DINO components): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")


# =============================================================================
# 方案三: DINO Encoder + Lightweight FPN Decoder
# =============================================================================

class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for DINO features.
    
    Takes multi-scale DINO features and progressively upsamples to full resolution.
    Uses SmoothUpsampler and PatchBoundaryBlender to eliminate grid artifacts.
    
    Architecture:
        Level 3 (1/16) → SmoothUp → Blend → Level 2 (1/8)
        Level 2 (1/8)  → SmoothUp → Blend → Level 1 (1/4)
        Level 1 (1/4)  → SmoothUp → Blend → Level 0 (1/2)
        Level 0 (1/2)  → SmoothUp → Blend → Output (1/1)
    
    Args:
        dino_dim: DINO feature dimension (default: 768 for ViT-B)
        hidden_dims: List of hidden dimensions for each level [level3, level2, level1, level0]
        out_channels: Output image channels (default: 3)
        use_boundary_blend: Whether to use PatchBoundaryBlender (default: True)
    """
    
    def __init__(
        self,
        dino_dim: int = 768,
        hidden_dims: List[int] = [384, 192, 96, 48],
        out_channels: int = 3,
        use_boundary_blend: bool = True,
    ):
        super().__init__()
        self.dino_dim = dino_dim
        self.hidden_dims = hidden_dims
        self.use_boundary_blend = use_boundary_blend
        
        # Feature projection layers for multi-scale DINO features
        # Assumes 3 DINO layers: [shallow, middle, deep]
        self.proj_deep = nn.Conv2d(dino_dim, hidden_dims[0], kernel_size=1)
        self.proj_middle = nn.Conv2d(dino_dim, hidden_dims[1], kernel_size=1)
        self.proj_shallow = nn.Conv2d(dino_dim, hidden_dims[2], kernel_size=1)
        
        # Decoder levels with SmoothUpsampler
        # Level 3 → Level 2 (1/16 → 1/8)
        self.up3_2 = SmoothUpsampler(hidden_dims[0], hidden_dims[1], scale_factor=2)
        self.blend3_2 = PatchBoundaryBlender(hidden_dims[1]) if use_boundary_blend else nn.Identity()
        self.fuse3_2 = nn.Conv2d(hidden_dims[1] * 2, hidden_dims[1], kernel_size=1)
        
        # Level 2 → Level 1 (1/8 → 1/4)
        self.up2_1 = SmoothUpsampler(hidden_dims[1], hidden_dims[2], scale_factor=2)
        self.blend2_1 = PatchBoundaryBlender(hidden_dims[2]) if use_boundary_blend else nn.Identity()
        self.fuse2_1 = nn.Conv2d(hidden_dims[2] * 2, hidden_dims[2], kernel_size=1)
        
        # Level 1 → Level 0 (1/4 → 1/2)
        self.up1_0 = SmoothUpsampler(hidden_dims[2], hidden_dims[3], scale_factor=2)
        self.blend1_0 = PatchBoundaryBlender(hidden_dims[3]) if use_boundary_blend else nn.Identity()
        
        # Level 0 → Output (1/2 → 1/1)
        self.up0_out = SmoothUpsampler(hidden_dims[3], hidden_dims[3], scale_factor=2)
        self.blend_out = PatchBoundaryBlender(hidden_dims[3]) if use_boundary_blend else nn.Identity()
        
        # Final output projection
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dims[3], hidden_dims[3], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dims[3], out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, dino_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale DINO features to full resolution.
        
        Args:
            dino_feats: List of DINO features [shallow, middle, deep]
                       Each has shape [B, dino_dim, H/16, W/16]
            
        Returns:
            Decoded features [B, out_channels, H, W]
        """
        # Unpack features (shallow=layer4, middle=layer8, deep=layer12)
        feat_shallow, feat_middle, feat_deep = dino_feats
        
        # Project to hidden dimensions
        feat_deep = self.proj_deep(feat_deep)      # [B, 384, H/16, W/16]
        feat_middle = self.proj_middle(feat_middle)  # [B, 192, H/16, W/16]
        feat_shallow = self.proj_shallow(feat_shallow)  # [B, 96, H/16, W/16]
        
        # Level 3 → Level 2
        up_deep = self.up3_2(feat_deep)  # [B, 192, H/8, W/8]
        up_deep = self.blend3_2(up_deep)
        # Upsample middle features to match
        feat_middle_up = F.interpolate(feat_middle, size=up_deep.shape[-2:], mode='bilinear', align_corners=False)
        feat_l2 = self.fuse3_2(torch.cat([up_deep, feat_middle_up], dim=1))
        
        # Level 2 → Level 1
        up_l2 = self.up2_1(feat_l2)  # [B, 96, H/4, W/4]
        up_l2 = self.blend2_1(up_l2)
        # Upsample shallow features to match
        feat_shallow_up = F.interpolate(feat_shallow, size=up_l2.shape[-2:], mode='bilinear', align_corners=False)
        feat_l1 = self.fuse2_1(torch.cat([up_l2, feat_shallow_up], dim=1))
        
        # Level 1 → Level 0
        feat_l0 = self.up1_0(feat_l1)  # [B, 48, H/2, W/2]
        feat_l0 = self.blend1_0(feat_l0)
        
        # Level 0 → Output
        feat_out = self.up0_out(feat_l0)  # [B, 48, H, W]
        feat_out = self.blend_out(feat_out)
        
        return self.output(feat_out)


class DINOEncoderDecoder(nn.Module):
    """
    方案三: DINO as primary encoder with lightweight FPN decoder.
    
    This architecture uses DINOv3 as the sole encoder backbone,
    extracting multi-scale features for FPN-style decoding.
    
    Architecture:
        Input Image ──→ DINOv3 (LoRA) ──┬──→ Layer 4  ──→ Proj ──→ ┐
                                        ├──→ Layer 8  ──→ Proj ──→ ├──→ FPN Decoder ──→ Output
                                        └──→ Layer 12 ──→ Proj ──→ ┘
    
    Args:
        inp_channels: Input image channels (default: 3)
        out_channels: Output image channels (default: 3)
        dino_model: DINO model variant
        dino_gamma: Gamma correction for preprocessing
        dino_local_path: Local path to DINO weights
        use_lora: Whether to use LoRA fine-tuning
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha (default: 32)
        lora_dropout: LoRA dropout (default: 0.1)
        extract_layers: DINO layers to extract features from
        hidden_dims: Decoder hidden dimensions
        use_boundary_blend: Whether to use boundary blending
    """
    
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dino_model: str = 'dinov3_vitb16',
        dino_gamma: float = 0.4,
        dino_local_path: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        extract_layers: List[int] = [4, 8, 12],
        hidden_dims: List[int] = [384, 192, 96, 48],
        use_boundary_blend: bool = True,
    ):
        super().__init__()
        
        self.use_lora = use_lora
        self.extract_layers = extract_layers
        
        # Get DINO dimension
        dino_dim = DINO_DIM_MAP.get(dino_model, 768)
        
        # Create LoRA config if enabled
        lora_config = None
        if use_lora:
            lora_config = LoRAConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        
        # DINO feature extractor (encoder)
        self.dino_extractor = DINOFeatureExtractor(
            model_name=dino_model,
            gamma=dino_gamma,
            local_model_path=dino_local_path,
            extract_layers=extract_layers,
            use_lora=use_lora,
            lora_config=lora_config,
        )
        
        # FPN Decoder
        self.decoder = FPNDecoder(
            dino_dim=dino_dim,
            hidden_dims=hidden_dims,
            out_channels=out_channels,
            use_boundary_blend=use_boundary_blend,
        )
    
    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: DINO encoding + FPN decoding.
        
        Args:
            inp_img: Input image [B, 3, H, W]
            
        Returns:
            Enhanced image [B, 3, H, W] with residual connection
        """
        # Extract multi-scale DINO features
        dino_feats = self.dino_extractor(inp_img)
        
        # Decode to full resolution
        decoded = self.decoder(dino_feats)
        
        # Residual connection to input
        return decoded + inp_img
    
    def save_lora_weights(self, path: str) -> None:
        """Save LoRA weights if enabled."""
        self.dino_extractor.save_lora_weights(path)
    
    def load_lora_weights(self, path: str) -> None:
        """Load LoRA weights if enabled."""
        self.dino_extractor.load_lora_weights(path)
