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
from einops import rearrange
from typing import List, Optional

from .restormer_arch import (
    Restormer,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
    TransformerBlock,
    LayerNorm,
)


# DINO model dimension and patch size mapping
# DINOv2 uses 14x14 patches, DINOv3 uses 16x16 patches
DINO_DIM_MAP = {
    # DINOv2 models (14x14 patches)
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
    # DINOv2 with registers (improved attention maps)
    'dinov2_vits14_reg': 384,
    'dinov2_vitb14_reg': 768,
    'dinov2_vitl14_reg': 1024,
    'dinov2_vitg14_reg': 1536,
    # DINOv3 models (16x16 patches) - HuggingFace transformers
    'dinov3_vits16': 384,
    'dinov3_vitsplus16': 384,   # ViT-S+/16 (distilled from ViT-7B)
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vithplus16': 1536,  # ViT-H+/16 (distilled from ViT-7B)
}

# HuggingFace model name mapping for DINOv3
DINO_HF_MODEL_MAP = {
    'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'dinov3_vitsplus16': 'facebook/dinov3-vitsplus16-pretrain-lvd1689m',
    'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'dinov3_vithplus16': 'facebook/dinov3-vithplus16-pretrain-lvd1689m',
}

DINO_PATCH_SIZE_MAP = {
    # DINOv2 models use 14x14 patches
    'dinov2_vits14': 14,
    'dinov2_vitb14': 14,
    'dinov2_vitl14': 14,
    'dinov2_vitg14': 14,
    'dinov2_vits14_reg': 14,
    'dinov2_vitb14_reg': 14,
    'dinov2_vitl14_reg': 14,
    'dinov2_vitg14_reg': 14,
    # DINOv3 models use 16x16 patches
    'dinov3_vits16': 16,
    'dinov3_vitsplus16': 16,
    'dinov3_vitb16': 16,
    'dinov3_vitl16': 16,
    'dinov3_vithplus16': 16,
}


class DINOFeatureExtractor(nn.Module):
    """
    Extract semantic features from frozen DINOv3 model.
    
    This module handles:
    1. Loading and freezing DINO model
    2. Preprocessing low-light images (gamma correction + ImageNet normalization)
    3. Extracting and reshaping DINO features to spatial feature maps
    
    Args:
        model_name: DINO model variant ('dinov3_vits16', 'dinov3_vitb16', etc.)
        gamma: Gamma correction value for low-light preprocessing (default: 0.4)
        local_model_path: Optional path to local DINO model weights
    """
    
    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        gamma: float = 0.4,
        local_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.gamma = gamma
        self.dino_dim = DINO_DIM_MAP.get(model_name, 768)
        self.patch_size = DINO_PATCH_SIZE_MAP.get(model_name, 14)  # DINOv2 uses 14x14 patches
        
        # Load DINO model
        self.dino = self._load_dino_model(model_name, local_model_path)
        
        # Freeze all DINO parameters
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
          dinov3_vitl16, dinov3_vith16plus)
        - Local model paths
        - Legacy DINOv2 models via torch.hub
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
            elif model_name.startswith('dinov2_'):
                # Load DINOv2 from torch.hub
                dino = torch.hub.load('facebookresearch/dinov2', model_name)
            else:
                # Fallback to torch.hub
                dino = torch.hub.load('facebookresearch/dinov3', model_name)
            return dino
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO model '{model_name}': {e}")
    
    def _freeze_dino(self):
        """Freeze all DINO parameters."""
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
    
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
            DINO feature map [B, dino_dim, H/14, W/14]
        """
        B, C, H, W = x.shape
        
        # Check input size compatibility
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input size ({H}, {W}) must be divisible by patch size {self.patch_size}"
            )
        
        # Preprocess for DINO
        x_preprocessed = self._preprocess(x)
        
        # Extract features (no gradient computation)
        with torch.no_grad():
            # Check if this is a HuggingFace transformers model
            if hasattr(self.dino, 'config'):
                # HuggingFace transformers model (DINOv3)
                outputs = self.dino(x_preprocessed, output_hidden_states=True)
                last_hidden = outputs.last_hidden_state
                # DINOv3 output: [B, 1 + num_registers + num_patches, dim]
                # Typically: 1 CLS token + 4 register tokens + patch tokens
                num_registers = getattr(self.dino.config, 'num_register_tokens', 4)
                patch_tokens = last_hidden[:, 1 + num_registers:]  # Remove CLS and register tokens
            elif hasattr(self.dino, 'forward_features'):
                # Standard DINOv2/v3 from torch.hub
                dino_out = self.dino.forward_features(x_preprocessed)
                # dino_out shape: [B, 1 + num_patches, dim] or dict
                if isinstance(dino_out, dict):
                    patch_tokens = dino_out.get('x_norm_patchtokens', dino_out.get('x_patchtokens'))
                else:
                    patch_tokens = dino_out[:, 1:]  # Remove CLS token
            else:
                # Fallback for other model types
                outputs = self.dino(x_preprocessed, output_hidden_states=True)
                if hasattr(outputs, 'last_hidden_state'):
                    patch_tokens = outputs.last_hidden_state[:, 1:]  # Remove CLS token
                else:
                    patch_tokens = outputs[0][:, 1:]
        
        # Reshape from [B, num_patches, dim] to [B, dim, h, w]
        h, w = H // self.patch_size, W // self.patch_size
        features = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=h, w=w)
        
        return features


class DINOGuidedAttention(nn.Module):
    """
    DINO-guided attention module.
    
    Uses DINO features to generate spatial and channel attention weights
    that modulate Restormer features. This is the "guided condition" approach
    where DINO provides "where to enhance" guidance rather than direct features.
    
    Args:
        cnn_dim: Dimension of CNN/Restormer features
        dino_dim: Dimension of DINO features (default: 768 for ViT-B)
    """
    
    def __init__(self, cnn_dim: int, dino_dim: int = 768):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.dino_dim = dino_dim
        
        # Spatial attention: which regions need enhancement
        # Output: [B, 1, H, W] attention map
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dino_dim, dino_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dino_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Channel attention: which feature channels to enhance
        # Output: [B, C, 1, 1] channel weights
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dino_dim, cnn_dim, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        dino_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply DINO-guided attention to CNN features.
        
        Args:
            cnn_feat: Restormer features [B, C, H, W]
            dino_feat: DINO features [B, dino_dim, h, w]
            
        Returns:
            Guided features [B, C, H, W] with residual connection
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
        
        # Generate attention weights
        spatial_weight = self.spatial_attn(dino_up)    # [B, 1, H, W]
        channel_weight = self.channel_attn(dino_feat)  # [B, C, 1, 1]
        
        # Apply guided attention (modulation, not replacement)
        guided_feat = cnn_feat * spatial_weight * channel_weight
        
        # Residual connection to preserve original features
        return cnn_feat + guided_feat


class DINOGuidedRestormer(nn.Module):
    """
    DINOv3-guided Restormer for low-light image enhancement.
    
    This architecture injects DINO semantic guidance at the Latent layer
    while preserving the original Restormer encoder-decoder structure.
    
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
    ):
        super().__init__()
        
        self.use_dino_guidance = use_dino_guidance
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
            
            # DINO feature projection (dino_dim -> latent_dim)
            dino_dim = DINO_DIM_MAP.get(dino_model, 768)
            self.dino_proj = nn.Sequential(
                nn.Conv2d(dino_dim, self.latent_dim, kernel_size=1),
                nn.GELU(),
            )
            
            # DINO guided attention
            self.dino_guide = DINOGuidedAttention(
                cnn_dim=self.latent_dim,
                dino_dim=dino_dim,
            )
    
    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional DINO guidance.
        
        Args:
            inp_img: Input image [B, 3, H, W]
            
        Returns:
            Enhanced image [B, 3, H, W]
        """
        # ===== Extract DINO features (if enabled) =====
        dino_feat = None
        if self.use_dino_guidance:
            dino_feat = self.dino_extractor(inp_img)
        
        # ===== Restormer Encoder =====
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        # ===== Apply DINO Guidance at Latent Layer =====
        if self.use_dino_guidance and dino_feat is not None:
            latent = self.dino_guide(latent, dino_feat)
        
        # ===== Restormer Decoder =====
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        
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
