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
        self.patch_size = DINO_PATCH_SIZE_MAP.get(model_name, 16)  # DINOv3 uses 16x16 patches
        
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
            DINO feature map [B, dino_dim, H/16, W/16]
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
        
        # Extract features (no gradient computation)
        with torch.no_grad():
            # HuggingFace transformers model (DINOv3)
            outputs = self.dino(x_preprocessed, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            # DINOv3 output: [B, 1 + num_registers + num_patches, dim]
            # Typically: 1 CLS token + 4 register tokens + patch tokens
            num_registers = getattr(self.dino.config, 'num_register_tokens', 4)
            patch_tokens = last_hidden[:, 1 + num_registers:]  # Remove CLS and register tokens
        
        # Reshape from [B, num_patches, dim] to [B, dim, h, w]
        h, w = H_padded // self.patch_size, W_padded // self.patch_size
        features = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=h, w=w)
        
        # Remove padding from feature map if needed
        if pad_h > 0 or pad_w > 0:
            # Feature map is at 1/16 resolution
            feat_h = H // self.patch_size
            feat_w = W // self.patch_size
            features = features[:, :, :feat_h, :feat_w]
        
        return features


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
            
            # DINO guided fusion (handles alignment internally)
            dino_dim = DINO_DIM_MAP.get(dino_model, 768)
            self.dino_guide = DINOGuidedAttention(
                cnn_dim=self.latent_dim,
                dino_dim=dino_dim,
                fusion_type=fusion_type,
                num_heads=fusion_num_heads,
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
