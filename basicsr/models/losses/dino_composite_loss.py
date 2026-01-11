"""
DINO Composite Loss for DINOv3-guided Restormer

This module implements a composite loss function combining:
1. Charbonnier Loss - pixel-level reconstruction (smooth L1)
2. SSIM Loss - structural similarity preservation
3. Perceptual Loss - VGG-based texture/feature matching
4. DINO Semantic Loss - semantic consistency using frozen DINOv3

Reference: loss.md design document
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1 variant).
    
    Better than standard L1 for edge preservation and more stable
    convergence than L2. Used in original Restormer paper.
    
    Args:
        eps: Small constant for numerical stability
        loss_weight: Weight multiplier for this loss component
    """
    
    def __init__(self, eps: float = 1e-3, loss_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return self.loss_weight * torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    
    Computes 1 - SSIM to use as a loss function. SSIM measures
    luminance, contrast, and structure similarity.
    
    Args:
        window_size: Size of the Gaussian window
        loss_weight: Weight multiplier for this loss component
    """
    
    def __init__(self, window_size: int = 11, loss_weight: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.loss_weight = loss_weight
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        def gaussian(window_size: int, sigma: float) -> torch.Tensor:
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor, 
        window: torch.Tensor,
        size_average: bool = True
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        channel = img1.size(1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self.window.to(pred.device).to(pred.dtype)
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel)
            self.window = self.window.to(pred.device).to(pred.dtype)
            self.channel = channel
        
        ssim_val = self._ssim(pred, target, self.window)
        return self.loss_weight * (1 - ssim_val)


class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss.
    
    Compares features extracted from a pretrained VGG19 network
    to encourage perceptually similar outputs.
    
    Args:
        layer_weights: Weights for different VGG layers
        loss_weight: Weight multiplier for this loss component
        use_input_norm: Whether to normalize inputs to ImageNet stats
    """
    
    def __init__(
        self, 
        layer_weights: Optional[Dict[str, float]] = None,
        loss_weight: float = 0.1,
        use_input_norm: bool = True
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        
        # Default: use conv3_4 (layer 16) and conv4_4 (layer 25)
        if layer_weights is None:
            layer_weights = {'16': 1.0, '25': 1.0}
        self.layer_weights = layer_weights
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        max_layer = max(int(k) for k in layer_weights.keys())
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:max_layer + 1])
        
        # Freeze VGG
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.vgg_layers.eval()
        
        # ImageNet normalization
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        if self.use_input_norm:
            return (x - self.mean) / self.std
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        
        loss = 0.0
        pred_feat = pred_norm
        target_feat = target_norm
        
        for i, layer in enumerate(self.vgg_layers):
            pred_feat = layer(pred_feat)
            with torch.no_grad():
                target_feat = layer(target_feat)
            
            if str(i) in self.layer_weights:
                loss += self.layer_weights[str(i)] * F.l1_loss(pred_feat, target_feat)
        
        return self.loss_weight * loss


class DINOSemanticLoss(nn.Module):
    """
    DINO Semantic Consistency Loss.
    
    Uses frozen DINOv3 features to ensure semantic consistency between
    enhanced output and ground truth.
    
    Implements: L_dino = || φ_dino(I_restored) - φ_dino(I_gt) ||_p
    
    Supports two modes:
    1. Raw DINO model mode: Pass a raw DINO model (legacy compatibility)
    2. Shared extractor mode: Pass a DINOFeatureExtractor instance (recommended)
    
    Supports two distance metrics:
    - 'l1': L1 distance
    - 'l2': L2 distance (MSE)
    - 'cosine': Cosine similarity (robust to lighting changes)
    
    Args:
        dino_model: Pre-loaded DINOv3 model or DINOFeatureExtractor instance
        gamma: Gamma correction for low-light preprocessing (only used with raw model)
        loss_weight: Weight multiplier for this loss component
        distance_type: 'l1', 'l2' or 'cosine' (default: 'cosine')
        extract_layers: Which DINO layers to use for loss (default: None = last layer only)
                       Only effective when using DINOFeatureExtractor
        normalize: Whether to normalize features before comparison (default: True)
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        gamma: float = 0.4,
        loss_weight: float = 0.05,
        distance_type: str = 'cosine',
        extract_layers: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.patch_size = 16  # DINOv3 uses 16x16 patches
        self.distance_type = distance_type
        self.extract_layers = extract_layers
        self.normalize = normalize
        
        # Check if using shared DINOFeatureExtractor or raw model
        self._use_shared_extractor = hasattr(dino_model, 'dino_dim') and hasattr(dino_model, '_preprocess')
        
        if self._use_shared_extractor:
            # Using DINOFeatureExtractor - store reference (don't freeze, it handles its own state)
            self.dino_extractor = dino_model
            self.dino = None
            self.patch_size = dino_model.patch_size
        else:
            # Using raw DINO model - freeze it
            self.dino_extractor = None
            self.dino = dino_model
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
            
            # ImageNet normalization (only needed for raw model mode)
            self.register_buffer(
                'mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
    
    def _preprocess(self, x: torch.Tensor, apply_gamma: bool = False) -> torch.Tensor:
        """Preprocess image for DINO (optional gamma + ImageNet norm). Only for raw model mode."""
        x = x.clamp(min=1e-8)
        if apply_gamma:
            x = torch.pow(x, self.gamma)
        return (x - self.mean) / self.std
    
    def _extract_features_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from raw DINO model."""
        with torch.no_grad() if not x.requires_grad else torch.enable_grad():
            if hasattr(self.dino, 'forward_features'):
                dino_out = self.dino.forward_features(x)
                if isinstance(dino_out, dict):
                    tokens = dino_out.get('x_norm_patchtokens', dino_out.get('x_patchtokens'))
                else:
                    tokens = dino_out[:, 1:]  # Remove CLS token
            else:
                outputs = self.dino(x, output_hidden_states=True)
                if hasattr(outputs, 'last_hidden_state'):
                    # DINOv3 HuggingFace: remove CLS and register tokens
                    num_registers = getattr(self.dino.config, 'num_register_tokens', 4)
                    tokens = outputs.last_hidden_state[:, 1 + num_registers:]
                else:
                    tokens = outputs[0][:, 1:]
        return tokens
    
    def _extract_features_shared(
        self, 
        x: torch.Tensor, 
        need_grad: bool = False
    ) -> List[torch.Tensor]:
        """
        Extract features using shared DINOFeatureExtractor.
        
        Supports multi-layer extraction when extract_layers is specified.
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
        
        # Preprocess using extractor's method
        x_preprocessed = self.dino_extractor._preprocess(x)
        
        h, w = H_padded // self.patch_size, W_padded // self.patch_size
        
        # Extract features
        context = torch.enable_grad() if need_grad else torch.no_grad()
        with context:
            dino = self.dino_extractor.dino
            outputs = dino(x_preprocessed, output_hidden_states=True)
            num_registers = getattr(dino.config, 'num_register_tokens', 4)
            
            if self.extract_layers is not None:
                # Multi-layer extraction
                hidden_states = outputs.hidden_states
                features_list = []
                for layer_idx in self.extract_layers:
                    if layer_idx < len(hidden_states):
                        layer_hidden = hidden_states[layer_idx]
                        patch_tokens = layer_hidden[:, 1 + num_registers:]
                        features_list.append(patch_tokens)
                return features_list
            else:
                # Single layer (last layer)
                patch_tokens = outputs.last_hidden_state[:, 1 + num_registers:]
                return [patch_tokens]
    
    def _compute_distance(
        self, 
        pred_feat: torch.Tensor, 
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between feature tensors."""
        # Optional normalization
        if self.normalize:
            pred_feat = F.normalize(pred_feat, p=2, dim=-1)
            target_feat = F.normalize(target_feat, p=2, dim=-1)
        
        if self.distance_type == 'l1':
            return F.l1_loss(pred_feat, target_feat)
        elif self.distance_type == 'l2':
            return F.mse_loss(pred_feat, target_feat)
        elif self.distance_type == 'cosine':
            pred_flat = pred_feat.flatten(0, 1)
            target_flat = target_feat.flatten(0, 1)
            labels = torch.ones(pred_flat.size(0), device=pred_feat.device)
            return F.cosine_embedding_loss(pred_flat, target_flat, labels)
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}. Use 'l1', 'l2' or 'cosine'.")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        
        # Check size compatibility with DINO patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_h = (H // self.patch_size) * self.patch_size
            new_w = (W // self.patch_size) * self.patch_size
            pred_resized = F.interpolate(pred, size=(new_h, new_w), mode='bilinear', align_corners=False)
            target_resized = F.interpolate(target, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            pred_resized = pred
            target_resized = target
        
        if self._use_shared_extractor:
            # Use shared DINOFeatureExtractor with multi-layer support
            pred_features = self._extract_features_shared(pred_resized, need_grad=True)
            with torch.no_grad():
                target_features = self._extract_features_shared(target_resized, need_grad=False)
            
            # Compute loss for each layer and average
            total_loss = 0.0
            for pred_feat, target_feat in zip(pred_features, target_features):
                total_loss = total_loss + self._compute_distance(pred_feat, target_feat)
            loss = total_loss / len(pred_features)
        else:
            # Legacy mode: use raw DINO model
            pred_norm = self._preprocess(pred_resized, apply_gamma=False)
            target_norm = self._preprocess(target_resized, apply_gamma=False)
            
            pred_tokens = self._extract_features_raw(pred_norm)
            with torch.no_grad():
                target_tokens = self._extract_features_raw(target_norm)
            
            loss = self._compute_distance(pred_tokens, target_tokens)
        
        return self.loss_weight * loss


class DINOCompositeLoss(nn.Module):
    """
    Composite Loss for DINO-guided Restormer training.
    
    Combines multiple loss components:
    - Charbonnier: pixel-level reconstruction
    - SSIM: structural similarity
    - Perceptual: VGG feature matching
    - DINO Semantic: semantic consistency (L1/L2/cosine distance)
    
    Args:
        dino_model: Pre-loaded DINOv3 model or DINOFeatureExtractor (optional)
        lambda_rec: Weight for Charbonnier loss (default: 1.0)
        lambda_ssim: Weight for SSIM loss (default: 1.0)
        lambda_per: Weight for Perceptual loss (default: 0.1)
        lambda_sem: Weight for DINO semantic loss (default: 0.05)
        use_perceptual: Whether to use perceptual loss (default: True)
        use_semantic: Whether to use DINO semantic loss (default: True)
        dino_gamma: Gamma correction for DINO preprocessing
        semantic_distance: Distance type - 'l1', 'l2' or 'cosine' (default: 'cosine')
        semantic_layers: Which DINO layers to use for loss (default: None = last layer)
        semantic_normalize: Whether to normalize features (default: True)
    """
    
    def __init__(
        self,
        dino_model: Optional[nn.Module] = None,
        lambda_rec: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_per: float = 0.1,
        lambda_sem: float = 0.05,
        use_perceptual: bool = True,
        use_semantic: bool = True,
        dino_gamma: float = 0.4,
        semantic_distance: str = 'cosine',
        semantic_layers: Optional[List[int]] = None,
        semantic_normalize: bool = True,
    ):
        super().__init__()
        
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.lambda_per = lambda_per
        self.lambda_sem = lambda_sem
        self.use_perceptual = use_perceptual
        self.use_semantic = use_semantic and (dino_model is not None)
        
        # Initialize loss components
        self.charbonnier = CharbonnierLoss(loss_weight=1.0)
        self.ssim = SSIMLoss(loss_weight=1.0)
        
        if use_perceptual:
            self.perceptual = PerceptualLoss(loss_weight=1.0)
        
        if self.use_semantic and dino_model is not None:
            self.semantic = DINOSemanticLoss(
                dino_model=dino_model,
                gamma=dino_gamma,
                loss_weight=1.0,
                distance_type=semantic_distance,
                extract_layers=semantic_layers,
                normalize=semantic_normalize,
            )
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.
        
        Args:
            pred: Predicted/enhanced image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss values for logging
        """
        loss_dict = {}
        
        # 1. Charbonnier (reconstruction)
        l_rec = self.charbonnier(pred, target)
        loss_dict['l_rec'] = l_rec.item()
        
        # 2. SSIM (structure)
        l_ssim = self.ssim(pred, target)
        loss_dict['l_ssim'] = l_ssim.item()
        
        # 3. Perceptual (texture)
        l_per = torch.tensor(0.0, device=pred.device)
        if self.use_perceptual:
            l_per = self.perceptual(pred, target)
            loss_dict['l_per'] = l_per.item()
        
        # 4. DINO Semantic (semantic consistency)
        l_sem = torch.tensor(0.0, device=pred.device)
        if self.use_semantic:
            l_sem = self.semantic(pred, target)
            loss_dict['l_sem'] = l_sem.item()
        
        # Total weighted loss
        total_loss = (
            self.lambda_rec * l_rec +
            self.lambda_ssim * l_ssim +
            self.lambda_per * l_per +
            self.lambda_sem * l_sem
        )
        loss_dict['l_total'] = total_loss.item()
        
        return total_loss, loss_dict
