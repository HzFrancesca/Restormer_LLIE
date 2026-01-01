"""
DINO-guided Image Restoration Model

This model extends ImageCleanModel to support:
1. DINOCompositeLoss with semantic consistency
2. Shared DINO model between network and loss function
3. Memory-efficient training with optional loss components
"""

import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.models.archs import define_network
from basicsr.utils import get_root_logger


class DINOImageRestorationModel(ImageCleanModel):
    """
    Image restoration model with DINO-guided composite loss.
    
    Extends ImageCleanModel to support DINOCompositeLoss which combines:
    - Charbonnier loss (pixel reconstruction)
    - SSIM loss (structural similarity)
    - Perceptual loss (VGG features)
    - DINO semantic loss (semantic consistency)
    """
    
    def __init__(self, opt):
        # Call parent init (will call init_training_settings)
        super().__init__(opt)
    
    def init_training_settings(self):
        """Initialize training settings with DINO composite loss support."""
        self.net_g.train()
        train_opt = self.opt['train']
        
        # EMA setup (same as parent)
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema, load_path,
                    self.opt['path'].get('strict_load_g', True), 'params_ema'
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        
        # Initialize losses
        self._init_losses(train_opt)
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def _init_losses(self, train_opt):
        """Initialize loss functions based on config."""
        from basicsr.models.losses import DINOCompositeLoss
        
        # Check if using composite loss
        if train_opt.get('composite_opt'):
            comp_opt = train_opt['composite_opt']
            
            # Get DINO model from network if available
            dino_model = None
            if hasattr(self.net_g, 'module'):
                net = self.net_g.module
            else:
                net = self.net_g
            
            # Try to get DINO model from network
            if hasattr(net, 'dino_extractor') and hasattr(net.dino_extractor, 'dino'):
                dino_model = net.dino_extractor.dino
                logger = get_root_logger()
                logger.info('Using DINO model from network for semantic loss')
            elif comp_opt.get('use_semantic', True):
                # Load separate DINO model for loss
                logger = get_root_logger()
                logger.warning('Network has no DINO model, loading separate DINO for loss')
                dino_model = self._load_dino_for_loss(comp_opt)
            
            # Create composite loss
            self.cri_composite = DINOCompositeLoss(
                dino_model=dino_model,
                lambda_rec=comp_opt.get('lambda_rec', 1.0),
                lambda_ssim=comp_opt.get('lambda_ssim', 1.0),
                lambda_per=comp_opt.get('lambda_per', 0.1),
                lambda_sem=comp_opt.get('lambda_sem', 0.05),
                use_perceptual=comp_opt.get('use_perceptual', True),
                use_semantic=comp_opt.get('use_semantic', True),
                dino_gamma=comp_opt.get('dino_gamma', 0.4),
            ).to(self.device)
            
            self.use_composite_loss = True
            logger = get_root_logger()
            logger.info(
                f'Using DINOCompositeLoss: '
                f'rec={comp_opt.get("lambda_rec", 1.0)}, '
                f'ssim={comp_opt.get("lambda_ssim", 1.0)}, '
                f'per={comp_opt.get("lambda_per", 0.1)}, '
                f'sem={comp_opt.get("lambda_sem", 0.05)}'
            )
            
            # Also set cri_pix for compatibility
            self.cri_pix = None
            
        elif train_opt.get('pixel_opt'):
            # Fallback to standard pixel loss
            import importlib
            loss_module = importlib.import_module('basicsr.models.losses')
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
            self.use_composite_loss = False
        else:
            raise ValueError('Either composite_opt or pixel_opt must be specified.')
    
    def _load_dino_for_loss(self, comp_opt):
        """Load DINO model specifically for loss computation."""
        try:
            dino_name = comp_opt.get('dino_model', 'dinov3_vitb16')
            local_path = comp_opt.get('dino_local_path', None)
            
            if local_path:
                from transformers import AutoModel
                dino = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            else:
                dino = torch.hub.load('facebookresearch/dinov3', dino_name)
            
            # Freeze and set to eval
            for param in dino.parameters():
                param.requires_grad = False
            dino.eval()
            
            return dino.to(self.device)
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f'Failed to load DINO for loss: {e}. Disabling semantic loss.')
            return None
    
    def optimize_parameters(self, current_iter):
        """Optimize with composite or standard loss."""
        self.optimizer_g.zero_grad()
        self.log_dict = OrderedDict()
        
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        
        self.output = preds[-1]
        
        if self.use_composite_loss:
            # Use composite loss
            total_loss = 0.0
            for pred in preds:
                loss, loss_components = self.cri_composite(pred, self.gt)
                total_loss = total_loss + loss
            
            # Log individual components (already float from .item())
            for k, v in loss_components.items():
                self.log_dict[k] = v
            
            total_loss.backward()
        else:
            # Standard pixel loss
            l_pix = 0.0
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            
            self.log_dict['l_pix'] = l_pix.item()
            l_pix.backward()
        
        # Gradient clipping
        if self.opt['train'].get('use_grad_clip', False):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        self.optimizer_g.step()
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
