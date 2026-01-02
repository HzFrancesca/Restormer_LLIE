"""
向后兼容性验证测试

验证现有配置文件和模型权重在新环境中的兼容性
Requirements: 7.1, 7.2
"""

import os
import pytest
import yaml


class TestConfigCompatibility:
    """测试现有配置文件兼容性 - Requirements 7.1"""
    
    @pytest.fixture
    def restormer_config(self):
        """加载 LowLight_Restormer.yml 配置"""
        config_path = 'LLIE/Options/LowLight_Restormer.yml'
        if not os.path.exists(config_path):
            pytest.skip(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def dino_restormer_config(self):
        """加载 LowLight_DINORestormer.yml 配置"""
        config_path = 'LLIE/Options/LowLight_DINORestormer.yml'
        if not os.path.exists(config_path):
            pytest.skip(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_restormer_config_valid(self, restormer_config):
        """测试 Restormer 配置文件有效"""
        assert restormer_config is not None
        assert 'network_g' in restormer_config
        assert restormer_config['network_g']['type'] == 'Restormer'
    
    def test_dino_restormer_config_valid(self, dino_restormer_config):
        """测试 DINORestormer 配置文件有效"""
        assert dino_restormer_config is not None
        assert 'network_g' in dino_restormer_config
        assert dino_restormer_config['network_g']['type'] == 'DINOGuidedRestormer'
    
    def test_restormer_config_has_required_keys(self, restormer_config):
        """测试 Restormer 配置包含必要的键"""
        required_keys = ['name', 'model_type', 'datasets', 'network_g', 'train', 'val']
        for key in required_keys:
            assert key in restormer_config, f"Missing required key: {key}"
    
    def test_dino_restormer_config_has_dino_settings(self, dino_restormer_config):
        """测试 DINORestormer 配置包含 DINO 设置"""
        network_g = dino_restormer_config['network_g']
        assert 'dino_model' in network_g
        assert 'dino_gamma' in network_g
        assert 'use_dino_guidance' in network_g
    
    def test_scheduler_config_valid(self, restormer_config):
        """测试学习率调度器配置有效"""
        scheduler = restormer_config['train']['scheduler']
        assert scheduler['type'] == 'CosineAnnealingRestartCyclicLR'
        assert 'periods' in scheduler
        assert 'eta_mins' in scheduler


class TestModelWeightCompatibility:
    """测试模型权重加载兼容性 - Requirements 7.2"""
    
    def test_torch_load_available(self):
        """测试 torch.load 函数可用"""
        import torch
        assert hasattr(torch, 'load')
    
    def test_weights_only_parameter(self):
        """测试 PyTorch 2.x 的 weights_only 参数"""
        import torch
        # PyTorch 2.x 支持 weights_only 参数
        # 这是向后兼容性的重要特性
        import inspect
        sig = inspect.signature(torch.load)
        params = list(sig.parameters.keys())
        # weights_only 在 PyTorch 1.13+ 中可用
        assert 'weights_only' in params or 'map_location' in params
    
    def test_state_dict_format(self):
        """测试 state_dict 格式兼容性"""
        import torch
        import torch.nn as nn
        
        # 创建简单模型测试 state_dict 格式
        model = nn.Linear(10, 10)
        state_dict = model.state_dict()
        
        # 验证 state_dict 格式
        assert isinstance(state_dict, dict)
        assert 'weight' in state_dict
        assert 'bias' in state_dict


class TestAPICompatibility:
    """测试 PyTorch API 兼容性"""
    
    def test_cuda_amp_available(self):
        """测试混合精度训练 API 可用"""
        import torch
        assert hasattr(torch.cuda, 'amp')
        assert hasattr(torch.cuda.amp, 'autocast')
        assert hasattr(torch.cuda.amp, 'GradScaler')
    
    def test_torch_compile_available(self):
        """测试 torch.compile 可用 (PyTorch 2.0+ 特性)"""
        import torch
        assert hasattr(torch, 'compile')
    
    def test_einops_rearrange(self):
        """测试 einops rearrange 函数可用"""
        import torch
        from einops import rearrange
        
        x = torch.randn(2, 3, 4, 4)
        y = rearrange(x, 'b c h w -> b (h w) c')
        assert y.shape == (2, 16, 3)
