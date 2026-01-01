"""
环境验证测试

测试 Python 版本、PyTorch 版本和 CUDA 可用性检查
Requirements: 1.3, 2.1, 2.2
"""

import sys
import pytest
from packaging import version


class TestPythonVersion:
    """Python 版本检查测试"""
    
    def test_python_version_minimum(self):
        """测试 Python 版本 >= 3.10"""
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert version.parse(current) >= version.parse("3.10"), \
            f"Python 3.10+ required, got {current}"
    
    def test_python_version_info(self):
        """测试 Python 版本信息可访问"""
        assert sys.version_info.major >= 3
        assert sys.version_info.minor >= 10


class TestPyTorchVersion:
    """PyTorch 版本检查测试"""
    
    def test_torch_import(self):
        """测试 PyTorch 可以导入"""
        import torch
        assert torch is not None
    
    def test_torch_version_minimum(self):
        """测试 PyTorch 版本 >= 2.1"""
        import torch
        torch_ver = torch.__version__.split('+')[0]
        assert version.parse(torch_ver) >= version.parse("2.1.0"), \
            f"PyTorch 2.1+ required, got {torch_ver}"


class TestCUDAAvailability:
    """CUDA 可用性检查测试"""
    
    def test_cuda_available(self):
        """测试 CUDA 是否可用"""
        import torch
        # 这个测试在没有 GPU 的环境中会跳过
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        assert torch.cuda.is_available()
    
    def test_cuda_version(self):
        """测试 CUDA 版本 >= 12.1"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cuda_ver = torch.version.cuda
        if cuda_ver:
            assert version.parse(cuda_ver) >= version.parse("12.1"), \
                f"CUDA 12.1+ recommended, got {cuda_ver}"
    
    def test_gpu_device_accessible(self):
        """测试 GPU 设备可访问"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device_count = torch.cuda.device_count()
        assert device_count > 0, "No CUDA devices found"
        
        # 测试可以获取设备名称
        device_name = torch.cuda.get_device_name(0)
        assert device_name is not None
