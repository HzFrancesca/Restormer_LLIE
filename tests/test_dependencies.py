"""
依赖版本属性测试

Property 1: 依赖版本满足最低要求
Validates: Requirements 5.1

使用 hypothesis 进行属性测试，验证所有核心依赖版本满足最低要求
"""

import pytest
from packaging import version
from hypothesis import given, strategies as st, settings


# 最低版本要求映射
MIN_VERSIONS = {
    'torch': ('torch', '2.1.0'),
    'torchvision': ('torchvision', '0.16.0'),
    'transformers': ('transformers', '4.30.0'),
    'numpy': ('numpy', '1.24.0'),
    'scipy': ('scipy', '1.10.0'),
    'opencv-python': ('cv2', '4.8.0'),
    'einops': ('einops', '0.6.0'),
    'lpips': ('lpips', '0.1.4'),
    'tqdm': ('tqdm', '4.65.0'),
}


def get_package_version(import_name: str) -> str:
    """获取包的版本号"""
    try:
        module = __import__(import_name)
        ver = getattr(module, '__version__', None)
        if ver:
            # 清理版本字符串
            return ver.split('+')[0].split('.post')[0]
        return None
    except ImportError:
        return None


class TestDependencyVersions:
    """
    Property 1: 依赖版本满足最低要求
    
    *For any* 已安装的核心依赖包，其版本号应大于或等于指定的最低版本要求。
    **Validates: Requirements 5.1**
    """
    
    @pytest.mark.parametrize("package_name,import_info", [
        (name, info) for name, info in MIN_VERSIONS.items()
    ])
    def test_package_version_meets_minimum(self, package_name, import_info):
        """测试单个包版本满足最低要求"""
        import_name, min_ver = import_info
        current_ver = get_package_version(import_name)
        
        if current_ver is None:
            pytest.skip(f"{package_name} not installed")
        
        assert version.parse(current_ver) >= version.parse(min_ver), \
            f"{package_name}: {current_ver} < {min_ver} (minimum required)"


# Property-based test using hypothesis
# Feature: python-env-upgrade, Property 1: 依赖版本满足最低要求
@settings(max_examples=100)
@given(st.sampled_from(list(MIN_VERSIONS.keys())))
def test_property_dependency_version_satisfies_minimum(package_name):
    """
    Property 1: 依赖版本满足最低要求
    
    *For any* 已安装的核心依赖包，其版本号应大于或等于指定的最低版本要求。
    **Validates: Requirements 5.1**
    """
    import_name, min_ver = MIN_VERSIONS[package_name]
    current_ver = get_package_version(import_name)
    
    # 如果包未安装，跳过（不是失败）
    if current_ver is None:
        return
    
    assert version.parse(current_ver) >= version.parse(min_ver), \
        f"{package_name}: version {current_ver} does not meet minimum {min_ver}"


class TestDependencyImports:
    """测试所有核心依赖可以正常导入"""
    
    def test_torch_import(self):
        """测试 torch 导入"""
        import torch
        assert torch is not None
    
    def test_torchvision_import(self):
        """测试 torchvision 导入"""
        import torchvision
        assert torchvision is not None
    
    def test_numpy_import(self):
        """测试 numpy 导入"""
        import numpy
        assert numpy is not None
    
    def test_scipy_import(self):
        """测试 scipy 导入"""
        import scipy
        assert scipy is not None
    
    def test_cv2_import(self):
        """测试 opencv 导入"""
        import cv2
        assert cv2 is not None
    
    def test_einops_import(self):
        """测试 einops 导入"""
        import einops
        assert einops is not None
    
    def test_tqdm_import(self):
        """测试 tqdm 导入"""
        import tqdm
        assert tqdm is not None
    
    def test_transformers_import(self):
        """测试 transformers 导入"""
        try:
            import transformers
            assert transformers is not None
        except ImportError:
            pytest.skip("transformers not installed")
    
    def test_lpips_import(self):
        """测试 lpips 导入"""
        try:
            import lpips
            assert lpips is not None
        except ImportError:
            pytest.skip("lpips not installed")
