#!/usr/bin/env python
"""
环境验证脚本 - 检查项目所需的 Python 环境配置

验证项目:
- Python 版本 >= 3.10
- PyTorch 版本 >= 2.1
- CUDA 版本 >= 12.1
- 所有核心依赖版本
"""

import sys
from typing import List, Tuple
from packaging import version


# 最低版本要求
MIN_VERSIONS = {
    'python': '3.10',
    'torch': '2.1.0',
    'torchvision': '0.16.0',
    'transformers': '4.30.0',
    'numpy': '1.24.0',
    'scipy': '1.10.0',
    'opencv-python': '4.8.0',
    'einops': '0.6.0',
    'lpips': '0.1.4',
    'tqdm': '4.65.0',
}

MIN_CUDA_VERSION = '12.1'


def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    required = MIN_VERSIONS['python']
    passed = version.parse(current) >= version.parse(required)
    return passed, f"Python {current} (需要 >= {required})"


def check_pytorch() -> Tuple[bool, str, str]:
    """检查 PyTorch 和 CUDA 版本"""
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0]
        required = MIN_VERSIONS['torch']
        torch_ok = version.parse(torch_ver) >= version.parse(required)
        torch_msg = f"PyTorch {torch_ver} (需要 >= {required})"
        
        # CUDA 检查
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "unknown"
            cuda_ok = version.parse(cuda_ver) >= version.parse(MIN_CUDA_VERSION)
            cuda_msg = f"CUDA {cuda_ver} (需要 >= {MIN_CUDA_VERSION})"
            gpu_name = torch.cuda.get_device_name(0)
            cuda_msg += f" - GPU: {gpu_name}"
        else:
            cuda_ok = False
            cuda_msg = "CUDA 不可用"
        
        return torch_ok, torch_msg, cuda_ok, cuda_msg
    except ImportError:
        return False, "PyTorch 未安装", False, "CUDA 检查跳过"


def check_package_version(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查单个包的版本"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    required = MIN_VERSIONS.get(package_name)
    if required is None:
        return True, f"{package_name}: 无版本要求"
    
    try:
        module = __import__(import_name)
        current = getattr(module, '__version__', 'unknown')
        # 处理版本字符串中的额外信息
        current_clean = current.split('+')[0].split('.post')[0]
        passed = version.parse(current_clean) >= version.parse(required)
        status = "✓" if passed else "✗"
        return passed, f"{status} {package_name}: {current} (需要 >= {required})"
    except ImportError:
        return False, f"✗ {package_name}: 未安装 (需要 >= {required})"


def check_opencv() -> Tuple[bool, str]:
    """检查 OpenCV 版本"""
    required = MIN_VERSIONS['opencv-python']
    try:
        import cv2
        current = cv2.__version__
        passed = version.parse(current) >= version.parse(required)
        status = "✓" if passed else "✗"
        return passed, f"{status} opencv-python: {current} (需要 >= {required})"
    except ImportError:
        return False, f"✗ opencv-python: 未安装 (需要 >= {required})"


def verify_environment() -> bool:
    """运行完整的环境验证"""
    print("=" * 60)
    print("环境验证报告")
    print("=" * 60)
    
    all_passed = True
    errors = []
    warnings = []
    
    # Python 版本
    print("\n[Python 环境]")
    py_ok, py_msg = check_python_version()
    status = "✓" if py_ok else "✗"
    print(f"  {status} {py_msg}")
    if not py_ok:
        all_passed = False
        errors.append(py_msg)
    
    # PyTorch 和 CUDA
    print("\n[深度学习框架]")
    torch_ok, torch_msg, cuda_ok, cuda_msg = check_pytorch()
    status = "✓" if torch_ok else "✗"
    print(f"  {status} {torch_msg}")
    if not torch_ok:
        all_passed = False
        errors.append(torch_msg)
    
    status = "✓" if cuda_ok else "⚠"
    print(f"  {status} {cuda_msg}")
    if not cuda_ok:
        warnings.append(cuda_msg)
    
    # 核心依赖
    print("\n[核心依赖]")
    packages = [
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('einops', 'einops'),
        ('lpips', 'lpips'),
        ('tqdm', 'tqdm'),
    ]
    
    for pkg_name, import_name in packages:
        ok, msg = check_package_version(pkg_name, import_name)
        print(f"  {msg}")
        if not ok:
            all_passed = False
            errors.append(msg)
    
    # OpenCV 单独检查
    cv_ok, cv_msg = check_opencv()
    print(f"  {cv_msg}")
    if not cv_ok:
        all_passed = False
        errors.append(cv_msg)
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed and not warnings:
        print("✓ 所有检查通过！环境配置正确。")
    elif all_passed:
        print("⚠ 环境基本满足要求，但有以下警告：")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("✗ 环境验证失败，请修复以下问题：")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = verify_environment()
    sys.exit(0 if success else 1)
