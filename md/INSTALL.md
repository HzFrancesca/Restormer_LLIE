# Installation

This repository requires Python 3.10+ and PyTorch 2.1+ with CUDA 12.1+ for optimal performance on RTX 4090 GPUs.

## System Requirements

- Python 3.10 or 3.11
- PyTorch 2.1.0+
- CUDA 12.1+ (for GPU acceleration)
- cuDNN 8.9+

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/Restormer.git
cd Restormer
```

### 2. Create conda environment
```bash
conda create -n restormer python=3.10
conda activate restormer
```

### 3. Install PyTorch with CUDA 12.1
```bash
# For CUDA 12.1
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Install BasicSR
```bash
# Without CUDA extensions (faster installation)
python setup.py develop --no_cuda_ext

# With CUDA extensions (requires CUDA toolkit)
python setup.py develop
```

### 6. Verify environment
```bash
python scripts/verify_environment.py
```

## CUDA 12.1+ Installation Guide

### Windows
1. Download CUDA Toolkit 12.1+ from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the prompts
3. Add CUDA to PATH (usually done automatically)
4. Verify installation: `nvcc --version`

### Linux (Ubuntu)
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-1

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

## Development Installation

For development and testing:
```bash
pip install -r requirements-dev.txt
```

## Troubleshooting

### Python version error
If you see "This project requires Python 3.10 or higher", upgrade your Python:
```bash
conda create -n restormer python=3.10
conda activate restormer
```

### CUDA version warning
If you see a CUDA version warning, consider upgrading to CUDA 12.1+:
- RTX 4090 performs best with CUDA 12.1+
- PyTorch 2.1+ has optimizations for newer CUDA versions

### CUDA extensions compilation error
If CUDA extensions fail to compile:
1. Ensure CUDA toolkit version matches PyTorch CUDA version
2. Try installing without CUDA extensions: `python setup.py develop --no_cuda_ext`

## Download datasets from Google Drive

To download datasets automatically, you need `gdown`:
```bash
pip install gdown
gdown <google_drive_file_id>
```
