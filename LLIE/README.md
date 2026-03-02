# Low-Light Image Enhancement

This directory contains scripts and configurations for the low-light image enhancement task using the Restormer model.

## 📁 Directory Structure

```
LLIE/
├── Datasets/                          # Dataset directory
├── Options/                           # Training and testing configuration files
│   ├── LowLight_Restormer.yml
│   └── ...
├── metrics_cal.py                     # Supervised metrics (PSNR, SSIM, LPIPS with reference)
├── unsupervised_metrics_cal.py        # Unsupervised metrics (NIQE, LPIPS without reference)
├── test_unsupervised_metrics.py       # Test script for unsupervised metrics
├── test.py                            # Model testing script
├── utils.py                           # Utility functions
├── README.md                          # This file
└── UNSUPERVISED_METRICS_README.md     # Detailed guide for unsupervised metrics
```

## 🚀 Training

To train the model, run the following command:

```bash
python basicsr/train.py -opt LLIE/Options/LowLight_Restormer.yml
```

## 🧪 Testing

To test the model, run the following command:

```bash
python LLIE/test.py -opt LLIE/Options/LowLight_Restormer.yml --weights <path_to_your_model.pth>
```

## 📊 Evaluation

### Supervised Metrics (with reference images)

Use `metrics_cal.py` to calculate PSNR, SSIM, and LPIPS when you have ground truth images:

```bash
python LLIE/metrics_cal.py \
    --dirA ./path/to/ground_truth \
    --dirB ./path/to/enhanced \
    --type png \
    --use_gpu \
    --save_txt results/supervised_metrics.txt
```

**Supported metrics:**
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Higher is better (0-1)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better

### Unsupervised Metrics (without reference images) ✨ NEW

Use `unsupervised_metrics_cal.py` to evaluate image quality without ground truth:

```bash
python LLIE/unsupervised_metrics_cal.py \
    --dir ./path/to/enhanced \
    --type png \
    --use_gpu \
    --save_txt results/unsupervised_metrics.txt
```

**Supported metrics:**
- **NIQE** (Natural Image Quality Evaluator): Lower is better (typically 0-10)
- **LPIPS** (Perceptual Quality Stability): Lower is better

For detailed usage and examples, see [UNSUPERVISED_METRICS_README.md](UNSUPERVISED_METRICS_README.md)

### Testing the Unsupervised Metrics Tool

To verify that the unsupervised metrics tool is properly installed:

```bash
cd LLIE
python test_unsupervised_metrics.py
```

This will:
1. Check all dependencies
2. Verify NIQE parameter files
3. Test the metrics calculation
4. Run a complete workflow with test images

## 📋 Quick Start Example

Complete workflow for evaluating enhanced images:

```bash
# 1. Train the model (if needed)
python basicsr/train.py -opt LLIE/Options/LowLight_Restormer.yml

# 2. Test/enhance images
python LLIE/test.py -opt LLIE/Options/LowLight_Restormer.yml --weights pretrained_models/restormer_llie.pth

# 3. Evaluate with supervised metrics (if you have ground truth)
python LLIE/metrics_cal.py \
    --dirA ./datasets/LOL-v2/Real_captured/Test/Normal \
    --dirB ./results/enhanced \
    --use_gpu \
    --save_txt results/supervised_metrics.txt

# 4. Evaluate with unsupervised metrics (always applicable)
python LLIE/unsupervised_metrics_cal.py \
    --dir ./results/enhanced \
    --use_gpu \
    --save_txt results/unsupervised_metrics.txt
```

## 🔧 Requirements

Essential packages:
- PyTorch
- OpenCV (cv2)
- NumPy
- lpips
- scipy
- natsort
- tqdm

Install dependencies:
```bash
pip install torch opencv-python numpy lpips scipy natsort tqdm
```

## 📚 Documentation

- **Main Project**: See root README for overall project structure
- **Unsupervised Metrics**: See [UNSUPERVISED_METRICS_README.md](UNSUPERVISED_METRICS_README.md) for detailed guide
- **Training Options**: Check `Options/` directory for configuration examples
