"""
DINOv3 Feature Map Visualization for Low-Light Images

This script extracts feature maps from DINOv3 H+ model and visualizes them
using PCA to reduce to 3 channels (RGB). This helps verify if DINOv3 can
extract clear object contours from dark regions in low-light images.

Usage:
    python scripts/dinov3_feature_viz.py --image path/to/low_light_image.png
    python scripts/dinov3_feature_viz.py --image path/to/image.png --output output_dir

Requirements:
    pip install transformers torch torchvision pillow scikit-learn matplotlib
    
Note: Requires transformers >= 4.56.0 for DINOv3 support.
      If not available, install from git: pip install git+https://github.com/huggingface/transformers
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


def load_dinov3_model(model_path: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"):
    """Load DINOv3 H+ model and processor from local path or HuggingFace."""
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise ImportError(
            "transformers library not found. Install with: "
            "pip install git+https://github.com/huggingface/transformers"
        )
    
    print(f"Loading DINOv3 model: {model_path}")
    
    # Check if it's a local path
    is_local = os.path.isdir(model_path)
    
    # Load from local path or HuggingFace
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=is_local)
    model = AutoModel.from_pretrained(model_path, local_files_only=is_local)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, processor, device


def extract_features(model, processor, image: Image.Image, device: torch.device):
    """Extract patch features from DINOv3 model."""
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get patch tokens (exclude CLS and register tokens)
    # DINOv3: 1 CLS + 4 register + N patch tokens
    last_hidden_state = outputs.last_hidden_state  # [B, num_tokens, hidden_dim]
    patch_tokens = last_hidden_state[:, 5:, :]  # Skip CLS (1) + registers (4)
    
    return patch_tokens


def visualize_features_pca(
    patch_tokens: torch.Tensor,
    original_image: Image.Image,
    output_path: str = None,
    patch_size: int = 16,
):
    """
    Visualize feature maps using PCA to reduce to 3 channels (RGB).
    
    Args:
        patch_tokens: [1, num_patches, hidden_dim] tensor
        original_image: Original input image for reference
        output_path: Path to save visualization
        patch_size: Patch size used by the model (default 16 for DINOv3)
    """
    # Convert to numpy
    features = patch_tokens.squeeze(0).cpu().numpy()  # [num_patches, hidden_dim]
    num_patches = features.shape[0]
    
    # Calculate spatial dimensions
    # For DINOv3 with 224x224 input and patch_size=16: 14x14 = 196 patches
    h = w = int(np.sqrt(num_patches))
    
    print(f"Feature shape: {features.shape}")
    print(f"Spatial resolution: {h}x{w} patches")
    
    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)  # [num_patches, 3]
    
    # Print explained variance
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Reshape to spatial dimensions
    features_pca = features_pca.reshape(h, w, 3)
    
    # Normalize to [0, 1] for visualization
    features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Low-Light Image")
    axes[0].axis("off")
    
    # PCA feature map (RGB)
    axes[1].imshow(features_pca)
    axes[1].set_title("DINOv3 H+ Features (PCA → RGB)")
    axes[1].axis("off")
    
    # Upscaled feature map to match original image size
    from PIL import Image as PILImage
    features_pca_uint8 = (features_pca * 255).astype(np.uint8)
    features_pca_img = PILImage.fromarray(features_pca_uint8)
    features_pca_upscaled = features_pca_img.resize(
        original_image.size, resample=PILImage.BILINEAR
    )
    axes[2].imshow(features_pca_upscaled)
    axes[2].set_title("Features Upscaled to Original Size")
    axes[2].axis("off")
    
    plt.suptitle("DINOv3 Feature Visualization for Low-Light Image Enhancement", fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")
        
        # Also save individual feature map
        feature_only_path = output_path.replace(".png", "_features_only.png")
        features_pca_upscaled.save(feature_only_path)
        print(f"Feature map saved to: {feature_only_path}")
    
    plt.show()
    
    return features_pca


def analyze_dark_regions(
    original_image: Image.Image,
    features_pca: np.ndarray,
    dark_threshold: float = 0.2,
):
    """
    Analyze if DINOv3 extracts meaningful features from dark regions.
    
    Args:
        original_image: Original input image
        features_pca: PCA-reduced features [H, W, 3]
        dark_threshold: Threshold to identify dark regions (0-1)
    """
    # Convert image to grayscale and normalize
    img_gray = np.array(original_image.convert("L")) / 255.0
    
    # Resize to match feature map resolution
    h, w = features_pca.shape[:2]
    from PIL import Image as PILImage
    img_gray_resized = np.array(
        PILImage.fromarray((img_gray * 255).astype(np.uint8)).resize((w, h))
    ) / 255.0
    
    # Identify dark regions
    dark_mask = img_gray_resized < dark_threshold
    
    # Calculate feature variance in dark vs bright regions
    features_flat = features_pca.reshape(-1, 3)
    dark_mask_flat = dark_mask.flatten()
    
    dark_features = features_flat[dark_mask_flat]
    bright_features = features_flat[~dark_mask_flat]
    
    print("\n" + "=" * 50)
    print("Dark Region Analysis")
    print("=" * 50)
    print(f"Dark region pixels: {dark_mask.sum()} / {dark_mask.size} ({dark_mask.mean():.1%})")
    
    if len(dark_features) > 0:
        dark_var = np.var(dark_features, axis=0).mean()
        print(f"Feature variance in dark regions: {dark_var:.4f}")
    
    if len(bright_features) > 0:
        bright_var = np.var(bright_features, axis=0).mean()
        print(f"Feature variance in bright regions: {bright_var:.4f}")
    
    if len(dark_features) > 0 and len(bright_features) > 0:
        ratio = dark_var / (bright_var + 1e-8)
        print(f"Dark/Bright variance ratio: {ratio:.2f}")
        
        if ratio > 0.5:
            print("\n✓ DINOv3 extracts meaningful features from dark regions!")
            print("  This suggests it can capture object contours even in low-light areas.")
        else:
            print("\n⚠ Feature variance is lower in dark regions.")
            print("  May need additional processing for low-light enhancement.")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DINOv3 features for low-light images"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input low-light image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dinov3/results",
        help="Output directory for visualizations (default: dinov3/results)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=r"D:\Downloads\AI_models\modelscope\hub\models\facebook\dinov3-vith16plus-pretrain-lvd1689m",
        help="DINOv3 model path (default: local H+ variant)"
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=0.2,
        help="Threshold for dark region analysis (0-1, default: 0.2)"
    )
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Setup output path
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(
            args.output,
            Path(args.image).stem + "_dinov3_viz.png"
        )
    else:
        output_path = str(Path(args.image).with_suffix("")) + "_dinov3_viz.png"
    
    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Load model
    model, processor, device = load_dinov3_model(args.model)
    
    # Extract features
    print("Extracting features...")
    patch_tokens = extract_features(model, processor, image, device)
    print(f"Extracted {patch_tokens.shape[1]} patch tokens with dim {patch_tokens.shape[2]}")
    
    # Visualize with PCA
    print("Applying PCA and visualizing...")
    features_pca = visualize_features_pca(
        patch_tokens,
        image,
        output_path=output_path,
    )
    
    # Analyze dark regions
    analyze_dark_regions(image, features_pca, args.dark_threshold)
    
    print("\nDone! Check the visualization to see if object contours are clear in dark regions.")


if __name__ == "__main__":
    main()
