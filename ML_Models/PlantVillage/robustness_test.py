"""
robustness_test.py — Real-World Robustness Testing for Plant Disease Model

Applies common real-world perturbations (blur, low brightness, noise, rotation)
to a test dataset and reports accuracy under each condition.

Usage:
    python robustness_test.py --data_dir ./Plant_leave_diseases_dataset_with_augmentation
                              --model_path plant_disease_model.pth
                              --num_samples 500

All evaluation runs on CPU by default.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

NUM_CLASSES = 39
IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Perturbation functions  (PIL Image → PIL Image)
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image, radius=3):
    """Apply Gaussian blur to simulate camera out-of-focus."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_low_brightness(image, factor=0.4):
    """Reduce brightness to simulate low-light conditions."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)  # factor < 1 darkens


def apply_gaussian_noise(image, std=25):
    """Add Gaussian noise to simulate sensor noise."""
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std, img_array.shape).astype(np.float32)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_rotation(image, angle=15):
    """Rotate the image to simulate tilted camera / leaf orientation."""
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


# Registry of perturbations — easy to extend
PERTURBATIONS = {
    "Clean":     None,                   # no perturbation
    "Blur":      apply_gaussian_blur,
    "Low Light": apply_low_brightness,
    "Noise":     apply_gaussian_noise,
    "Rotation":  apply_rotation,
}


# ---------------------------------------------------------------------------
# Standard pre-processing (same as training validation)
# ---------------------------------------------------------------------------

STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model loader (CPU only)
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load the trained EfficientNet-B0 model on CPU."""
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_condition(model, image_label_pairs, perturbation_fn=None):
    """
    Run inference on a list of (PIL_image, label) pairs.

    Args:
        model              : loaded PyTorch model (eval mode, CPU).
        image_label_pairs  : list of (PIL.Image, int_label).
        perturbation_fn    : optional callable(PIL.Image) → PIL.Image.

    Returns:
        accuracy : float in [0, 1].
    """
    correct = 0
    total = len(image_label_pairs)

    for pil_image, label in image_label_pairs:
        # Apply perturbation if provided
        if perturbation_fn is not None:
            pil_image = perturbation_fn(pil_image)

        # Pre-process and add batch dimension
        tensor = STANDARD_TRANSFORM(pil_image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()

        if pred == label:
            correct += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataset sampler (uses ImageFolder structure)
# ---------------------------------------------------------------------------

def sample_image_label_pairs(data_dir, num_samples):
    """
    Sample (PIL_image, label) pairs from an ImageFolder-structured directory.

    Images are loaded as-is (no transforms) so that perturbations can be
    applied to the raw image before the standard pre-processing.
    """
    # Use a plain ImageFolder with no transform — we will transform later
    dataset = datasets.ImageFolder(data_dir, transform=None)

    # Sub-sample for speed
    indices = list(range(len(dataset)))
    if num_samples < len(indices):
        random.seed(42)
        indices = random.sample(indices, num_samples)

    pairs = []
    for idx in indices:
        path, label = dataset.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            pairs.append((img, label))
        except Exception:
            continue  # skip corrupt images

    return pairs


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_robustness_test(model_path, data_dir, num_samples=500):
    """Run the full robustness evaluation and print results."""

    print("=" * 50)
    print("  Plant Disease Model — Robustness Test")
    print("=" * 50)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully (CPU).")

    # Load test images
    print(f"Sampling {num_samples} images from: {data_dir}")
    pairs = sample_image_label_pairs(data_dir, num_samples)
    print(f"Loaded {len(pairs)} images.\n")

    # Evaluate under each condition
    results = {}
    for name, fn in PERTURBATIONS.items():
        print(f"  Evaluating: {name} ...", end="", flush=True)
        acc = evaluate_condition(model, pairs, perturbation_fn=fn)
        results[name] = acc
        print(f"  {acc * 100:.1f}%")

    # Print summary table
    print("\n" + "=" * 35)
    print(f"{'Condition':<15} {'Accuracy':>10}")
    print("-" * 35)
    for name, acc in results.items():
        print(f"{name:<15} {acc * 100:.1f}%")
    print("=" * 35)

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test plant disease model robustness under real-world perturbations."
    )
    parser.add_argument(
        "--model_path", type=str, default="plant_disease_model.pth",
        help="Path to the trained .pth model file."
    )
    parser.add_argument(
        "--data_dir", type=str, default="./Plant_leave_diseases_dataset_with_augmentation",
        help="Path to the PlantVillage dataset (ImageFolder structure)."
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="Number of images to sample for testing (default 500)."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Dataset directory not found: {args.data_dir}")
        print("Please provide the correct path via --data_dir.")
        exit(1)

    if not os.path.isfile(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        print("Please provide the correct path via --model_path.")
        exit(1)

    run_robustness_test(args.model_path, args.data_dir, args.num_samples)
