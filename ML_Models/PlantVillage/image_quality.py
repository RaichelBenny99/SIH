"""
image_quality.py — Image Quality Assessment (IQC) Module

Checks uploaded images for common quality problems BEFORE running
the disease-detection model.  Three checks are performed:

    1. Blur detection   — Laplacian variance (low = blurry)
    2. Brightness check — mean pixel value   (too dark / too bright)
    3. Resolution check — minimum width & height in pixels

Usage:
    from image_quality import check_image_quality

    is_good, message, metrics = check_image_quality(pil_image)
"""

import cv2
import numpy as np
from PIL import Image

# =====================================================================
# CONFIGURABLE THRESHOLDS  (tweak these for your use-case)
# =====================================================================

# Laplacian variance below this → image is too blurry
BLUR_THRESHOLD = 50.0

# Mean brightness (0–255 greyscale) bounds
BRIGHTNESS_LOW = 40      # below → too dark
BRIGHTNESS_HIGH = 220    # above → too bright (washed out)

# Minimum acceptable resolution (pixels)
MIN_WIDTH = 100
MIN_HEIGHT = 100


# =====================================================================
# PUBLIC API
# =====================================================================

def check_image_quality(
    pil_image: Image.Image,
    blur_threshold: float = BLUR_THRESHOLD,
    brightness_low: int = BRIGHTNESS_LOW,
    brightness_high: int = BRIGHTNESS_HIGH,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
) -> tuple:
    """
    Assess whether *pil_image* is good enough for reliable prediction.

    Parameters
    ----------
    pil_image       : PIL.Image   — the uploaded image (any mode).
    blur_threshold  : float       — Laplacian-variance cutoff.
    brightness_low  : int (0-255) — reject if mean brightness < this.
    brightness_high : int (0-255) — reject if mean brightness > this.
    min_width       : int         — minimum acceptable width  (px).
    min_height      : int         — minimum acceptable height (px).

    Returns
    -------
    is_good  : bool   — True if the image passes ALL checks.
    message  : str    — Human-readable summary (empty when is_good=True).
    metrics  : dict   — Raw numbers for every check:
                        {blur_score, mean_brightness, width, height}
    """

    # Convert PIL → RGB numpy → greyscale for analysis
    rgb = np.array(pil_image.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ---- 1. Blur detection (Laplacian variance) ----
    # The Laplacian highlights edges; its variance is low when the image
    # is blurry because there are fewer sharp edges.
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # ---- 2. Brightness (mean of greyscale pixels) ----
    mean_brightness = float(grey.mean())

    # ---- 3. Resolution ----
    height, width = rgb.shape[:2]

    # ---- Collect metrics ----
    metrics = {
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "width": width,
        "height": height,
    }

    # ---- Evaluate each check ----
    issues = []  # collect all problems so user sees everything at once

    if blur_score < blur_threshold:
        issues.append(
            f"Image is too blurry (score {blur_score:.1f}, "
            f"need ≥ {blur_threshold:.0f}).\n"
            "💡 Suggestion: Re-take the photo with a steady hand or "
            "use a tripod."
        )

    if mean_brightness < brightness_low:
        issues.append(
            f"Image is too dark (brightness {mean_brightness:.0f}/255, "
            f"need ≥ {brightness_low}).\n"
            "💡 Suggestion: Use better lighting or a camera flash."
        )
    elif mean_brightness > brightness_high:
        issues.append(
            f"Image is over-exposed / too bright "
            f"(brightness {mean_brightness:.0f}/255, "
            f"need ≤ {brightness_high}).\n"
            "💡 Suggestion: Reduce direct sunlight or lower exposure."
        )

    if width < min_width or height < min_height:
        issues.append(
            f"Resolution too low ({width}×{height} px, "
            f"need ≥ {min_width}×{min_height}).\n"
            "💡 Suggestion: Upload a higher-resolution image."
        )

    # ---- Build result ----
    is_good = len(issues) == 0
    message = "\n\n".join(issues) if issues else ""

    return is_good, message, metrics


# =====================================================================
# Quick self-test  (python image_quality.py)
# =====================================================================

if __name__ == "__main__":
    # Create a dummy 300×300 test image
    dummy = Image.fromarray(np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8))
    ok, msg, m = check_image_quality(dummy)
    print(f"is_good={ok}  metrics={m}")
    if msg:
        print(msg)
