"""
severity_estimator.py — Disease Severity Estimation from Grad-CAM

Uses the Grad-CAM heatmap produced by the existing explainability
pipeline to estimate what proportion of the leaf is affected and map
that to a severity level:

    < 15 %  →  Mild
    15–40 % →  Moderate
    > 40 %  →  Severe

Usage:
    from severity_estimator import estimate_severity

    result = estimate_severity(gradcam_heatmap, original_pil_image)
    print(result["severity"], result["infected_pct"])
"""

import numpy as np
from PIL import Image

# =====================================================================
# CONFIGURABLE PARAMETERS
# =====================================================================

# Heatmap activation threshold (0–1 scale).
# Pixels with a Grad-CAM value above this are considered "infected".
ACTIVATION_THRESHOLD = 0.35

# Severity-level boundaries (infected-area percentage)
MILD_UPPER = 15.0       # < 15 % → Mild
MODERATE_UPPER = 40.0   # 15–40 % → Moderate
                         # > 40 % → Severe


# =====================================================================
# PUBLIC API
# =====================================================================

def estimate_severity(
    gradcam_heatmap: np.ndarray,
    original_image: Image.Image = None,
    activation_threshold: float = ACTIVATION_THRESHOLD,
    mild_upper: float = MILD_UPPER,
    moderate_upper: float = MODERATE_UPPER,
) -> dict:
    """
    Estimate disease severity from a Grad-CAM heatmap.

    Parameters
    ----------
    gradcam_heatmap      : numpy array (H, W) with values in [0, 1].
                           This is the raw heatmap returned by
                           ``generate_gradcam()``.
    original_image       : (Optional) PIL.Image — not used for
                           computation, reserved for future leaf-mask
                           segmentation.
    activation_threshold : float — Grad-CAM value above which a pixel
                           is counted as "infected".
    mild_upper           : float — upper bound (%) for Mild severity.
    moderate_upper       : float — upper bound (%) for Moderate severity.

    Returns
    -------
    dict with keys:
        severity      : str   — "Mild", "Moderate", or "Severe"
        infected_pct  : float — percentage of activated area (0–100)
        color         : str   — hex colour for the severity label
                                (green / orange / red)
    """

    # ---- Validate input ----
    if gradcam_heatmap is None or gradcam_heatmap.size == 0:
        return {
            "severity": "Unknown",
            "infected_pct": 0.0,
            "color": "#808080",   # grey
        }

    # ---- Compute infected-area ratio ----
    # Binary mask: True where activation exceeds threshold
    mask = gradcam_heatmap >= activation_threshold
    total_pixels = mask.size
    infected_pixels = int(mask.sum())

    infected_pct = (infected_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # ---- Map to severity level ----
    if infected_pct < mild_upper:
        severity = "Mild"
        color = "#4CAF50"   # green
    elif infected_pct < moderate_upper:
        severity = "Moderate"
        color = "#FF9800"   # orange
    else:
        severity = "Severe"
        color = "#F44336"   # red

    return {
        "severity": severity,
        "infected_pct": round(infected_pct, 1),
        "color": color,
    }


# =====================================================================
# Quick self-test  (python severity_estimator.py)
# =====================================================================

if __name__ == "__main__":
    # Simulate a heatmap where ~25 % of pixels are "hot"
    fake_heatmap = np.zeros((224, 224), dtype=np.float32)
    fake_heatmap[50:160, 50:160] = 0.8   # hot region
    result = estimate_severity(fake_heatmap)
    print(f"Severity : {result['severity']}")
    print(f"Infected : {result['infected_pct']} %")
    print(f"Colour   : {result['color']}")
