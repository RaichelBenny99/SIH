"""
gradcam.py — Grad-CAM Explainable AI for EfficientNet-B0

Generates a class activation heatmap showing which parts of the image
the model focused on when making its prediction.

Usage:
    from gradcam import generate_gradcam, overlay_heatmap

    heatmap = generate_gradcam(model, image_tensor)
    result_image = overlay_heatmap(original_pil_image, heatmap)
"""

import torch
import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Core Grad-CAM implementation
# ---------------------------------------------------------------------------

def generate_gradcam(model, image_tensor, target_class=None):
    """
    Compute a Grad-CAM heatmap for the given image tensor.

    Args:
        model        : A loaded EfficientNet-B0 model (in eval mode).
        image_tensor : Pre-processed image tensor with shape (1, 3, 224, 224).
        target_class : (Optional) Class index to visualise. If None the
                       predicted class is used.

    Returns:
        heatmap : numpy array (224, 224) with values in [0, 1].
    """

    # --- Determine device from model parameters ---
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # --- Storage for activations and gradients ---
    activations = []
    gradients = []

    # --- Hook the last convolutional layer ---
    # In EfficientNet-B0 this is model.features[-1]  (the final MBConv block)
    target_layer = model.features[-1]

    def forward_hook(module, input, output):
        # Save the feature-map activations
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        # Save the gradients flowing back
        gradients.append(grad_output[0].detach())

    # Register the hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # --- Forward pass ---
    model.eval()
    output = model(image_tensor)

    # Pick the target class (highest score if not specified)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # --- Backward pass on the target class score ---
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # --- Compute the Grad-CAM heatmap ---
    # Global-average-pool the gradients → channel weights
    grad = gradients[0]                       # (1, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Weighted combination of activation maps
    act = activations[0]                      # (1, C, H, W)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)

    # ReLU — keep only positive influence
    cam = torch.relu(cam)

    # Resize to input image size (224×224)
    cam = torch.nn.functional.interpolate(
        cam, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Normalise to [0, 1]
    cam = cam.squeeze().cpu().numpy()
    if cam.max() != 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # --- Clean up hooks ---
    fwd_handle.remove()
    bwd_handle.remove()

    return cam  # numpy array (224, 224)


# ---------------------------------------------------------------------------
# Heatmap overlay helper
# ---------------------------------------------------------------------------

def overlay_heatmap(original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay a Grad-CAM heatmap on the original PIL image.

    Args:
        original_image : PIL.Image (any size).
        heatmap        : numpy array (224, 224) with values in [0, 1].
        alpha          : Blending factor (0 = only image, 1 = only heatmap).
        colormap       : OpenCV colour-map constant.

    Returns:
        result : PIL.Image with the heatmap overlay at the original size.
    """

    # Convert PIL image to numpy RGB
    img = np.array(original_image.convert("RGB"))
    h, w = img.shape[:2]

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to colour (0-255, uint8)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    # OpenCV returns BGR — convert to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image with the heatmap
    blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Standalone quick-test (python gradcam.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("gradcam.py loaded successfully. Import and use generate_gradcam().")
