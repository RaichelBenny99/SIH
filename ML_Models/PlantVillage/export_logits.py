"""
export_logits.py — Export Validation Logits for Temperature Scaling

Runs the trained EfficientNet-B0 over the validation split and saves
the raw logits (before softmax) and ground-truth labels as .npy files.

These files are then used by TemperatureScaler to learn the optimal
temperature T for confidence calibration — without retraining the model.

Usage (standalone):
    python export_logits.py

Usage (from another script / notebook):
    from export_logits import export_validation_logits
    paths = export_validation_logits(model, val_loader, save_dir=".")
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# =====================================================================
# CONFIGURATION  (mirrors plantVillage.py)
# =====================================================================

# IMPORTANT: Change this to YOUR dataset path
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"

MODEL_PATH = "plant_disease_model.pth"
NUM_CLASSES = 39
BATCH_SIZE = 32

# Where to save the exported numpy files
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# CORE EXPORT FUNCTION  (reusable from any script)
# =====================================================================

def export_validation_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = None,
    save_dir: str = ".",
) -> dict:
    """
    Run the model over a validation DataLoader and save raw logits + labels.

    Parameters
    ----------
    model      : nn.Module   — trained model (will be set to eval mode).
    val_loader : DataLoader  — validation data.
    device     : str or None — "cuda" / "cpu"; auto-detected if None.
    save_dir   : str         — directory to save the .npy files.

    Returns
    -------
    dict with keys:
        logits_path : str   — path to saved val_logits.npy  (N, C)
        labels_path : str   — path to saved val_labels.npy  (N,)
        num_samples : int   — total number of validation samples
        accuracy    : float — top-1 accuracy on the validation set
    """

    # ---- Device setup ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ---- Validate loader ----
    if len(val_loader) == 0:
        raise ValueError("Validation DataLoader is empty — nothing to export.")

    # ---- Collect logits and labels ----
    all_logits = []
    all_labels = []
    total_correct = 0
    total_samples = 0

    print(f"Exporting logits on {device.upper()} ...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass — raw logits (NO softmax)
            logits = model(inputs)

            # Track accuracy
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Move to CPU to save memory
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Progress update every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(val_loader)} done")

    elapsed = time.time() - t0

    # ---- Stack into single arrays ----
    logits_np = np.concatenate(all_logits, axis=0)   # (N, C)
    labels_np = np.concatenate(all_labels, axis=0)    # (N,)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # ---- Save ----
    os.makedirs(save_dir, exist_ok=True)
    logits_path = os.path.join(save_dir, "val_logits.npy")
    labels_path = os.path.join(save_dir, "val_labels.npy")

    np.save(logits_path, logits_np)
    np.save(labels_path, labels_np)

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"  Export complete in {elapsed:.1f}s")
    print(f"  Samples  : {total_samples}")
    print(f"  Logits   : {logits_np.shape}  → {logits_path}")
    print(f"  Labels   : {labels_np.shape}  → {labels_path}")
    print(f"  Val Acc  : {accuracy*100:.2f} %")
    print(f"{'='*50}\n")

    return {
        "logits_path": logits_path,
        "labels_path": labels_path,
        "num_samples": total_samples,
        "accuracy": accuracy,
    }


# =====================================================================
# CONVENIENCE: export + calibrate in one step
# =====================================================================

def export_and_calibrate(
    model: nn.Module,
    val_loader: DataLoader,
    save_dir: str = ".",
    device: str = None,
) -> float:
    """
    Export logits, fit temperature, save temperature.pth — all in one call.

    Returns the learned temperature T.
    """
    # Avoid circular import at module level
    from temperature_scaling import TemperatureScaler

    paths = export_validation_logits(model, val_loader, device=device,
                                      save_dir=save_dir)

    logits = torch.from_numpy(np.load(paths["logits_path"])).float()
    labels = torch.from_numpy(np.load(paths["labels_path"])).long()

    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels)

    temp_path = os.path.join(save_dir, "temperature.pth")
    scaler.save(temp_path)
    print(f"Temperature T = {T:.4f}  saved to {temp_path}")

    return T


# =====================================================================
# STANDALONE MODE  (python export_logits.py)
# =====================================================================

if __name__ == "__main__":
    # ---- Reproduce the same val split as plantVillage.py ----
    print("Setting up dataset (must match training split) ...")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Dataset directory not found: {DATA_DIR}")
        print("Please set DATA_DIR at the top of this file to your dataset path.")
        exit(1)

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes, {len(full_dataset)} images total.")

    # Same 80/20 split — use the same seed / logic as plantVillage.py
    # NOTE: random_split uses the default PyTorch RNG. If you used a
    #       fixed seed during training, set the same seed here.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    # ---- Load trained model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Train the model first using plantVillage.py, or set MODEL_PATH.")
        exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded from {MODEL_PATH}")

    # ---- Export + Calibrate ----
    T = export_and_calibrate(model, val_loader, save_dir=SAVE_DIR, device=device)
    print(f"\nDone! Learned temperature: {T:.4f}")
    print(f"Files saved in: {SAVE_DIR}")
    print("  → val_logits.npy")
    print("  → val_labels.npy")
    print("  → temperature.pth")
    print("\nThe Streamlit app will automatically load temperature.pth on next run.")
