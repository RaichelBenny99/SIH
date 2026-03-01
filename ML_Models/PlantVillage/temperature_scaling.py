"""
temperature_scaling.py — Confidence Calibration via Temperature Scaling

Temperature scaling is a simple post-hoc calibration technique that
learns a single scalar *temperature* T on a held-out validation set.
During inference the raw logits are divided by T before applying
softmax, producing better-calibrated probabilities WITHOUT retraining
the model.

Key components
--------------
    TemperatureScaler  — class that learns T and applies calibrated softmax
    compute_ece        — Expected Calibration Error metric
    plot_reliability   — reliability diagram (saved to file or returned)

Usage:
    from temperature_scaling import TemperatureScaler, compute_ece

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)          # one-time calibration
    cal_probs = scaler.calibrated_softmax(logits)  # inference
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# matplotlib is optional (only needed for reliability diagram)
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =====================================================================
# TemperatureScaler
# =====================================================================

class TemperatureScaler:
    """
    Learn and apply temperature scaling.

    Attributes
    ----------
    temperature : float — learned temperature (default 1.0 = uncalibrated).
    fitted      : bool  — True after ``fit()`` has been called.
    """

    def __init__(self, init_temp: float = 1.5, lr: float = 0.01,
                 max_iter: int = 200):
        """
        Parameters
        ----------
        init_temp : float — initial temperature guess (> 0).
        lr        : float — learning rate for LBFGS / gradient descent.
        max_iter  : int   — maximum optimisation steps.
        """
        self.init_temp = init_temp
        self.lr = lr
        self.max_iter = max_iter

        # Will be set after fit()
        self.temperature = 1.0
        self.fitted = False

    # -----------------------------------------------------------------
    # Fitting  (learn temperature from validation logits)
    # -----------------------------------------------------------------

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Learn the optimal temperature on validation data.

        Parameters
        ----------
        logits : Tensor of shape (N, C) — raw model outputs (before softmax).
        labels : Tensor of shape (N,)   — ground-truth class indices.

        Returns
        -------
        temperature : float — the learned temperature.
        """
        # Ensure tensors are on CPU (calibration is lightweight)
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()

        # Learnable temperature parameter (log-space to keep > 0)
        log_temp = nn.Parameter(torch.log(torch.tensor(self.init_temp)))

        # Use LBFGS optimiser — works well for single-parameter problems
        optimiser = torch.optim.LBFGS([log_temp], lr=self.lr,
                                       max_iter=self.max_iter)

        def closure():
            optimiser.zero_grad()
            temp = torch.exp(log_temp)              # ensure T > 0
            scaled = logits / temp
            loss = F.cross_entropy(scaled, labels)   # NLL loss
            loss.backward()
            return loss

        optimiser.step(closure)

        self.temperature = float(torch.exp(log_temp).item())
        self.fitted = True
        return self.temperature

    # -----------------------------------------------------------------
    # Inference  (apply calibrated softmax)
    # -----------------------------------------------------------------

    def calibrated_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature-scaled softmax to raw logits.

        Parameters
        ----------
        logits : Tensor of shape (N, C) or (C,).

        Returns
        -------
        Tensor of calibrated probabilities, same shape as input.
        """
        temp = self.temperature if self.fitted else 1.0
        return F.softmax(logits / temp, dim=-1)

    # -----------------------------------------------------------------
    # Persistence  (save / load the learned temperature)
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the learned temperature to a small file."""
        torch.save({"temperature": self.temperature, "fitted": self.fitted},
                    path)

    def load(self, path: str) -> None:
        """Load a previously saved temperature."""
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.temperature = float(data["temperature"])
            self.fitted = bool(data["fitted"])


# =====================================================================
# Expected Calibration Error  (ECE)
# =====================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Compute the Expected Calibration Error.

    Parameters
    ----------
    probs  : numpy array (N, C) — predicted probabilities.
    labels : numpy array (N,)   — ground-truth class indices.
    n_bins : int — number of equally-spaced confidence bins.

    Returns
    -------
    ece : float — weighted average of |accuracy − confidence| per bin.
    """
    # Top-1 confidence and predicted class
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = accuracies[in_bin].mean()
        ece += (count / len(labels)) * abs(avg_acc - avg_conf)

    return float(ece)


# =====================================================================
# Reliability Diagram  (optional visualisation)
# =====================================================================

def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: str = None,
) -> "matplotlib.figure.Figure | None":
    """
    Plot a reliability diagram (calibration curve).

    Parameters
    ----------
    probs     : numpy array (N, C).
    labels    : numpy array (N,).
    n_bins    : int.
    save_path : str or None — if given, save the figure to this path.

    Returns
    -------
    matplotlib Figure, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centres = []
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_centres.append((lo + hi) / 2)
        bin_accs.append(accuracies[in_bin].mean())
        bin_confs.append(confidences[in_bin].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_centres, bin_accs, width=1.0 / n_bins, alpha=0.6,
           edgecolor="black", label="Accuracy")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)

    return fig


# =====================================================================
# Convenience: fit from a saved-logits file
# =====================================================================

def fit_temperature_from_file(logits_path: str, labels_path: str,
                               save_path: str = None) -> TemperatureScaler:
    """
    Utility to calibrate from pre-saved numpy files.

    Files should contain:
        logits : numpy (N, C)
        labels : numpy (N,)

    Returns a fitted TemperatureScaler.
    """
    logits = torch.from_numpy(np.load(logits_path)).float()
    labels = torch.from_numpy(np.load(labels_path)).long()

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    if save_path:
        scaler.save(save_path)
        print(f"Temperature saved to {save_path}  (T = {scaler.temperature:.4f})")

    return scaler


# =====================================================================
# Quick self-test
# =====================================================================

if __name__ == "__main__":
    # Synthetic data: 500 samples, 10 classes
    np.random.seed(42)
    N, C = 500, 10
    fake_logits = torch.randn(N, C)
    fake_labels = torch.randint(0, C, (N,))

    scaler = TemperatureScaler()
    T = scaler.fit(fake_logits, fake_labels)
    print(f"Learned temperature: {T:.4f}")

    cal_probs = scaler.calibrated_softmax(fake_logits).numpy()
    ece_val = compute_ece(cal_probs, fake_labels.numpy())
    print(f"Calibrated ECE: {ece_val:.4f}")

    fig = plot_reliability_diagram(cal_probs, fake_labels.numpy(),
                                    save_path="reliability_diagram.png")
    if fig:
        print("Reliability diagram saved to reliability_diagram.png")
