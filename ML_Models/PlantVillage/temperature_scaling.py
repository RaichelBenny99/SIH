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
# Reliability Diagram  (paper-quality, with gap visualisation)
# =====================================================================


def _draw_reliability_on_ax(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    title: str,
    ece: float,
    ax: "matplotlib.axes.Axes",
) -> None:
    """
    Draw a single reliability diagram onto an existing Axes.

    Blue bars  — fraction correct in each confidence bin.
    Red fill   — over-confidence gap  (confidence > accuracy).
    Green fill — under-confidence gap (accuracy  > confidence).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bar_width = (bin_boundaries[1] - bin_boundaries[0]) * 0.90

    bin_centres, bin_accs, bin_confs = [], [], []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centres.append((lo + hi) / 2)
        bin_accs.append(accuracies[mask].mean())
        bin_confs.append(confidences[mask].mean())

    bin_centres = np.array(bin_centres)
    bin_accs    = np.array(bin_accs)
    bin_confs   = np.array(bin_confs)

    # Accuracy bars (blue)
    ax.bar(bin_centres, bin_accs, width=bar_width,
           color="#4C8CBF", alpha=0.85, edgecolor="white",
           linewidth=0.6, label="Accuracy", zorder=3)

    # Over-confidence gap (red — confidence exceeds accuracy)
    over = bin_confs > bin_accs
    if over.any():
        ax.bar(bin_centres[over],
               bin_confs[over] - bin_accs[over],
               width=bar_width, bottom=bin_accs[over],
               color="#E74C3C", alpha=0.55, edgecolor="white",
               linewidth=0.4, label="Over-confidence gap", zorder=4)

    # Under-confidence gap (green — accuracy exceeds confidence)
    under = bin_accs > bin_confs
    if under.any():
        ax.bar(bin_centres[under],
               bin_accs[under] - bin_confs[under],
               width=bar_width, bottom=bin_confs[under],
               color="#27AE60", alpha=0.55, edgecolor="white",
               linewidth=0.4, label="Under-confidence gap", zorder=4)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.75,
            label="Perfect calibration", zorder=5)

    # ECE annotation box
    ax.text(0.05, 0.93, f"ECE = {ece * 100:.2f}%",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.30", fc="white",
                      ec="#888", alpha=0.90))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence", fontsize=10)
    ax.set_ylabel("Fraction correct (accuracy)", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.30, zorder=0)

    # Deduplicated legend
    handles, lbls = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, lbl in zip(handles, lbls):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(list(seen.values()), list(seen.keys()),
              loc="lower right", fontsize=8)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: str = None,
) -> "matplotlib.figure.Figure | None":
    """
    Plot a reliability diagram with gap visualisation and sample-count bars.

    Parameters
    ----------
    probs     : numpy array (N, C).
    labels    : numpy array (N,).
    n_bins    : int.
    title     : str — figure title.
    save_path : str or None — if given, save the figure to this path.

    Returns
    -------
    matplotlib Figure, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    ece = compute_ece(probs, labels, n_bins)

    # Sample counts per bin (for the counts subplot)
    confidences = probs.max(axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bar_width = (bin_boundaries[1] - bin_boundaries[0]) * 0.90
    bin_centres_all = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2
                       for i in range(n_bins)]
    bin_counts = [
        int(((confidences > bin_boundaries[i]) &
             (confidences <= bin_boundaries[i + 1])).sum())
        for i in range(n_bins)
    ]

    fig, (ax_main, ax_cnt) = plt.subplots(
        2, 1, figsize=(5.5, 6.5),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.10},
    )

    _draw_reliability_on_ax(probs, labels, n_bins,
                             title=title, ece=ece, ax=ax_main)
    ax_main.set_xlabel("")   # x-label lives on the counts subplot

    # Sample-count subplot
    ax_cnt.bar(bin_centres_all, bin_counts, width=bar_width,
               color="#4C8CBF", alpha=0.65, edgecolor="white", linewidth=0.5)
    ax_cnt.set_xlim(0, 1)
    ax_cnt.set_xlabel("Mean predicted confidence", fontsize=10)
    ax_cnt.set_ylabel("Count", fontsize=9)
    ax_cnt.tick_params(axis="y", labelsize=8)
    for sp in ["top", "right"]:
        ax_cnt.spines[sp].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reliability_comparison(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    T: float = None,
    save_path: str = None,
) -> "matplotlib.figure.Figure | None":
    """
    Side-by-side reliability diagram: before vs. after temperature scaling.

    This is the standard calibration-proof figure for papers.

    Parameters
    ----------
    probs_before : numpy array (N, C) — raw softmax probabilities (T = 1.0).
    probs_after  : numpy array (N, C) — calibrated probabilities after scaling.
    labels       : numpy array (N,)   — ground-truth class indices.
    n_bins       : int                — number of confidence bins.
    T            : float or None      — learned temperature (shown in title).
    save_path    : str or None        — path to save the figure.

    Returns
    -------
    matplotlib Figure, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    ece_before = compute_ece(probs_before, labels, n_bins)
    ece_after  = compute_ece(probs_after,  labels, n_bins)
    delta_ece  = ece_before - ece_after   # positive = improvement

    t_str = f"  |  T = {T:.4f}" if T is not None else ""

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))

    _draw_reliability_on_ax(
        probs_before, labels, n_bins,
        title="Before Calibration\n(T = 1.0, raw softmax)",
        ece=ece_before, ax=ax_l,
    )
    _draw_reliability_on_ax(
        probs_after, labels, n_bins,
        title=(
            f"After Temperature Scaling\n(T = {T:.4f})"
            if T is not None else "After Temperature Scaling"
        ),
        ece=ece_after, ax=ax_r,
    )

    fig.suptitle(
        "Calibration Reliability Diagram — PlantVillage EfficientNet-B0\n"
        f"ECE:  {ece_before * 100:.2f}% → {ece_after * 100:.2f}%"
        f"  (Δ = −{delta_ece * 100:.2f}%){t_str}",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

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
