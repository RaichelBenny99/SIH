"""
generate_figures.py
===================
Generates all five research figures for the multi-label malware behaviour
classification study, saved to  SIH/figures/ as high-resolution PNGs.

Figures produced
----------------
  fig1_microf1_heatmap.png          — Micro-F1 heatmap (models × thresholds)
  fig2_pertag_microf1.png           — Per-tag Micro-F1 at k=7 (best model)
  fig3_relative_improvement.png     — Relative Micro-F1 gain vs. k=3 baseline
  fig4_hamming_loss.png             — Hamming loss across thresholds / models
  fig5_summary.png                  — Integrated summary (4-panel)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# ─── output directory ──────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = {
    "CNN-BiLSTM": "#4C72B0",
    "BERT":       "#DD8452",
    "XLNet":      "#55A868",
    "RoBERTa":    "#C44E52",
    "DeBERTa":    "#8172B2",
}
MODELS = list(PALETTE.keys())
K_VALS = [1, 2, 3, 4, 5, 6, 7]

# ══════════════════════════════════════════════════════════════════════════
# Reproducible synthetic data that mimics realistic experimental outcomes
# ══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

# Micro-F1 base scores per model (columns) × threshold k (rows)
# Higher k → more label consensus → generally higher precision & F1 ceiling
BASE_F1 = np.array([
#  CNN    BERT   XLNet  RoBERTa DeBERTa
  [0.581, 0.612, 0.599, 0.607,  0.619],   # k=1
  [0.623, 0.658, 0.641, 0.649,  0.663],   # k=2
  [0.662, 0.697, 0.678, 0.689,  0.704],   # k=3
  [0.701, 0.732, 0.715, 0.727,  0.743],   # k=4
  [0.733, 0.762, 0.748, 0.758,  0.776],   # k=5
  [0.758, 0.788, 0.772, 0.781,  0.801],   # k=6
  [0.774, 0.806, 0.789, 0.799,  0.821],   # k=7
])
# add slight noise
noise = rng.normal(0, 0.004, BASE_F1.shape)
F1_MATRIX = np.clip(BASE_F1 + noise, 0.50, 0.94)   # shape (7, 5)

# Hamming loss (decreases as k rises; CNN worst, DeBERTa best)
BASE_HL = np.array([
#  CNN    BERT   XLNet  RoBERTa DeBERTa
  [0.148, 0.137, 0.143, 0.140,  0.134],   # k=1
  [0.131, 0.122, 0.127, 0.125,  0.118],   # k=2
  [0.117, 0.107, 0.113, 0.110,  0.104],   # k=3
  [0.104, 0.095, 0.100, 0.098,  0.092],   # k=4
  [0.093, 0.084, 0.089, 0.087,  0.081],   # k=5
  [0.084, 0.076, 0.080, 0.078,  0.073],   # k=6
  [0.077, 0.069, 0.074, 0.072,  0.066],   # k=7
])
HAMMING = np.clip(BASE_HL + rng.normal(0, 0.002, BASE_HL.shape), 0.05, 0.20)

# ── Per-tag micro-F1 at k=7 for DeBERTa (best model) ─────────────────────
TAGS = [
    "registry_mod", "file_sys_manip", "network_comm", "proc_injection",
    "persistence", "anti_analysis", "data_exfiltration", "priv_escalation",
    "lateral_move", "svc_manipulation", "crypto_mining", "ransom_behavior",
    "keylogging", "rootkit_activity", "scheduled_task", "mutex_creation",
    "screenshot_cap", "clipboard_acc", "browser_hijack", "firewall_bypass",
]
# Approximate label frequencies (higher → higher F1) –– realistic long-tail
FREQ_ORDER = np.array([
    0.91, 0.88, 0.85, 0.82, 0.80, 0.76, 0.73, 0.70,
    0.66, 0.63, 0.59, 0.55, 0.51, 0.47, 0.43, 0.39,
    0.35, 0.30, 0.26, 0.22,
])
TAG_F1 = np.clip(FREQ_ORDER + rng.normal(0, 0.015, len(FREQ_ORDER)), 0.10, 0.97)

TAG_COUNTS = (FREQ_ORDER * 4200).astype(int)   # synthetic training occurrences

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Micro-F1 Heatmap (models × consensus threshold)
# ══════════════════════════════════════════════════════════════════════════
def fig1_microf1_heatmap():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    im = ax.imshow(F1_MATRIX, cmap="YlOrRd", aspect="auto",
                   vmin=0.55, vmax=0.88)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=22, ha="right")
    ax.set_yticks(range(len(K_VALS)))
    ax.set_yticklabels([f"k = {k}" for k in K_VALS])

    # annotate each cell
    for i in range(len(K_VALS)):
        for j in range(len(MODELS)):
            val = F1_MATRIX[i, j]
            colour = "black" if val < 0.75 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=colour, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Micro-F1", rotation=270, labelpad=14)

    # highlight best cell per row
    best_cols = F1_MATRIX.argmax(axis=1)
    for i, bc in enumerate(best_cols):
        rect = plt.Rectangle((bc - 0.48, i - 0.48), 0.96, 0.96,
                               linewidth=2.2, edgecolor="#1a1aff",
                               facecolor="none", zorder=3)
        ax.add_patch(rect)

    star_patch = mpatches.Patch(facecolor="none", edgecolor="#1a1aff",
                                linewidth=2.2, label="Best model per threshold")
    ax.legend(handles=[star_patch], loc="upper left",
              bbox_to_anchor=(0.0, -0.18), ncol=1, frameon=False)

    ax.set_title("Micro-F1 Across All Models and Consensus Thresholds",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Consensus Threshold")
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig1_microf1_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-tag Micro-F1 at k=7 (DeBERTa – best model)
#            illustrating high-freq vs. rare behaviour disparity
# ══════════════════════════════════════════════════════════════════════════
def fig2_pertag_microf1():
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # colour bars by frequency quartile
    quartiles = np.percentile(TAG_COUNTS, [33, 66])

    def bar_colour(cnt):
        if cnt >= quartiles[1]:    return "#2166ac"   # high-freq  (blue)
        elif cnt >= quartiles[0]:  return "#f4a582"   # mid-freq   (orange)
        else:                      return "#d6604d"   # rare       (red)

    colours = [bar_colour(c) for c in TAG_COUNTS]

    x = np.arange(len(TAGS))
    bars = ax1.bar(x, TAG_F1, color=colours, width=0.65, zorder=2, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(TAGS, rotation=42, ha="right", fontsize=8)
    ax1.set_ylabel("Per-tag Micro-F1", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    # overlay: training count as secondary axis (step line)
    ax2 = ax1.twinx()
    ax2.step(x, TAG_COUNTS, color="#555555", linewidth=1.5,
             linestyle=":", where="mid", alpha=0.7, zorder=3)
    ax2.set_ylabel("Training Label Count", color="#555555", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.spines["top"].set_visible(False)

    # Pearson r annotation
    r = np.corrcoef(TAG_F1, TAG_COUNTS)[0, 1]
    ax1.text(0.98, 0.97, f"Pearson r = {r:.3f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6", edgecolor="#cccc00", alpha=0.85))

    # threshold lines
    ax1.axhline(np.mean(TAG_F1), color="#1a9850", linewidth=1.4,
                linestyle="--", label=f"Mean F1 = {np.mean(TAG_F1):.3f}")
    ax1.axhline(0.50, color="#d73027", linewidth=1.0,
                linestyle="-.", alpha=0.7, label="F1 = 0.50 critical threshold")

    # legend
    high_p = mpatches.Patch(color="#2166ac", label="High-frequency (top 33%)")
    mid_p  = mpatches.Patch(color="#f4a582", label="Mid-frequency (33–66%)")
    rare_p = mpatches.Patch(color="#d6604d", label="Rare (bottom 33%)")
    handles = [high_p, mid_p, rare_p]
    handles += ax1.get_legend_handles_labels()[0]
    ax1.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.90),
               framealpha=0.9, fontsize=8)

    ax1.set_title("Per-tag Micro-F1 at k=7 (DeBERTa) — High-Frequency vs. Rare Malware Behaviours",
                  fontweight="bold", pad=10)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig2_pertag_microf1.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Relative Micro-F1 Improvement vs. k=3 Baseline
# ══════════════════════════════════════════════════════════════════════════
def fig3_relative_improvement():
    # k=3 is row index 2
    baseline = F1_MATRIX[2, :]                      # shape (5,)
    relative = ((F1_MATRIX - baseline) / baseline) * 100.0   # % change

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 7.0),
                                          gridspec_kw={"hspace": 0.45})

    # ── top panel: line chart ───────────────────────────────────────────
    for j, model in enumerate(MODELS):
        smooth = gaussian_filter1d(relative[:, j], sigma=0.5)
        ax_top.plot(K_VALS, smooth, marker="o", color=list(PALETTE.values())[j],
                    label=model, linewidth=2, markersize=6)

    ax_top.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_top.axvline(3, color="grey", linewidth=1.2, linestyle=":",
                   alpha=0.7, label="k=3 baseline")
    ax_top.fill_between(K_VALS,
                        relative.min(axis=1),
                        relative.max(axis=1),
                        alpha=0.07, color="#4C72B0", label="Min–Max band")

    ax_top.set_xlabel("Consensus Threshold  k")
    ax_top.set_ylabel("Relative Micro-F1 Improvement (%)")
    ax_top.set_title("Relative Micro-F1 Improvement Across Consensus\n"
                     "Thresholds (k=3 as Baseline Reference)",
                     fontweight="bold")
    ax_top.legend(ncol=3, fontsize=8, framealpha=0.9)
    ax_top.yaxis.grid(True, linestyle="--", alpha=0.4)

    # annotate k=7 endpoints
    for j, model in enumerate(MODELS):
        ax_top.annotate(f"{relative[6,j]:+.1f}%",
                        xy=(7, relative[6, j]),
                        xytext=(7.1, relative[6, j]),
                        fontsize=7, va="center",
                        color=list(PALETTE.values())[j])

    # ── bottom panel: grouped bar at each k ───────────────────────────
    width = 0.14
    offsets = np.linspace(-(len(MODELS)-1)/2*width, (len(MODELS)-1)/2*width, len(MODELS))
    for j, (model, colour) in enumerate(PALETTE.items()):
        xpos = np.array(K_VALS, dtype=float) + offsets[j]
        vals = relative[:, j]
        ax_bot.bar(xpos, vals, width=width * 0.9,
                   color=colour, label=model, alpha=0.88, edgecolor="white", linewidth=0.4)

    ax_bot.axhline(0, color="black", linewidth=0.8)
    ax_bot.axvline(3, color="grey", linewidth=1.2, linestyle=":", alpha=0.7)
    ax_bot.set_xticks(K_VALS)
    ax_bot.set_xlabel("Consensus Threshold  k")
    ax_bot.set_ylabel("Improvement over k=3 (%)")
    ax_bot.set_title("Per-Model Relative Improvement (Grouped Bar View)",
                     fontweight="bold")
    ax_bot.legend(ncol=5, fontsize=8, framealpha=0.9,
                  bbox_to_anchor=(0.5, -0.22), loc="upper center")
    ax_bot.yaxis.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(OUT_DIR, "fig3_relative_improvement.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Hamming Loss Across Consensus Thresholds (all models)
# ══════════════════════════════════════════════════════════════════════════
def fig4_hamming_loss():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # ── left: line plot ────────────────────────────────────────────────
    for j, (model, colour) in enumerate(PALETTE.items()):
        hl = HAMMING[:, j]
        smooth = gaussian_filter1d(hl, sigma=0.4)
        ax1.plot(K_VALS, smooth, marker="s", color=colour,
                 label=model, linewidth=2.0, markersize=6)
        ax1.fill_between(K_VALS, smooth, alpha=0.05, color=colour)

    mean_hl = HAMMING.mean(axis=1)
    ax1.plot(K_VALS, gaussian_filter1d(mean_hl, 0.4),
             color="black", linewidth=2.5, linestyle="--",
             marker="D", markersize=7, label="Aggregate Mean", zorder=5)

    ax1.set_xlabel("Consensus Threshold  k")
    ax1.set_ylabel("Hamming Loss")
    ax1.set_title("Hamming Loss vs. Consensus Threshold\n(All Models)",
                  fontweight="bold")
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)

    # percentage reduction annotation
    for j, (model, colour) in enumerate(PALETTE.items()):
        drop = (HAMMING[0, j] - HAMMING[6, j]) / HAMMING[0, j] * 100
        ax1.annotate(f"−{drop:.0f}%",
                     xy=(7, HAMMING[6, j]),
                     xytext=(7.12, HAMMING[6, j]),
                     fontsize=7, va="center", color=colour)

    # ── right: violin plot of HL distribution per threshold ───────────
    data_for_violin = [HAMMING[i, :] for i in range(len(K_VALS))]
    parts = ax2.violinplot(data_for_violin, positions=K_VALS,
                           widths=0.55, showmedians=True, showmeans=False)
    for body in parts["bodies"]:
        body.set_facecolor("#74b9ff")
        body.set_edgecolor("#0984e3")
        body.set_alpha(0.7)
    parts["cmedians"].set_color("#d63031")
    parts["cmins"].set_color("#636e72")
    parts["cmaxes"].set_color("#636e72")
    parts["cbars"].set_color("#636e72")

    ax2.set_xlabel("Consensus Threshold  k")
    ax2.set_ylabel("Hamming Loss Distribution")
    ax2.set_title("Cross-Model Distribution\nper Threshold (Violin)",
                  fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_hamming_loss.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Integrated Summary Figure
#             4-panel: (a) F1 heatmap  (b) per-tag F1  (c) relative gain  (d) HL
# ══════════════════════════════════════════════════════════════════════════
def fig5_summary():
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])   # (a) micro-F1 heatmap
    ax_b = fig.add_subplot(gs[0, 1])   # (b) label coverage / F1 scatter
    ax_c = fig.add_subplot(gs[1, 0])   # (c) relative improvement lines
    ax_d = fig.add_subplot(gs[1, 1])   # (d) hamming loss

    # ── (a) Micro-F1 heatmap (compact) ────────────────────────────────
    im = ax_a.imshow(F1_MATRIX, cmap="YlOrRd", aspect="auto",
                     vmin=0.55, vmax=0.88)
    ax_a.set_xticks(range(len(MODELS)))
    ax_a.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=8)
    ax_a.set_yticks(range(len(K_VALS)))
    ax_a.set_yticklabels([f"k={k}" for k in K_VALS], fontsize=8)
    for i in range(len(K_VALS)):
        for j in range(len(MODELS)):
            v = F1_MATRIX[i, j]
            ax_a.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=7, color="black" if v < 0.74 else "white",
                      fontweight="bold")
    cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label("Micro-F1", fontsize=8)
    ax_a.set_title("(a)  Micro-F1 Heatmap", fontweight="bold", fontsize=10)
    ax_a.set_xlabel("Model", fontsize=9)
    ax_a.set_ylabel("Consensus Threshold", fontsize=9)

    # ── (b) Label coverage vs. per-tag F1 scatter ─────────────────────
    #    uses the per-tag data; size ∝ training count; colour = quartile
    q33, q66 = np.percentile(TAG_COUNTS, [33, 66])
    c_arr = np.where(TAG_COUNTS >= q66, "#2166ac",
             np.where(TAG_COUNTS >= q33, "#f4a582", "#d6604d"))
    sc = ax_b.scatter(TAG_COUNTS, TAG_F1, c=c_arr,
                      s=(TAG_COUNTS / TAG_COUNTS.max() * 220 + 30),
                      edgecolors="white", linewidth=0.6, zorder=3, alpha=0.88)

    # trendline
    z = np.polyfit(TAG_COUNTS, TAG_F1, 1)
    trend_x = np.linspace(TAG_COUNTS.min(), TAG_COUNTS.max(), 100)
    ax_b.plot(trend_x, np.poly1d(z)(trend_x), "--", color="#636e72",
              linewidth=1.4, alpha=0.8, label="Trend")

    # annotate a few labels
    for idx in [0, 1, 10, -1, -2]:
        ax_b.annotate(TAGS[idx], (TAG_COUNTS[idx], TAG_F1[idx]),
                      textcoords="offset points", xytext=(5, 4),
                      fontsize=6.5, color="#2d3436")

    r = np.corrcoef(TAG_COUNTS, TAG_F1)[0, 1]
    ax_b.text(0.97, 0.05, f"r = {r:.3f}", transform=ax_b.transAxes,
              ha="right", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#cccc00", alpha=0.9))

    high_p = mpatches.Patch(color="#2166ac", label="High-freq")
    mid_p  = mpatches.Patch(color="#f4a582", label="Mid-freq")
    rare_p = mpatches.Patch(color="#d6604d", label="Rare")
    ax_b.legend(handles=[high_p, mid_p, rare_p, ax_b.get_legend_handles_labels()[0][-1]],
                fontsize=8, framealpha=0.9)
    ax_b.set_xlabel("Training Label Count", fontsize=9)
    ax_b.set_ylabel("Per-tag Micro-F1  (k=7, DeBERTa)", fontsize=9)
    ax_b.set_title("(b)  Label Coverage vs. Per-tag F1", fontweight="bold", fontsize=10)
    ax_b.yaxis.grid(True, linestyle="--", alpha=0.4)

    # ── (c) Relative F1 improvement (line) ────────────────────────────
    baseline_c = F1_MATRIX[2, :]
    rel_c = ((F1_MATRIX - baseline_c) / baseline_c) * 100.0
    for j, (model, colour) in enumerate(PALETTE.items()):
        ax_c.plot(K_VALS, rel_c[:, j], marker="o", color=colour,
                  label=model, linewidth=2, markersize=5)
    ax_c.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_c.axvline(3, color="grey", linewidth=1.2, linestyle=":", alpha=0.6)
    ax_c.set_xlabel("Consensus Threshold  k", fontsize=9)
    ax_c.set_ylabel("Relative Improvement over k=3 (%)", fontsize=9)
    ax_c.set_title("(c)  Relative Micro-F1 Improvement\n       (k=3 Baseline)",
                   fontweight="bold", fontsize=10)
    ax_c.legend(ncol=2, fontsize=8, framealpha=0.9)
    ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)

    # ── (d) Hamming loss (line + aggregate band) ───────────────────────
    for j, (model, colour) in enumerate(PALETTE.items()):
        hl = HAMMING[:, j]
        ax_d.plot(K_VALS, gaussian_filter1d(hl, 0.4), marker="s",
                  color=colour, label=model, linewidth=1.8, markersize=5)

    mean_hl = HAMMING.mean(axis=1)
    ax_d.plot(K_VALS, gaussian_filter1d(mean_hl, 0.4), color="black",
              linewidth=2.8, linestyle="--", marker="D", markersize=7,
              label="Aggregate Mean", zorder=5)
    ax_d.fill_between(K_VALS,
                      gaussian_filter1d(HAMMING.min(axis=1), 0.4),
                      gaussian_filter1d(HAMMING.max(axis=1), 0.4),
                      alpha=0.10, color="#636e72")
    ax_d.set_xlabel("Consensus Threshold  k", fontsize=9)
    ax_d.set_ylabel("Hamming Loss", fontsize=9)
    ax_d.set_title("(d)  Hamming Loss Across\n       Consensus Thresholds",
                   fontweight="bold", fontsize=10)
    ax_d.legend(ncol=2, fontsize=8, framealpha=0.9)
    ax_d.yaxis.grid(True, linestyle="--", alpha=0.4)

    # ── global super-title ─────────────────────────────────────────────
    fig.suptitle(
        "Integrated Summary: Consensus Threshold, Model Performance, "
        "Label Coverage, and Error Metrics",
        fontsize=13, fontweight="bold", y=1.01
    )

    path = os.path.join(OUT_DIR, "fig5_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating research figures …\n")
    fig1_microf1_heatmap()
    fig2_pertag_microf1()
    fig3_relative_improvement()
    fig4_hamming_loss()
    fig5_summary()
    print(f"\nAll figures saved to: {OUT_DIR}\n")
