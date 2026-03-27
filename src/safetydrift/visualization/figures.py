"""Publication-quality figures for the SafetyDrift paper.

All figures saved as PDF for LaTeX inclusion. Uses matplotlib + seaborn
with serif fonts and 300 DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


RISK_NAMES = ["SAFE", "MILD", "ELEVATED", "CRITICAL", "VIOLATED"]
RISK_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


def plot_transition_heatmap(
    matrix: np.ndarray,
    output_path: str | Path = "results/figures/transition_heatmap.pdf",
) -> Path:
    """5x5 transition matrix heatmap with annotated probabilities."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=RISK_NAMES,
        yticklabels=RISK_NAMES,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Transition probability"},
    )
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Safety state transition probabilities")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_absorption_curves(
    finite_horizon_probs: dict[int, np.ndarray],
    output_path: str | Path = "results/figures/absorption_curves.pdf",
) -> Path:
    """P(VIOLATED within N steps) for each state across horizons."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    horizons = sorted(finite_horizon_probs.keys())

    for state_idx in range(4):  # skip VIOLATED (always 1.0)
        probs = [finite_horizon_probs[h][state_idx] for h in horizons]
        ax.plot(horizons, probs, "o-", label=RISK_NAMES[state_idx],
                color=RISK_COLORS[state_idx], linewidth=2, markersize=6)

    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.85)")
    ax.axhline(y=0.40, color="gray", linestyle=":", alpha=0.5, label="Threshold (0.40)")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("P(VIOLATED within N steps)")
    ax.set_title("Finite-horizon violation probabilities by state")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(horizons) + 0.5)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_per_category_absorption(
    per_category_data: dict[str, dict[int, np.ndarray]],
    output_path: str | Path = "results/figures/per_category_absorption.pdf",
) -> Path:
    """2x2 subplot showing absorption curves per category."""
    _setup_style()
    categories = sorted(per_category_data.keys())
    n_cats = len(categories)
    cols = 2
    rows = (n_cats + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)

    for idx, cat in enumerate(categories):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        fh = per_category_data[cat]
        horizons = sorted(fh.keys())

        for state_idx in range(4):
            probs = [fh[h][state_idx] for h in horizons]
            ax.plot(horizons, probs, "o-", label=RISK_NAMES[state_idx],
                    color=RISK_COLORS[state_idx], linewidth=1.5, markersize=4)

        ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(cat.replace("_", " ").title())
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("P(VIOLATED)")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused subplots
    for idx in range(n_cats, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_monitor_roc(
    roc_curves: dict[str, list[dict]],
    baseline_points: list[dict],
    output_path: str | Path = "results/figures/monitor_roc.pdf",
) -> Path:
    """Detection rate vs FPR for all baselines + Markov sweep."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # ROC curves (Markov monitor threshold sweep)
    for name, points in roc_curves.items():
        fprs = [p["false_positive_rate"] for p in points]
        drs = [p["detection_rate"] for p in points]
        ax.plot(fprs, drs, "o-", label=name, linewidth=2, markersize=4)

    # Baseline points (single dot each)
    markers = ["s", "^", "D"]
    for i, bp in enumerate(baseline_points):
        ax.scatter(bp["fpr"], bp["detection"], s=100, marker=markers[i % len(markers)],
                   label=bp["name"], zorder=5, edgecolors="black", linewidth=0.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Detection rate")
    ax.set_title("Monitor comparison: detection vs. false positive tradeoff")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_early_warning_distribution(
    early_warnings: list[int],
    output_path: str | Path = "results/figures/early_warning_dist.pdf",
) -> Path:
    """Histogram of early warning steps."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    if early_warnings:
        ax.hist(early_warnings, bins=range(0, max(early_warnings) + 2),
                color=RISK_COLORS[1], edgecolor="black", alpha=0.8)
        mean_ew = np.mean(early_warnings)
        ax.axvline(mean_ew, color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_ew:.1f} steps")
        ax.legend()

    ax.set_xlabel("Steps of early warning")
    ax.set_ylabel("Count")
    ax.set_title("Early warning distribution (Markov monitor)")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_ablation_bars(
    ablation_data: list[dict],
    output_path: str | Path = "results/figures/ablation_bars.pdf",
) -> Path:
    """Bar chart showing accuracy when removing each dimension."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = []
    accuracies = []
    colors = []
    for item in ablation_data:
        removed = item.get("removed")
        labels.append("Full model" if removed is None else f"- {removed}")
        accuracies.append(item["accuracy"])
        colors.append("#3498db" if removed is None else "#e74c3c")

    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Prediction accuracy")
    ax.set_title("Dimension ablation study")
    ax.set_ylim(0, 1.0)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_learning_curve(
    learning_data: list[dict],
    output_path: str | Path = "results/figures/learning_curve.pdf",
) -> Path:
    """Accuracy vs number of training traces with error bands."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    n_traces = [p["num_traces"] for p in learning_data]
    accs = [p["accuracy"] for p in learning_data]
    stds = [p.get("accuracy_std", 0) for p in learning_data]

    ax.plot(n_traces, accs, "o-", color="#3498db", linewidth=2, markersize=6)
    ax.fill_between(n_traces,
                     [a - s for a, s in zip(accs, stds)],
                     [a + s for a, s in zip(accs, stds)],
                     alpha=0.2, color="#3498db")

    ax.set_xlabel("Number of training traces")
    ax.set_ylabel("Prediction accuracy")
    ax.set_title("Learning curve: accuracy vs. training data")
    ax.set_ylim(0, 1.0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path
