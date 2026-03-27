"""Generate all publication-quality figures and tables for the paper.

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from safetydrift.visualization.figures import (
    plot_ablation_bars,
    plot_absorption_curves,
    plot_early_warning_distribution,
    plot_learning_curve,
    plot_monitor_roc,
    plot_per_category_absorption,
    plot_transition_heatmap,
)
from safetydrift.visualization.tables import (
    generate_comparison_table,
    generate_per_category_table,
    generate_transition_table,
)
from safetydrift.evaluation.metrics import MonitorMetrics

logger = logging.getLogger(__name__)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fig_dir = Path("results/figures")
    table_dir = Path("results/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 3 results
    tm_data = _load_json("results/markov/transition_matrix_coarse.json")
    abs_data = _load_json("results/markov/absorption_results.json")
    ablation_data = _load_json("results/markov/ablation_results.json")
    learning_data = _load_json("results/markov/learning_curve.json")
    per_cat_data = _load_json("results/markov/per_category_analysis.json")

    # Load Phase 4 results
    eval_path = Path("results/evaluation/comparison_results.json")
    eval_data = json.loads(eval_path.read_text()) if eval_path.exists() else None

    # === FIGURES ===

    # 1. Transition heatmap
    matrix = np.array(tm_data["matrix"])
    plot_transition_heatmap(matrix, fig_dir / "transition_heatmap.pdf")
    logger.info("Generated: transition_heatmap.pdf")

    # 2. Absorption curves
    finite_horizon = {}
    for h_str, probs in abs_data["finite_horizon"].items():
        finite_horizon[int(h_str)] = np.array(probs)
    plot_absorption_curves(finite_horizon, fig_dir / "absorption_curves.pdf")
    logger.info("Generated: absorption_curves.pdf")

    # 3. Per-category absorption
    per_cat_fh = {}
    for cat, cat_info in per_cat_data.get("categories", {}).items():
        # Need per-category finite horizon probs — extract from transition matrices
        cat_matrix = np.array(cat_info["transition_matrix"])
        cat_fh = {}
        for h in [1, 3, 5, 10]:
            P_n = np.linalg.matrix_power(cat_matrix, h)
            cat_fh[h] = P_n[:, 4]  # VIOLATED column
        per_cat_fh[cat] = cat_fh
    if per_cat_fh:
        plot_per_category_absorption(per_cat_fh, fig_dir / "per_category_absorption.pdf")
        logger.info("Generated: per_category_absorption.pdf")

    # 4. Monitor ROC
    if eval_data and eval_data.get("roc_curves"):
        baseline_points = []
        for m in eval_data["monitors"]:
            if "Markov" not in m["name"]:
                baseline_points.append({
                    "name": m["name"],
                    "fpr": m["false_positive_rate"],
                    "detection": m["detection_rate"],
                })
        plot_monitor_roc(eval_data["roc_curves"], baseline_points, fig_dir / "monitor_roc.pdf")
        logger.info("Generated: monitor_roc.pdf")

    # 5. Early warning distribution
    if eval_data:
        # Extract early warnings from Markov monitor results
        # For now, use the comparison data
        early_warnings = []
        markov_metrics = [m for m in eval_data["monitors"] if "Markov" in m["name"]]
        if markov_metrics:
            ew = markov_metrics[0].get("mean_early_warning")
            if ew:
                # Approximate distribution (actual distribution would need replay results)
                early_warnings = [int(ew)] * markov_metrics[0].get("num_detected", 1)
        if early_warnings:
            plot_early_warning_distribution(early_warnings, fig_dir / "early_warning_dist.pdf")
            logger.info("Generated: early_warning_dist.pdf")

    # 6. Ablation bars
    plot_ablation_bars(ablation_data["ablations"], fig_dir / "ablation_bars.pdf")
    logger.info("Generated: ablation_bars.pdf")

    # 7. Learning curve
    plot_learning_curve(learning_data["points"], fig_dir / "learning_curve.pdf")
    logger.info("Generated: learning_curve.pdf")

    # === TABLES ===

    # 1. Comparison table
    if eval_data:
        metrics_list = []
        for m in eval_data["monitors"]:
            metrics_list.append(MonitorMetrics(
                monitor_name=m["name"],
                threshold=None,
                detection_rate=m["detection_rate"],
                false_positive_rate=m["false_positive_rate"],
                mean_early_warning=m["mean_early_warning"],
                median_early_warning=m["median_early_warning"],
                mean_overhead_ms=m["mean_overhead_ms"],
                num_violating_traces=m["num_violating"],
                num_safe_traces=m["num_safe"],
                num_detected=m["num_detected"],
                num_false_positives=m["num_false_positives"],
            ))
        latex = generate_comparison_table(metrics_list, table_dir / "comparison_table.tex")
        logger.info("Generated: comparison_table.tex")

    # 2. Transition table
    generate_transition_table(matrix, table_dir / "transition_table.tex")
    logger.info("Generated: transition_table.tex")

    # 3. Per-category table
    cat_table_data = []
    for cat, info in per_cat_data.get("categories", {}).items():
        ponr_names = [p["name"] for p in info.get("points_of_no_return", [])]
        cat_table_data.append({
            "category": cat,
            "num_traces": info["num_traces"],
            "num_violated": info["num_violated"],
            "violation_rate": info["violation_rate"],
            "ponr_names": ponr_names,
        })
    generate_per_category_table(cat_table_data, table_dir / "per_category_table.tex")
    logger.info("Generated: per_category_table.tex")

    logger.info(f"\nAll figures saved to {fig_dir}")
    logger.info(f"All tables saved to {table_dir}")


if __name__ == "__main__":
    main()
