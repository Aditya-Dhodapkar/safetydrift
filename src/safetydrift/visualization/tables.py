"""LaTeX table generation for the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from safetydrift.evaluation.metrics import MonitorMetrics


def generate_comparison_table(
    metrics: list[MonitorMetrics],
    output_path: str | Path | None = None,
) -> str:
    """Generate LaTeX table comparing all monitors (the key result table)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime monitor comparison on test traces.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Monitor & Detection (\%) & FPR (\%) & Early Warning & Overhead (ms) \\",
        r"\midrule",
    ]

    # Find best values for bolding
    if metrics:
        best_det = max(m.detection_rate for m in metrics)
        best_fpr = min(m.false_positive_rate for m in metrics if m.detection_rate > 0) if any(m.detection_rate > 0 for m in metrics) else 0
        best_ew = max((m.mean_early_warning or 0) for m in metrics)

    for m in metrics:
        det = f"{100*m.detection_rate:.1f}"
        fpr = f"{100*m.false_positive_rate:.1f}"
        ew = f"{m.mean_early_warning:.1f}" if m.mean_early_warning is not None else "---"
        overhead = f"{m.mean_overhead_ms:.3f}"

        # Bold best values
        if m.detection_rate == best_det and m.detection_rate > 0:
            det = r"\textbf{" + det + "}"
        if m.mean_early_warning and m.mean_early_warning == best_ew:
            ew = r"\textbf{" + ew + "}"

        lines.append(f"{m.monitor_name} & {det} & {fpr} & {ew} & {overhead} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex)

    return latex


def generate_transition_table(
    matrix: np.ndarray,
    output_path: str | Path | None = None,
) -> str:
    """Generate LaTeX table for the transition matrix."""
    names = ["SAFE", "MILD", "ELEVATED", "CRITICAL", "VIOLATED"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Estimated transition probabilities (coarse 5-state model).}",
        r"\label{tab:transition}",
        r"\begin{tabular}{l" + "c" * 5 + "}",
        r"\toprule",
        r"From $\backslash$ To & " + " & ".join(names) + r" \\",
        r"\midrule",
    ]

    for i, name in enumerate(names):
        row_vals = []
        for j in range(5):
            val = matrix[i][j]
            if val == 0:
                row_vals.append("---")
            elif val >= 0.5:
                row_vals.append(r"\textbf{" + f"{val:.2f}" + "}")
            else:
                row_vals.append(f"{val:.2f}")
        lines.append(f"{name} & " + " & ".join(row_vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex)

    return latex


def generate_per_category_table(
    category_data: list[dict],
    output_path: str | Path | None = None,
) -> str:
    """Generate LaTeX table with per-category statistics."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-category safety drift statistics.}",
        r"\label{tab:categories}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Category & Traces & Violations & Rate (\%) & Points of no return \\",
        r"\midrule",
    ]

    for cat in category_data:
        name = cat["category"].replace("_", " ").title()
        ponr = ", ".join(cat.get("ponr_names", [])) or "---"
        lines.append(
            f"{name} & {cat['num_traces']} & {cat['num_violated']} & "
            f"{100*cat['violation_rate']:.0f} & {ponr} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex)

    return latex
