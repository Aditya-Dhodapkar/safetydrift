"""Run Phase 4 evaluation: all monitors on test traces (Experiment 3).

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py monitor.threshold=0.30
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from safetydrift.evaluation.comparison import run_comparison
from safetydrift.evaluation.metrics import compute_metrics
from safetydrift.evaluation.replay import replay_all
from safetydrift.markov.estimation import train_test_split_traces
from safetydrift.monitor.baselines import (
    CategoryPriorMonitor,
    KeywordMonitor,
    NoMonitor,
    Pro2GuardStyleMonitor,
)
from safetydrift.monitor.markov_monitor import MarkovMonitor
from safetydrift.traces.io import load_trace

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # 1. Load traces
    traces_dir = Path(cfg.markov.traces_dir)
    traces = [load_trace(f) for f in sorted(traces_dir.rglob("*.json"))]
    logger.info(f"Loaded {len(traces)} traces")

    # 2. Same train/test split as Phase 3
    _, test_traces = train_test_split_traces(
        traces, test_fraction=cfg.markov.test_fraction, seed=cfg.markov.seed,
    )
    logger.info(f"Test set: {len(test_traces)} traces")

    # 3. Build monitors
    markov = MarkovMonitor.from_json(cfg.monitor.lookup_path, threshold=cfg.monitor.threshold)
    pro2guard = Pro2GuardStyleMonitor.from_json(
        "results/markov/pro2guard_lookup.json", threshold=0.50,
    )
    category_prior = CategoryPriorMonitor()
    monitors = [NoMonitor(), KeywordMonitor(), category_prior, pro2guard, markov]

    # 4. Run comparison
    sweep_thresholds = list(np.arange(
        cfg.monitor.threshold_sweep.start,
        cfg.monitor.threshold_sweep.stop,
        cfg.monitor.threshold_sweep.step,
    ))
    comparison = run_comparison(monitors, test_traces, thresholds=sweep_thresholds)

    # 5. Print results
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: RUNTIME MONITOR COMPARISON")
    print("=" * 70)

    for m in comparison.metrics:
        ew = f"{m.mean_early_warning:.1f}" if m.mean_early_warning else "---"
        print(f"\n{m.monitor_name}:")
        print(f"  Detection rate: {100*m.detection_rate:.1f}% ({m.num_detected}/{m.num_violating_traces})")
        print(f"  False positive rate: {100*m.false_positive_rate:.1f}% ({m.num_false_positives}/{m.num_safe_traces})")
        print(f"  Mean early warning: {ew} steps")
        print(f"  Mean overhead: {m.mean_overhead_ms:.4f} ms/step")

    # Per-category
    print("\n--- Per-Category Breakdown ---")
    for cat, cat_metrics in sorted(comparison.per_category.items()):
        print(f"\n{cat}:")
        for m in cat_metrics:
            det = f"{100*m.detection_rate:.0f}%"
            fpr = f"{100*m.false_positive_rate:.0f}%"
            print(f"  {m.monitor_name}: det={det}, fpr={fpr}")

    # 6. Save results
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "monitors": [{
            "name": m.monitor_name,
            "detection_rate": m.detection_rate,
            "false_positive_rate": m.false_positive_rate,
            "mean_early_warning": m.mean_early_warning,
            "median_early_warning": m.median_early_warning,
            "mean_overhead_ms": m.mean_overhead_ms,
            "num_detected": m.num_detected,
            "num_violating": m.num_violating_traces,
            "num_false_positives": m.num_false_positives,
            "num_safe": m.num_safe_traces,
        } for m in comparison.metrics],
        "roc_curves": {
            name: [{"threshold": p.threshold, "detection_rate": p.detection_rate,
                     "false_positive_rate": p.false_positive_rate} for p in points]
            for name, points in comparison.roc_curves.items()
        },
        "per_category": {
            cat: [{"name": m.monitor_name, "detection_rate": m.detection_rate,
                    "false_positive_rate": m.false_positive_rate,
                    "mean_early_warning": m.mean_early_warning} for m in metrics]
            for cat, metrics in comparison.per_category.items()
        },
    }

    (output_dir / "comparison_results.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
