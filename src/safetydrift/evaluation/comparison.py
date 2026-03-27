"""Run all monitors and produce the comparison table + ROC curves."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safetydrift.evaluation.metrics import MonitorMetrics, compute_metrics
from safetydrift.evaluation.replay import replay_all
from safetydrift.monitor.base import BaseMonitor
from safetydrift.monitor.markov_monitor import MarkovMonitor
from safetydrift.traces.models import Trace


@dataclass
class ROCPoint:
    threshold: float
    detection_rate: float
    false_positive_rate: float


@dataclass
class MonitorComparison:
    """Full comparison results."""

    metrics: list[MonitorMetrics]
    roc_curves: dict[str, list[ROCPoint]] = field(default_factory=dict)
    per_category: dict[str, list[MonitorMetrics]] = field(default_factory=dict)


def run_comparison(
    monitors: list[BaseMonitor],
    test_traces: list[Trace],
    thresholds: list[float] | None = None,
) -> MonitorComparison:
    """Run all monitors on test traces and produce comparison."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    all_metrics = []
    roc_curves: dict[str, list[ROCPoint]] = {}
    per_category: dict[str, list[MonitorMetrics]] = {}

    for monitor in monitors:
        # Full replay
        results = replay_all(monitor, test_traces)
        metrics = compute_metrics(results, monitor.name)
        all_metrics.append(metrics)

        # Per-category breakdown
        categories = set(r.scenario_category for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r.scenario_category == cat]
            cat_metrics = compute_metrics(cat_results, monitor.name)
            per_category.setdefault(cat, []).append(cat_metrics)

        # ROC sweep for threshold-based monitors
        if isinstance(monitor, MarkovMonitor):
            roc = []
            original = monitor.threshold
            for t in thresholds:
                monitor.threshold = t
                t_results = replay_all(monitor, test_traces)
                t_metrics = compute_metrics(t_results, monitor.name, threshold=t)
                roc.append(ROCPoint(
                    threshold=t,
                    detection_rate=t_metrics.detection_rate,
                    false_positive_rate=t_metrics.false_positive_rate,
                ))
            monitor.threshold = original
            roc_curves[monitor.name] = roc

    return MonitorComparison(
        metrics=all_metrics,
        roc_curves=roc_curves,
        per_category=per_category,
    )
