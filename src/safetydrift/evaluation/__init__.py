"""Evaluation framework: baselines, metrics, and comparison."""

from safetydrift.evaluation.comparison import MonitorComparison, ROCPoint, run_comparison
from safetydrift.evaluation.metrics import MonitorMetrics, compute_metrics
from safetydrift.evaluation.replay import TraceReplayResult, replay_all, replay_trace

__all__ = [
    "MonitorComparison",
    "MonitorMetrics",
    "ROCPoint",
    "TraceReplayResult",
    "compute_metrics",
    "replay_all",
    "replay_trace",
    "run_comparison",
]
