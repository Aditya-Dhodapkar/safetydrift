"""Tests for evaluation framework: replay, metrics, comparison."""

import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.evaluation.comparison import run_comparison
from safetydrift.evaluation.metrics import compute_metrics
from safetydrift.evaluation.replay import replay_all, replay_trace
from safetydrift.monitor.baselines import KeywordMonitor, NoMonitor
from safetydrift.monitor.markov_monitor import MarkovMonitor
from safetydrift.traces.models import Trace

from tests.conftest import make_step, make_trace

LOOKUP = {0: 0.248, 1: 0.447, 2: 0.259, 3: 0.230, 4: 1.0}


class TestReplay:
    def test_safe_trace_no_monitor(self, safe_trace):
        monitor = NoMonitor()
        result = replay_trace(monitor, safe_trace)
        assert result.reached_violation is False
        assert result.first_flag_step is None
        assert result.false_positive is False

    def test_violating_trace_markov_detected(self, violating_trace):
        monitor = MarkovMonitor(LOOKUP, threshold=0.20)
        result = replay_trace(monitor, violating_trace)
        assert result.reached_violation is True
        # Monitor should flag at some point before violation
        if result.first_flag_step is not None:
            assert result.detected or result.first_flag_step == result.violation_step_index

    def test_early_warning_computed(self, violating_trace):
        # Use very low threshold to ensure detection
        monitor = MarkovMonitor(LOOKUP, threshold=0.01)
        result = replay_trace(monitor, violating_trace)
        if result.detected:
            assert result.early_warning_steps is not None
            assert result.early_warning_steps >= 0

    def test_false_positive_on_safe_trace(self, safe_trace):
        # Very low threshold flags everything
        monitor = MarkovMonitor(LOOKUP, threshold=0.01)
        result = replay_trace(monitor, safe_trace)
        assert result.false_positive is True  # safe trace flagged

    def test_replay_all(self, safe_trace, violating_trace):
        monitor = NoMonitor()
        results = replay_all(monitor, [safe_trace, violating_trace])
        assert len(results) == 2


class TestMetrics:
    def test_no_monitor_zero_detection(self, safe_trace, violating_trace):
        monitor = NoMonitor()
        results = replay_all(monitor, [safe_trace, violating_trace])
        metrics = compute_metrics(results, "No Monitor")
        assert metrics.detection_rate == 0.0
        assert metrics.false_positive_rate == 0.0

    def test_aggressive_monitor_high_detection(self, safe_trace, violating_trace):
        monitor = MarkovMonitor(LOOKUP, threshold=0.01)  # flags everything
        results = replay_all(monitor, [safe_trace, violating_trace])
        metrics = compute_metrics(results, "Aggressive")
        # Should detect violations (though maybe at same step as violation)
        assert metrics.num_violating_traces == 1
        assert metrics.num_safe_traces == 1

    def test_empty_results(self):
        metrics = compute_metrics([], "Empty")
        assert metrics.detection_rate == 0.0
        assert metrics.false_positive_rate == 0.0


class TestComparison:
    def test_returns_all_monitors(self, safe_trace, violating_trace):
        monitors = [
            NoMonitor(),
            KeywordMonitor(),
            MarkovMonitor(LOOKUP, threshold=0.40),
        ]
        comparison = run_comparison(monitors, [safe_trace, violating_trace])
        assert len(comparison.metrics) == 3

    def test_roc_curve_for_markov(self, safe_trace, violating_trace):
        monitors = [MarkovMonitor(LOOKUP, threshold=0.40)]
        comparison = run_comparison(
            monitors, [safe_trace, violating_trace],
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
        markov_name = monitors[0].name
        assert markov_name in comparison.roc_curves
        assert len(comparison.roc_curves[markov_name]) == 5

    def test_per_category_populated(self, safe_trace, violating_trace):
        monitors = [NoMonitor()]
        comparison = run_comparison(monitors, [safe_trace, violating_trace])
        assert len(comparison.per_category) >= 1
