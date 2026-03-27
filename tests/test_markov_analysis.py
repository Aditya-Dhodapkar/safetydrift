"""Tests for points of no return, per-category analysis, validation, and ablation."""

import numpy as np
import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.markov.absorption import AbsorptionResult, analyze_absorption, compute_finite_horizon_probs
from safetydrift.markov.analysis import find_points_of_no_return, analyze_per_category, compare_category_absorption
from safetydrift.markov.estimation import TransitionMatrix, estimate_transition_matrix, train_test_split_traces
from safetydrift.markov.validation import extract_ngram_transitions, validate_markov_property
from safetydrift.markov.ablation import run_dimension_ablation, compute_learning_curve, project_state
from safetydrift.traces.io import load_trace
from safetydrift.traces.models import Trace

from tests.conftest import make_step, make_trace
from pathlib import Path


@pytest.fixture
def real_traces():
    """Load a subset of actual labeled traces for integration tests."""
    labeled_dir = Path("results/traces/labeled")
    if not labeled_dir.exists():
        pytest.skip("No labeled traces available")
    traces = []
    for f in sorted(labeled_dir.rglob("*.json"))[:20]:  # first 20 for speed
        traces.append(load_trace(f))
    if len(traces) < 5:
        pytest.skip("Not enough traces")
    return traces


class TestPointsOfNoReturn:
    def test_all_points_above_threshold(self, real_traces):
        tm = estimate_transition_matrix(real_traces, granularity="coarse")
        ar = analyze_absorption(tm, horizons=[5])
        result = find_points_of_no_return(ar, threshold=0.5, horizon=5)
        for p in result.points_of_no_return:
            assert p.finite_horizon_prob >= 0.5

    def test_zones_are_disjoint(self, real_traces):
        tm = estimate_transition_matrix(real_traces, granularity="coarse")
        ar = analyze_absorption(tm, horizons=[5])
        result = find_points_of_no_return(ar, threshold=0.85, horizon=5)
        safe = set(result.zones["safe"])
        trans = set(result.zones["transition"])
        danger = set(result.zones["danger"])
        assert safe.isdisjoint(trans)
        assert safe.isdisjoint(danger)
        assert trans.isdisjoint(danger)

    def test_cliff_description_not_empty(self, real_traces):
        tm = estimate_transition_matrix(real_traces, granularity="coarse")
        ar = analyze_absorption(tm, horizons=[5])
        result = find_points_of_no_return(ar, threshold=0.85, horizon=5)
        assert len(result.cliff_description) > 0


class TestPerCategoryAnalysis:
    def test_returns_categories(self, real_traces):
        results = analyze_per_category(real_traces, granularity="coarse")
        categories = {r.category for r in results}
        assert len(categories) >= 1

    def test_each_has_transition_matrix(self, real_traces):
        results = analyze_per_category(real_traces, granularity="coarse")
        for r in results:
            assert r.transition_matrix.matrix.shape == (5, 5)
            np.testing.assert_allclose(r.transition_matrix.matrix.sum(axis=1), np.ones(5), atol=1e-10)

    def test_comparison(self, real_traces):
        results = analyze_per_category(real_traces, granularity="coarse")
        comparison = compare_category_absorption(results)
        assert len(comparison) >= 1


class TestValidation:
    def test_returns_all_orders(self, real_traces):
        train, test = train_test_split_traces(real_traces, test_fraction=0.3)
        result = validate_markov_property(train, test, granularity="coarse")
        assert result.order_1_accuracy >= 0
        assert result.order_2_accuracy >= 0
        assert result.order_3_accuracy >= 0

    def test_log_likelihood_negative(self, real_traces):
        train, test = train_test_split_traces(real_traces, test_fraction=0.3)
        result = validate_markov_property(train, test, granularity="coarse")
        assert result.order_1_log_likelihood <= 0

    def test_chi_squared_pvalue_in_range(self, real_traces):
        train, test = train_test_split_traces(real_traces, test_fraction=0.3)
        result = validate_markov_property(train, test, granularity="coarse")
        assert 0 <= result.chi_squared_p_value <= 1


class TestNgrams:
    def test_order1_count(self, real_traces):
        ngrams = extract_ngram_transitions(real_traces, order=1, granularity="coarse")
        total_steps = sum(len(t.steps) for t in real_traces)
        assert len(ngrams) == total_steps

    def test_order2_count(self, real_traces):
        ngrams = extract_ngram_transitions(real_traces, order=2, granularity="coarse")
        # Order 2 needs at least 2 states in history, so loses 1 per trace
        assert len(ngrams) > 0
        assert len(ngrams) < sum(len(t.steps) for t in real_traces)


class TestAblation:
    def test_baseline_included(self, real_traces):
        train, test = train_test_split_traces(real_traces, test_fraction=0.3)
        results = run_dimension_ablation(train, test)
        baseline = [r for r in results if r.removed_dimension is None]
        assert len(baseline) == 1

    def test_three_ablations(self, real_traces):
        train, test = train_test_split_traces(real_traces, test_fraction=0.3)
        results = run_dimension_ablation(train, test)
        assert len(results) == 4  # baseline + 3 dimensions

    def test_project_state_full(self):
        s = SafetyState.initial()
        proj = project_state(s, None)
        assert proj == (0, 0, 0)

    def test_project_state_remove_one(self):
        s = SafetyState(
            data_exposure=DataExposure.SENSITIVE,
            tool_escalation=ToolEscalation.CODE_EXEC,
            reversibility=Reversibility.PARTIALLY,
        )
        proj = project_state(s, "data_exposure")
        assert proj == (2, 1)  # (tool_escalation, reversibility)


class TestLearningCurve:
    def test_returns_points(self, real_traces):
        curve = compute_learning_curve(real_traces, fractions=[0.5, 1.0], n_folds=2)
        assert len(curve) >= 1

    def test_accuracy_nonnegative(self, real_traces):
        curve = compute_learning_curve(real_traces, fractions=[0.5, 1.0], n_folds=2)
        for point in curve:
            assert point.accuracy >= 0
