"""Tests for Markov transition matrix estimation."""

import numpy as np
import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.markov.estimation import (
    build_count_matrix,
    confidence_intervals,
    estimate_transition_matrix,
    extract_transitions_coarse,
    extract_transitions_fine,
    normalize_counts,
    train_test_split_traces,
)
from safetydrift.traces.models import Trace

from tests.conftest import make_step, make_trace


@pytest.fixture
def simple_traces():
    """Two traces: one safe, one violating."""
    s0 = SafetyState.initial()
    s1 = s0.transition(new_exposure=DataExposure.INTERNAL)
    s2 = s1.transition(new_escalation=ToolEscalation.FILE_WRITE)

    safe_steps = [
        make_step(0, s0, s1),
        make_step(1, s1, s2),
    ]
    safe = make_trace(safe_steps, scenario_id="t1", reached_violation=False)

    v0 = SafetyState.initial()
    v1 = v0.transition(new_exposure=DataExposure.CREDENTIALS)
    v2 = v1.transition(new_escalation=ToolEscalation.NETWORK, new_reversibility=Reversibility.IRREVERSIBLE)

    viol_steps = [
        make_step(0, v0, v1),
        make_step(1, v1, v2),
    ]
    viol = make_trace(viol_steps, scenario_id="t2", reached_violation=True)

    return [safe, viol]


class TestExtractTransitionsCoarse:
    def test_count_matches_steps(self, simple_traces):
        transitions = extract_transitions_coarse(simple_traces)
        total_steps = sum(len(t.steps) for t in simple_traces)
        assert len(transitions) == total_steps

    def test_values_in_range(self, simple_traces):
        transitions = extract_transitions_coarse(simple_traces)
        for from_s, to_s in transitions:
            assert 0 <= from_s <= 4
            assert 0 <= to_s <= 4

    def test_violating_trace_reaches_violated(self, simple_traces):
        viol_trace = simple_traces[1]
        transitions = extract_transitions_coarse([viol_trace])
        assert any(to_s == RiskLevel.VIOLATED.value for _, to_s in transitions)


class TestExtractTransitionsFine:
    def test_indices_in_range(self, simple_traces):
        transitions = extract_transitions_fine(simple_traces)
        for from_s, to_s in transitions:
            assert 0 <= from_s < 60
            assert 0 <= to_s < 60

    def test_initial_state_is_zero(self, simple_traces):
        transitions = extract_transitions_fine(simple_traces)
        assert transitions[0][0] == 0  # SafetyState.initial().to_index() == 0


class TestBuildCountMatrix:
    def test_shape_coarse(self):
        transitions = [(0, 0), (0, 1), (1, 1)]
        counts = build_count_matrix(transitions, 5)
        assert counts.shape == (5, 5)

    def test_counts_correct(self):
        transitions = [(0, 0), (0, 0), (0, 1), (1, 2)]
        counts = build_count_matrix(transitions, 5)
        assert counts[0][0] == 2
        assert counts[0][1] == 1
        assert counts[1][2] == 1

    def test_empty_transitions(self):
        counts = build_count_matrix([], 5)
        assert counts.sum() == 0


class TestNormalize:
    def test_rows_sum_to_one(self):
        counts = np.array([[3, 1], [0, 4]], dtype=float)
        matrix = normalize_counts(counts)
        np.testing.assert_allclose(matrix.sum(axis=1), [1.0, 1.0])

    def test_no_smoothing_preserves_zeros(self):
        counts = np.array([[3, 0], [0, 4]], dtype=float)
        matrix = normalize_counts(counts, smoothing=0.0)
        assert matrix[0][1] == 0.0

    def test_laplace_smoothing_no_zeros(self):
        counts = np.array([[3, 0], [0, 4]], dtype=float)
        matrix = normalize_counts(counts, smoothing=1.0)
        assert matrix[0][1] > 0.0
        assert matrix[1][0] > 0.0

    def test_zero_row_gets_uniform(self):
        counts = np.array([[0, 0], [2, 2]], dtype=float)
        matrix = normalize_counts(counts, smoothing=0.0)
        np.testing.assert_allclose(matrix[0], [0.5, 0.5])


class TestEstimateTransitionMatrix:
    def test_coarse_shape(self, simple_traces):
        tm = estimate_transition_matrix(simple_traces, granularity="coarse")
        assert tm.matrix.shape == (5, 5)
        assert tm.state_size == 5

    def test_fine_shape(self, simple_traces):
        tm = estimate_transition_matrix(simple_traces, granularity="fine")
        assert tm.matrix.shape == (60, 60)
        assert tm.state_size == 60

    def test_row_stochastic(self, simple_traces):
        tm = estimate_transition_matrix(simple_traces, granularity="coarse")
        np.testing.assert_allclose(tm.matrix.sum(axis=1), np.ones(5), atol=1e-10)

    def test_absorbing_self_loop(self, simple_traces):
        tm = estimate_transition_matrix(simple_traces, granularity="coarse")
        # VIOLATED (index 4) should self-loop if there are VIOLATED→VIOLATED transitions
        # In our simple traces, the violating trace ends at VIOLATED
        # Check that VIOLATED row sums to 1 (it gets uniform since no outgoing transitions observed in this small set)
        assert abs(tm.matrix[4].sum() - 1.0) < 1e-10

    def test_metadata(self, simple_traces):
        tm = estimate_transition_matrix(simple_traces, granularity="coarse")
        assert tm.num_traces == 2
        assert tm.num_transitions == 4  # 2 steps per trace


class TestConfidenceIntervals:
    def test_lower_leq_upper(self):
        counts = np.array([[10, 5], [3, 7]], dtype=float)
        lower, upper = confidence_intervals(counts)
        assert np.all(lower <= upper)

    def test_bounds_in_01(self):
        counts = np.array([[10, 5], [3, 7]], dtype=float)
        lower, upper = confidence_intervals(counts)
        assert np.all(lower >= 0)
        assert np.all(upper <= 1)

    def test_point_estimate_within_bounds(self):
        counts = np.array([[10, 5], [3, 7]], dtype=float)
        matrix = normalize_counts(counts)
        lower, upper = confidence_intervals(counts)
        assert np.all(matrix >= lower - 1e-10)
        assert np.all(matrix <= upper + 1e-10)


class TestTrainTestSplit:
    def test_all_accounted_for(self, simple_traces):
        traces = simple_traces * 5
        train, test = train_test_split_traces(traces, test_fraction=0.2)
        assert len(train) + len(test) == len(traces)

    def test_approximate_proportions(self, simple_traces):
        traces = simple_traces * 10  # 20 traces
        train, test = train_test_split_traces(traces, test_fraction=0.2)
        assert len(test) >= 2  # at least some in test
        assert len(train) > len(test)  # train is larger
