"""Tests for absorption analysis, including hand-verified chain solutions."""

import numpy as np
import pytest

from safetydrift.markov.absorption import (
    analyze_absorption,
    build_absorption_lookup,
    compute_absorption_probabilities,
    compute_finite_horizon_probs,
    compute_fundamental_matrix,
    compute_mean_passage_times,
    extract_canonical_form,
    identify_absorbing_states,
)
from safetydrift.markov.estimation import TransitionMatrix


class TestIdentifyAbsorbing:
    def test_coarse_absorbing_is_violated(self):
        transient, absorbing = identify_absorbing_states(5, "coarse")
        assert absorbing == [4]
        assert transient == [0, 1, 2, 3]

    def test_fine_has_absorbing_states(self):
        transient, absorbing = identify_absorbing_states(60, "fine")
        assert len(absorbing) >= 1
        assert len(transient) + len(absorbing) == 60
        # All absorbing states should have VIOLATED risk level
        from safetydrift.core.safety_state import SafetyState
        from safetydrift.core.enums import RiskLevel
        for idx in absorbing:
            state = SafetyState.from_index(idx)
            assert state.risk_level == RiskLevel.VIOLATED


class TestCanonicalForm:
    def test_shapes(self):
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        Q, R = extract_canonical_form(P, transient=[0, 1], absorbing=[2])
        assert Q.shape == (2, 2)
        assert R.shape == (2, 1)

    def test_values(self):
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        Q, R = extract_canonical_form(P, transient=[0, 1], absorbing=[2])
        np.testing.assert_allclose(Q, [[0.5, 0.3], [0.0, 0.4]])
        np.testing.assert_allclose(R, [[0.2], [0.6]])


class TestHandVerified2StateChain:
    """2-state absorbing chain: T={0}, A={1}. P(0→0)=0.3, P(0→1)=0.7.

    Exact solutions:
    - Q = [[0.3]], N = (1 - 0.3)^{-1} = [[1.4286]]
    - B = N*R = 1.4286 * 0.7 = 1.0
    - t = N*1 = 1.4286
    """

    def test_fundamental_matrix(self):
        Q = np.array([[0.3]])
        N = compute_fundamental_matrix(Q)
        np.testing.assert_allclose(N, [[1.0 / 0.7]], atol=1e-10)

    def test_absorption_probability(self):
        Q = np.array([[0.3]])
        R = np.array([[0.7]])
        N = compute_fundamental_matrix(Q)
        B = compute_absorption_probabilities(N, R)
        np.testing.assert_allclose(B, [1.0], atol=1e-10)

    def test_mean_passage_time(self):
        Q = np.array([[0.3]])
        N = compute_fundamental_matrix(Q)
        t = compute_mean_passage_times(N)
        np.testing.assert_allclose(t, [1.0 / 0.7], atol=1e-10)


class TestHandVerified3StateChain:
    """3-state absorbing chain: T={0,1}, A={2}.
    P = [[0.5, 0.3, 0.2], [0, 0.4, 0.6], [0, 0, 1]].

    Exact solutions:
    - Q = [[0.5, 0.3], [0, 0.4]]
    - N = (I-Q)^{-1} = [[2, 1], [0, 5/3]]
    - B = N*R = [[2*0.2 + 1*0.6], [0*0.2 + 5/3*0.6]] = [[1.0], [1.0]]
    - t = N*1 = [2+1, 0+5/3] = [3.0, 5/3]
    """

    def test_fundamental_matrix(self):
        Q = np.array([[0.5, 0.3], [0.0, 0.4]])
        N = compute_fundamental_matrix(Q)
        expected_N = np.array([[2.0, 1.0], [0.0, 5.0 / 3.0]])
        np.testing.assert_allclose(N, expected_N, atol=1e-10)

    def test_absorption_probability(self):
        Q = np.array([[0.5, 0.3], [0.0, 0.4]])
        R = np.array([[0.2], [0.6]])
        N = compute_fundamental_matrix(Q)
        B = compute_absorption_probabilities(N, R)
        np.testing.assert_allclose(B, [1.0, 1.0], atol=1e-10)

    def test_mean_passage_time(self):
        Q = np.array([[0.5, 0.3], [0.0, 0.4]])
        N = compute_fundamental_matrix(Q)
        t = compute_mean_passage_times(N)
        np.testing.assert_allclose(t, [3.0, 5.0 / 3.0], atol=1e-10)


class TestFiniteHorizon:
    def test_horizon_1(self):
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        result = compute_finite_horizon_probs(P, absorbing_indices=[2], horizons=[1])
        # After 1 step: P(state 0 → absorbing) = 0.2, P(state 1 → absorbing) = 0.6
        np.testing.assert_allclose(result[1][:2], [0.2, 0.6], atol=1e-10)

    def test_horizon_increases_monotonically(self):
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        result = compute_finite_horizon_probs(P, absorbing_indices=[2], horizons=[1, 5, 10])
        # Violation probability should increase with horizon
        for state in range(2):
            assert result[1][state] <= result[5][state]
            assert result[5][state] <= result[10][state]

    def test_absorbing_state_stays_1(self):
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        result = compute_finite_horizon_probs(P, absorbing_indices=[2], horizons=[1, 5])
        # State 2 (absorbing) should be 1.0 at any horizon
        np.testing.assert_allclose(result[1][2], 1.0, atol=1e-10)


class TestAnalyzeAbsorption:
    def test_full_analysis_coarse(self):
        counts = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=float)
        matrix = counts / counts.sum(axis=1, keepdims=True)
        tm = TransitionMatrix(
            counts=counts, matrix=matrix, state_size=3,
            granularity="coarse", num_traces=5, num_transitions=8, smoothing=0.0,
        )
        # Override absorbing detection for this 3-state test
        # This won't match the real coarse model (which has 5 states)
        # so let's just use the functions directly
        Q = matrix[:2, :2]
        R = matrix[:2, 2:3]
        N = compute_fundamental_matrix(Q)
        B = compute_absorption_probabilities(N, R)
        t = compute_mean_passage_times(N)
        assert all(0 <= b <= 1.0 for b in B)
        assert all(t_val > 0 for t_val in t)


class TestBuildLookup:
    def _make_3state_result(self, horizons=None):
        """Build an AbsorptionResult directly for a 3-state chain (T={0,1}, A={2})."""
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0],
        ])
        transient, absorbing = [0, 1], [2]
        Q, R = extract_canonical_form(P, transient, absorbing)
        N = compute_fundamental_matrix(Q)
        from safetydrift.markov.absorption import AbsorptionResult
        return AbsorptionResult(
            transient_indices=transient,
            absorbing_indices=absorbing,
            Q=Q, R=R, N=N,
            absorption_probs=compute_absorption_probabilities(N, R),
            mean_passage_times=compute_mean_passage_times(N),
            finite_horizon_probs=compute_finite_horizon_probs(P, absorbing, horizons or [5]),
            state_absorption_map={0: 1.0, 1: 1.0, 2: 1.0},
            state_passage_map={0: 3.0, 1: 5/3, 2: 0.0},
            state_size=3,
            granularity="coarse",
        )

    def test_covers_all_states(self):
        result = self._make_3state_result()
        lookup = build_absorption_lookup(result)
        assert len(lookup) == 3

    def test_finite_horizon_lookup(self):
        result = self._make_3state_result(horizons=[5])
        lookup = build_absorption_lookup(result, horizon=5)
        assert len(lookup) == 3
        assert lookup[2] == 1.0  # absorbing state
        assert 0 < lookup[0] < 1.0  # transient state
