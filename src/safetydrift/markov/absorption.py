"""Absorbing Markov chain analysis.

Computes fundamental matrix N = (I-Q)^{-1}, absorption probabilities B = N*R,
mean passage times t = N*1, and finite-horizon violation probabilities P^n.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import linalg

from safetydrift.core.enums import RiskLevel
from safetydrift.core.safety_state import SafetyState
from safetydrift.markov.estimation import TransitionMatrix


@dataclass
class AbsorptionResult:
    """Complete absorption analysis results."""

    transient_indices: list[int]
    absorbing_indices: list[int]
    Q: np.ndarray  # transient-to-transient (|T| x |T|)
    R: np.ndarray  # transient-to-absorbing (|T| x |A|)
    N: np.ndarray  # fundamental matrix (|T| x |T|)
    absorption_probs: np.ndarray  # P(eventually absorbed) per transient state
    mean_passage_times: np.ndarray  # E[steps to absorption] per transient state
    finite_horizon_probs: dict[int, np.ndarray]  # horizon -> P(VIOLATED within n steps) per state
    state_absorption_map: dict[int, float]  # state_index -> P(VIOLATED)
    state_passage_map: dict[int, float]  # state_index -> E[steps to VIOLATED]
    state_size: int
    granularity: str


def identify_absorbing_states(
    state_size: int,
    granularity: Literal["coarse", "fine"] = "coarse",
) -> tuple[list[int], list[int]]:
    """Identify transient and absorbing state indices.

    For coarse: absorbing = [4] (VIOLATED).
    For fine: absorbing = all states where risk_level == VIOLATED.
    """
    if granularity == "coarse":
        absorbing = [RiskLevel.VIOLATED.value]  # [4]
        transient = [i for i in range(state_size) if i not in absorbing]
    else:
        absorbing = []
        transient = []
        for idx in range(state_size):
            state = SafetyState.from_index(idx)
            if state.risk_level == RiskLevel.VIOLATED:
                absorbing.append(idx)
            else:
                transient.append(idx)
    return transient, absorbing


def extract_canonical_form(
    matrix: np.ndarray,
    transient: list[int],
    absorbing: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Q and R sub-matrices from the full transition matrix.

    Q = transitions among transient states.
    R = transitions from transient to absorbing states.
    """
    t_idx = np.array(transient)
    a_idx = np.array(absorbing)

    Q = matrix[np.ix_(t_idx, t_idx)]
    R = matrix[np.ix_(t_idx, a_idx)]

    return Q, R


def compute_fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    """N = (I - Q)^{-1}, computed via solving (I-Q) * N = I.

    Uses scipy.linalg.solve for numerical stability.
    """
    n = Q.shape[0]
    I = np.eye(n)
    I_minus_Q = I - Q

    # Solve (I - Q) * N = I for N
    N = linalg.solve(I_minus_Q, I)
    return N


def compute_absorption_probabilities(N: np.ndarray, R: np.ndarray) -> np.ndarray:
    """B = N * R. For single absorbing class, sum across absorbing columns.

    Returns 1D array: P(eventually reach any absorbing state) per transient state.
    """
    B = N @ R
    # Sum across absorbing states (in case there are multiple VIOLATED states)
    return B.sum(axis=1)


def compute_mean_passage_times(N: np.ndarray) -> np.ndarray:
    """t = N * 1. Expected number of steps before absorption.

    Returns 1D array: E[steps to absorption] per transient state.
    """
    ones = np.ones(N.shape[1])
    return N @ ones


def compute_finite_horizon_probs(
    matrix: np.ndarray,
    absorbing_indices: list[int],
    horizons: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Compute P(VIOLATED within n steps | current state) for various horizons.

    Uses matrix exponentiation: P^n gives n-step transition probabilities.
    The column(s) corresponding to absorbing states give violation probability.

    Returns:
        dict mapping horizon -> array of P(VIOLATED within n steps) per state.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    result = {}
    for h in horizons:
        P_n = np.linalg.matrix_power(matrix, h)
        # Sum probabilities of being in any absorbing state after n steps
        violation_prob = P_n[:, absorbing_indices].sum(axis=1)
        result[h] = violation_prob

    return result


def analyze_absorption(
    tm: TransitionMatrix,
    horizons: list[int] | None = None,
) -> AbsorptionResult:
    """Main entry point: full absorption analysis from a TransitionMatrix."""
    transient, absorbing = identify_absorbing_states(tm.state_size, tm.granularity)
    Q, R = extract_canonical_form(tm.matrix, transient, absorbing)
    N = compute_fundamental_matrix(Q)
    absorption_probs = compute_absorption_probabilities(N, R)
    mean_passage = compute_mean_passage_times(N)
    finite_horizon = compute_finite_horizon_probs(tm.matrix, absorbing, horizons)

    # Build convenience maps from original state indices
    state_absorption_map: dict[int, float] = {}
    state_passage_map: dict[int, float] = {}

    for i, t_idx in enumerate(transient):
        state_absorption_map[t_idx] = float(absorption_probs[i])
        state_passage_map[t_idx] = float(mean_passage[i])

    for a_idx in absorbing:
        state_absorption_map[a_idx] = 1.0
        state_passage_map[a_idx] = 0.0

    return AbsorptionResult(
        transient_indices=transient,
        absorbing_indices=absorbing,
        Q=Q,
        R=R,
        N=N,
        absorption_probs=absorption_probs,
        mean_passage_times=mean_passage,
        finite_horizon_probs=finite_horizon,
        state_absorption_map=state_absorption_map,
        state_passage_map=state_passage_map,
        state_size=tm.state_size,
        granularity=tm.granularity,
    )


def build_absorption_lookup(
    result: AbsorptionResult,
    horizon: int | None = None,
) -> dict[int, float]:
    """Build lookup table for Phase 4's runtime monitor.

    Args:
        result: AbsorptionResult from analyze_absorption.
        horizon: If specified, use finite-horizon probability. If None, use infinite-horizon.

    Returns:
        Dict mapping state_index -> P(VIOLATED).
    """
    if horizon is not None and horizon in result.finite_horizon_probs:
        probs = result.finite_horizon_probs[horizon]
        return {i: float(probs[i]) for i in range(result.state_size)}
    else:
        return dict(result.state_absorption_map)
