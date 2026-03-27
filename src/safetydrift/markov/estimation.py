"""Transition matrix estimation from labeled execution traces.

Extracts state transitions, builds count matrices, normalizes to stochastic
matrices, computes Wilson confidence intervals, and handles train/test splitting.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats

from safetydrift.core.enums import RiskLevel
from safetydrift.traces.models import Trace


@dataclass
class TransitionMatrix:
    """Estimated transition matrix with metadata."""

    counts: np.ndarray  # raw count matrix (n x n)
    matrix: np.ndarray  # row-stochastic probability matrix (n x n)
    state_size: int
    granularity: str  # "coarse" or "fine"
    num_traces: int
    num_transitions: int
    smoothing: float


def extract_transitions_coarse(traces: list[Trace]) -> list[tuple[int, int]]:
    """Extract (from_risk_level, to_risk_level) pairs from traces.

    The initial state for each trace is the safety_state_before of step 0.
    Each step produces one transition: before → after.
    """
    transitions = []
    for trace in traces:
        for step in trace.steps:
            from_state = step.safety_state_before.risk_level.value
            to_state = step.safety_state_after.risk_level.value
            transitions.append((from_state, to_state))
    return transitions


def extract_transitions_fine(traces: list[Trace]) -> list[tuple[int, int]]:
    """Extract (from_state_index, to_state_index) pairs using dense [0,59] indexing."""
    transitions = []
    for trace in traces:
        for step in trace.steps:
            from_idx = step.safety_state_before.to_index()
            to_idx = step.safety_state_after.to_index()
            transitions.append((from_idx, to_idx))
    return transitions


def build_count_matrix(transitions: list[tuple[int, int]], state_size: int) -> np.ndarray:
    """Build raw count matrix from transition pairs."""
    counts = np.zeros((state_size, state_size), dtype=float)
    for from_s, to_s in transitions:
        if 0 <= from_s < state_size and 0 <= to_s < state_size:
            counts[from_s][to_s] += 1
    return counts


def normalize_counts(counts: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Row-normalize count matrix to get a stochastic matrix.

    Args:
        counts: Raw count matrix.
        smoothing: Laplace smoothing alpha. 0.0 = no smoothing.

    Returns:
        Row-stochastic matrix (each row sums to 1.0).
        Zero-sum rows get uniform distribution.
    """
    n = counts.shape[0]
    smoothed = counts + smoothing
    row_sums = smoothed.sum(axis=1, keepdims=True)

    # Handle zero rows (unvisited states)
    zero_mask = row_sums.flatten() == 0
    matrix = np.zeros_like(smoothed)
    nonzero = ~zero_mask

    matrix[nonzero] = smoothed[nonzero] / row_sums[nonzero]
    matrix[zero_mask] = 1.0 / n  # uniform for unvisited states

    return matrix


def estimate_transition_matrix(
    traces: list[Trace],
    granularity: Literal["coarse", "fine"] = "coarse",
    smoothing: float = 0.0,
) -> TransitionMatrix:
    """Main entry point: estimate transition matrix from traces."""
    if granularity == "coarse":
        transitions = extract_transitions_coarse(traces)
        state_size = len(RiskLevel)  # 5
    else:
        transitions = extract_transitions_fine(traces)
        state_size = 60

    counts = build_count_matrix(transitions, state_size)
    matrix = normalize_counts(counts, smoothing)

    return TransitionMatrix(
        counts=counts,
        matrix=matrix,
        state_size=state_size,
        granularity=granularity,
        num_traces=len(traces),
        num_transitions=len(transitions),
        smoothing=smoothing,
    )


def confidence_intervals(
    counts: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score confidence intervals for each transition probability.

    Returns:
        (lower, upper) arrays of same shape as counts.
    """
    n = counts.shape[0]
    lower = np.zeros_like(counts, dtype=float)
    upper = np.zeros_like(counts, dtype=float)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    for i in range(n):
        row_total = counts[i].sum()
        if row_total == 0:
            lower[i] = 0.0
            upper[i] = 1.0
            continue

        for j in range(n):
            k = counts[i][j]
            n_total = row_total
            p_hat = k / n_total

            # Wilson score interval
            denom = 1 + z**2 / n_total
            center = (p_hat + z**2 / (2 * n_total)) / denom
            spread = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom

            lower[i][j] = max(0.0, center - spread)
            upper[i][j] = min(1.0, center + spread)

    return lower, upper


def train_test_split_traces(
    traces: list[Trace],
    test_fraction: float = 0.2,
    seed: int = 42,
    stratify_by: Literal["violation", "category", "both"] = "both",
) -> tuple[list[Trace], list[Trace]]:
    """Split traces into train/test with stratification.

    Stratifies to ensure representative proportions of violation status
    and/or scenario categories in both splits.
    """
    rng = random.Random(seed)

    # Group traces by stratum
    strata: dict[Any, list[Trace]] = defaultdict(list)
    for trace in traces:
        if stratify_by == "violation":
            key = trace.metadata.reached_violation
        elif stratify_by == "category":
            key = trace.metadata.scenario_category
        else:  # both
            key = (trace.metadata.scenario_category, trace.metadata.reached_violation)
        strata[key].append(trace)

    train, test = [], []
    for key, group in strata.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_fraction))
        # Ensure at least 1 in train if possible
        if n_test >= len(shuffled):
            n_test = len(shuffled) - 1 if len(shuffled) > 1 else 0
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    return train, test
