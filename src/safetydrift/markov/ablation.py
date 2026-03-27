"""Dimension ablation study and learning curves.

Tests which state dimensions matter most for prediction accuracy,
and how many traces are needed for reliable transition estimation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from safetydrift.core.safety_state import SafetyState
from safetydrift.traces.models import Trace


@dataclass
class AblationResult:
    """Result of removing one state dimension."""

    removed_dimension: str | None  # None = full model (baseline)
    remaining_dimensions: list[str]
    state_size: int
    accuracy: float
    log_likelihood: float


@dataclass
class LearningCurvePoint:
    """One point on the learning curve."""

    num_traces: int
    accuracy: float
    accuracy_std: float
    log_likelihood: float
    log_likelihood_std: float


DIMENSIONS = ["data_exposure", "tool_escalation", "reversibility"]


def project_state(state: SafetyState, remove: str | None) -> tuple:
    """Project a SafetyState to a reduced representation.

    If remove is None, returns the full (d, t, r) tuple.
    Otherwise, removes the specified dimension.
    """
    d = state.data_exposure.value
    t = state.tool_escalation.value
    r = state.reversibility.value

    if remove is None:
        return (d, t, r)
    elif remove == "data_exposure":
        return (t, r)
    elif remove == "tool_escalation":
        return (d, r)
    elif remove == "reversibility":
        return (d, t)
    else:
        raise ValueError(f"Unknown dimension: {remove}")


def _extract_projected_transitions(
    traces: list[Trace],
    remove: str | None,
) -> list[tuple[tuple, tuple]]:
    """Extract transitions in projected state space."""
    transitions = []
    for trace in traces:
        for step in trace.steps:
            from_proj = project_state(step.safety_state_before, remove)
            to_proj = project_state(step.safety_state_after, remove)
            transitions.append((from_proj, to_proj))
    return transitions


def _build_projected_model(
    transitions: list[tuple[tuple, tuple]],
    smoothing: float = 1e-6,
) -> dict[tuple, dict[tuple, float]]:
    """Build a transition model in projected space.

    Returns dict mapping from_state → {to_state: probability}.
    """
    counts: dict[tuple, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
    for from_s, to_s in transitions:
        counts[from_s][to_s] += 1

    model = {}
    for from_s, to_counts in counts.items():
        total = sum(to_counts.values()) + smoothing * len(to_counts)
        model[from_s] = {to_s: (c + smoothing) / total for to_s, c in to_counts.items()}

    return model


def _evaluate_projected(
    test_transitions: list[tuple[tuple, tuple]],
    model: dict[tuple, dict[tuple, float]],
) -> tuple[float, float]:
    """Evaluate projected model. Returns (accuracy, mean_log_likelihood)."""
    if not test_transitions:
        return 0.0, 0.0

    correct = 0
    total_ll = 0.0

    for from_s, to_s in test_transitions:
        probs = model.get(from_s, {})
        if probs:
            best = max(probs, key=probs.get)
            if best == to_s:
                correct += 1
            prob = probs.get(to_s, 1e-10)
        else:
            prob = 1e-10

        total_ll += np.log(max(prob, 1e-10))

    n = len(test_transitions)
    return correct / n, total_ll / n


def run_dimension_ablation(
    train_traces: list[Trace],
    test_traces: list[Trace],
) -> list[AblationResult]:
    """Remove each dimension and measure prediction accuracy degradation.

    Returns list with baseline (full model) + 3 ablations.
    """
    results = []

    for remove in [None] + DIMENSIONS:
        train_trans = _extract_projected_transitions(train_traces, remove)
        test_trans = _extract_projected_transitions(test_traces, remove)
        model = _build_projected_model(train_trans)
        accuracy, ll = _evaluate_projected(test_trans, model)

        if remove is None:
            remaining = list(DIMENSIONS)
            name = None
        else:
            remaining = [d for d in DIMENSIONS if d != remove]
            name = remove

        # Count unique states
        all_states = set()
        for from_s, to_s in train_trans:
            all_states.add(from_s)
            all_states.add(to_s)

        results.append(AblationResult(
            removed_dimension=name,
            remaining_dimensions=remaining,
            state_size=len(all_states),
            accuracy=accuracy,
            log_likelihood=ll,
        ))

    return results


def compute_learning_curve(
    traces: list[Trace],
    fractions: list[float] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> list[LearningCurvePoint]:
    """How many traces are needed for a reliable transition matrix?

    For each fraction of training data, uses cross-validation to compute
    accuracy and its standard deviation.
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)

    results = []

    for frac in fractions:
        n_use = max(2, int(len(shuffled) * frac))

        fold_accuracies = []
        fold_lls = []

        for fold in range(n_folds):
            # Shuffle with different seed per fold
            fold_rng = random.Random(seed + fold)
            fold_traces = list(shuffled[:n_use])
            fold_rng.shuffle(fold_traces)

            # 80/20 split within the fold
            split = max(1, int(len(fold_traces) * 0.8))
            train = fold_traces[:split]
            test = fold_traces[split:]

            if not test:
                continue

            train_trans = _extract_projected_transitions(train, remove=None)
            test_trans = _extract_projected_transitions(test, remove=None)
            model = _build_projected_model(train_trans)
            acc, ll = _evaluate_projected(test_trans, model)

            fold_accuracies.append(acc)
            fold_lls.append(ll)

        if fold_accuracies:
            results.append(LearningCurvePoint(
                num_traces=n_use,
                accuracy=float(np.mean(fold_accuracies)),
                accuracy_std=float(np.std(fold_accuracies)),
                log_likelihood=float(np.mean(fold_lls)),
                log_likelihood_std=float(np.std(fold_lls)),
            ))

    return results
