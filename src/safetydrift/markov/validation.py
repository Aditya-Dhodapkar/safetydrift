"""Markov property validation: compare 1st, 2nd, 3rd order models.

Tests whether 1st-order Markov (P(next | current)) is sufficient,
or if longer history (P(next | current, previous, ...)) helps significantly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from safetydrift.markov.estimation import (
    TransitionMatrix,
    estimate_transition_matrix,
    extract_transitions_coarse,
    extract_transitions_fine,
)
from safetydrift.traces.models import Trace


@dataclass
class ValidationResult:
    """Results of Markov property validation."""

    order_1_accuracy: float
    order_1_log_likelihood: float
    order_2_accuracy: float
    order_2_log_likelihood: float
    order_3_accuracy: float
    order_3_log_likelihood: float
    per_state_accuracy: dict[int, float]
    chi_squared_statistic: float
    chi_squared_p_value: float
    degrees_of_freedom: int
    num_train_transitions: int
    num_test_transitions: int


def extract_ngram_transitions(
    traces: list[Trace],
    order: int,
    granularity: Literal["coarse", "fine"] = "coarse",
) -> list[tuple[tuple[int, ...], int]]:
    """Extract n-gram transitions of given order.

    For order=1: returns [((current,), next), ...]
    For order=2: returns [((prev, current,), next), ...]
    """
    result = []
    for trace in traces:
        if granularity == "coarse":
            states = [step.safety_state_before.risk_level.value for step in trace.steps]
            states.append(trace.steps[-1].safety_state_after.risk_level.value)
        else:
            states = [step.safety_state_before.to_index() for step in trace.steps]
            states.append(trace.steps[-1].safety_state_after.to_index())

        for i in range(order, len(states)):
            context = tuple(states[i - order:i])
            target = states[i]
            result.append((context, target))

    return result


def build_higher_order_model(
    ngram_transitions: list[tuple[tuple[int, ...], int]],
    state_size: int,
    smoothing: float = 1e-6,
) -> dict[tuple[int, ...], np.ndarray]:
    """Build a higher-order transition model.

    Returns mapping from context tuple to probability distribution over next state.
    """
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(state_size))

    for context, target in ngram_transitions:
        counts[context][target] += 1

    model = {}
    for context, count_vec in counts.items():
        smoothed = count_vec + smoothing
        total = smoothed.sum()
        model[context] = smoothed / total if total > 0 else np.ones(state_size) / state_size

    return model


def evaluate_prediction_accuracy(
    test_transitions: list[tuple[tuple[int, ...], int]],
    model: dict[tuple[int, ...], np.ndarray],
    state_size: int,
) -> tuple[float, float]:
    """Evaluate a model on test transitions.

    Returns (accuracy, mean_log_likelihood).
    """
    correct = 0
    total_ll = 0.0
    n = len(test_transitions)

    if n == 0:
        return 0.0, 0.0

    uniform = np.ones(state_size) / state_size

    for context, target in test_transitions:
        probs = model.get(context, uniform)
        predicted = int(np.argmax(probs))
        if predicted == target:
            correct += 1
        prob = max(probs[target], 1e-10)  # avoid log(0)
        total_ll += np.log(prob)

    return correct / n, total_ll / n


def chi_squared_markov_test(
    traces: list[Trace],
    granularity: Literal["coarse", "fine"] = "coarse",
) -> tuple[float, float, int]:
    """Chi-squared test for 1st-order Markov property.

    Tests H0: P(X_{t+1} | X_t, X_{t-1}) = P(X_{t+1} | X_t).
    """
    # Build 1st-order model
    order1_transitions = extract_ngram_transitions(traces, order=1, granularity=granularity)
    state_size = 5 if granularity == "coarse" else 60

    # Count 1st-order transitions
    count_1: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(state_size))
    for (current,), target in order1_transitions:
        count_1[current][target] += 1

    # Count 2nd-order transitions
    order2_transitions = extract_ngram_transitions(traces, order=2, granularity=granularity)
    count_2: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(state_size))
    for (prev, current), target in order2_transitions:
        count_2[(prev, current)][target] += 1

    # Chi-squared statistic
    chi2 = 0.0
    df = 0

    for (prev, current), observed in count_2.items():
        n_context = observed.sum()
        if n_context < 5:  # skip sparse contexts
            continue

        # Expected under 1st-order: P(target | current) * n_context
        total_1 = count_1[current].sum()
        if total_1 == 0:
            continue

        expected = (count_1[current] / total_1) * n_context

        for target in range(state_size):
            if expected[target] > 0:
                chi2 += (observed[target] - expected[target]) ** 2 / expected[target]
                if observed[target] > 0 or expected[target] >= 1:
                    df += 1

    df = max(1, df - 1)  # adjust degrees of freedom

    from scipy import stats
    p_value = 1.0 - stats.chi2.cdf(chi2, df)

    return float(chi2), float(p_value), df


def validate_markov_property(
    train_traces: list[Trace],
    test_traces: list[Trace],
    granularity: Literal["coarse", "fine"] = "coarse",
) -> ValidationResult:
    """Main entry point: full Markov validation."""
    state_size = 5 if granularity == "coarse" else 60

    results = {}
    for order in [1, 2, 3]:
        train_ngrams = extract_ngram_transitions(train_traces, order, granularity)
        test_ngrams = extract_ngram_transitions(test_traces, order, granularity)
        model = build_higher_order_model(train_ngrams, state_size)
        acc, ll = evaluate_prediction_accuracy(test_ngrams, model, state_size)
        results[order] = (acc, ll)

    # Per-state accuracy for order 1
    test_1grams = extract_ngram_transitions(test_traces, 1, granularity)
    model_1 = build_higher_order_model(
        extract_ngram_transitions(train_traces, 1, granularity), state_size
    )
    per_state_correct: dict[int, list[bool]] = defaultdict(list)
    uniform = np.ones(state_size) / state_size
    for (current,), target in test_1grams:
        probs = model_1.get((current,), uniform)
        per_state_correct[current].append(int(np.argmax(probs)) == target)

    per_state_accuracy = {
        state: sum(hits) / len(hits) if hits else 0.0
        for state, hits in per_state_correct.items()
    }

    # Chi-squared test
    chi2, p_val, df = chi_squared_markov_test(train_traces + test_traces, granularity)

    train_count = sum(len(t.steps) for t in train_traces)
    test_count = sum(len(t.steps) for t in test_traces)

    return ValidationResult(
        order_1_accuracy=results[1][0],
        order_1_log_likelihood=results[1][1],
        order_2_accuracy=results[2][0],
        order_2_log_likelihood=results[2][1],
        order_3_accuracy=results[3][0],
        order_3_log_likelihood=results[3][1],
        per_state_accuracy=per_state_accuracy,
        chi_squared_statistic=chi2,
        chi_squared_p_value=p_val,
        degrees_of_freedom=df,
        num_train_transitions=train_count,
        num_test_transitions=test_count,
    )
