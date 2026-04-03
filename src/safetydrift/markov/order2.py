"""2nd-order Markov analysis via product-space embedding.

Converts a 2nd-order chain on 5 risk-level states into a 1st-order chain
on 25 product states (prev, curr), then reuses existing absorption code.
"""

from __future__ import annotations

import numpy as np

from safetydrift.core.enums import RiskLevel
from safetydrift.markov.absorption import compute_finite_horizon_probs
from safetydrift.markov.estimation import normalize_counts
from safetydrift.traces.models import Trace

N_RISK = 5  # number of coarse risk levels


def product_index(prev: int, curr: int) -> int:
    """Map (prev_risk, curr_risk) to a single product-state index."""
    return prev * N_RISK + curr


def unpack_index(idx: int) -> tuple[int, int]:
    """Reverse: product index → (prev_risk, curr_risk)."""
    return divmod(idx, N_RISK)


def extract_trigrams(traces: list[Trace]) -> list[tuple[int, int, int]]:
    """Extract (s_{t-1}, s_t, s_{t+1}) trigrams from traces."""
    trigrams = []
    for trace in traces:
        levels = [s.risk_level.value for s in trace.safety_trajectory]
        for i in range(len(levels) - 2):
            trigrams.append((levels[i], levels[i + 1], levels[i + 2]))
    return trigrams


def build_product_matrix(traces: list[Trace], smoothing: float = 1e-6) -> np.ndarray:
    """Build 25x25 product-space transition matrix from traces.

    Product state (prev, curr) transitions to (curr, next).
    """
    n = N_RISK * N_RISK  # 25
    counts = np.zeros((n, n), dtype=float)

    for prev, curr, nxt in extract_trigrams(traces):
        from_idx = product_index(prev, curr)
        to_idx = product_index(curr, nxt)
        counts[from_idx, to_idx] += 1

    return normalize_counts(counts, smoothing)


def product_absorbing_indices() -> list[int]:
    """Product states where curr == VIOLATED (absorbing)."""
    viol = RiskLevel.VIOLATED.value
    return [product_index(prev, viol) for prev in range(N_RISK)]


def build_order2_absorption_lookup(
    traces: list[Trace],
    horizon: int = 5,
    smoothing: float = 1e-6,
) -> dict[tuple[int, int], float]:
    """End-to-end: traces → lookup mapping (prev, curr) → P(VIOLATED in horizon).

    Returns dict keyed by (prev_risk_level, curr_risk_level) tuples.
    """
    matrix = build_product_matrix(traces, smoothing)
    absorbing = product_absorbing_indices()
    fh = compute_finite_horizon_probs(matrix, absorbing, [horizon])
    probs = fh[horizon]  # array of length 25

    lookup = {}
    for idx in range(N_RISK * N_RISK):
        prev, curr = unpack_index(idx)
        lookup[(prev, curr)] = float(probs[idx])
    return lookup


def build_per_category_order2_lookups(
    traces: list[Trace],
    horizon: int = 5,
    smoothing: float = 1e-6,
) -> tuple[dict[str, dict[tuple[int, int], float]], dict[tuple[int, int], float]]:
    """Build per-category 2nd-order absorption lookups.

    Returns:
        (category_lookups, fallback_lookup)
        category_lookups: {category: {(prev, curr): prob}}
        fallback_lookup: aggregate {(prev, curr): prob}
    """
    # Aggregate fallback
    fallback = build_order2_absorption_lookup(traces, horizon, smoothing)

    # Per-category
    categories: dict[str, list[Trace]] = {}
    for trace in traces:
        cat = trace.metadata.scenario_category
        categories.setdefault(cat, []).append(trace)

    category_lookups = {}
    for cat, cat_traces in sorted(categories.items()):
        category_lookups[cat] = build_order2_absorption_lookup(
            cat_traces, horizon, max(smoothing, 1e-6)
        )

    return category_lookups, fallback
