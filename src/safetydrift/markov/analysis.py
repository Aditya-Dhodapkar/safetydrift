"""Points of no return and per-category analysis.

Identifies states where finite-horizon violation probability exceeds a threshold,
and compares absorption dynamics across scenario categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.markov.absorption import AbsorptionResult, analyze_absorption
from safetydrift.markov.estimation import TransitionMatrix, estimate_transition_matrix
from safetydrift.traces.models import Trace


@dataclass
class PointOfNoReturn:
    """A state identified as a point of no return."""

    state_index: int
    risk_level_name: str
    finite_horizon_prob: float  # P(VIOLATED within N steps)
    mean_steps_to_violation: float
    horizon: int


@dataclass
class PointsOfNoReturnResult:
    """Analysis of points of no return."""

    threshold: float
    horizon: int
    points_of_no_return: list[PointOfNoReturn]
    cliff_description: str
    zones: dict[str, list[int]]  # "safe", "transition", "danger" → state indices
    all_states_sorted: list[tuple[int, float]]  # (state_index, finite_horizon_prob)


@dataclass
class PerCategoryResult:
    """Results for a single scenario category."""

    category: str
    num_traces: int
    num_violated: int
    violation_rate: float
    transition_matrix: TransitionMatrix
    absorption_result: AbsorptionResult
    points_of_no_return: PointsOfNoReturnResult


def find_points_of_no_return(
    absorption_result: AbsorptionResult,
    threshold: float = 0.85,
    horizon: int = 5,
) -> PointsOfNoReturnResult:
    """Identify states where finite-horizon violation probability exceeds threshold.

    Uses finite-horizon probabilities (not infinite-horizon, which is always 1.0
    due to monotonicity).
    """
    if horizon not in absorption_result.finite_horizon_probs:
        raise ValueError(f"Horizon {horizon} not computed. Available: {list(absorption_result.finite_horizon_probs.keys())}")

    fh_probs = absorption_result.finite_horizon_probs[horizon]

    # Sort states by finite-horizon probability
    all_sorted = sorted(
        [(i, float(fh_probs[i])) for i in range(absorption_result.state_size)],
        key=lambda x: x[1],
        reverse=True,
    )

    # Find points of no return
    ponr = []
    for idx, prob in all_sorted:
        if idx in absorption_result.absorbing_indices:
            continue  # skip already-violated states
        if prob >= threshold:
            if absorption_result.granularity == "coarse":
                name = RiskLevel(idx).name
            else:
                state = SafetyState.from_index(idx)
                name = f"d={state.data_exposure.name},t={state.tool_escalation.name},r={state.reversibility.name}"

            passage = absorption_result.state_passage_map.get(idx, 0.0)
            ponr.append(PointOfNoReturn(
                state_index=idx,
                risk_level_name=name,
                finite_horizon_prob=prob,
                mean_steps_to_violation=passage,
                horizon=horizon,
            ))

    # Characterize zones
    safe_zone = [i for i, p in all_sorted if p < 0.30 and i not in absorption_result.absorbing_indices]
    transition_zone = [i for i, p in all_sorted if 0.30 <= p < threshold and i not in absorption_result.absorbing_indices]
    danger_zone = [i for i, p in all_sorted if p >= threshold and i not in absorption_result.absorbing_indices]

    cliff_desc = _describe_cliff(safe_zone, transition_zone, danger_zone, absorption_result.granularity)

    return PointsOfNoReturnResult(
        threshold=threshold,
        horizon=horizon,
        points_of_no_return=ponr,
        cliff_description=cliff_desc,
        zones={"safe": safe_zone, "transition": transition_zone, "danger": danger_zone},
        all_states_sorted=all_sorted,
    )


def _describe_cliff(
    safe: list[int],
    transition: list[int],
    danger: list[int],
    granularity: str,
) -> str:
    """Generate a human-readable cliff characterization."""
    if granularity == "coarse":
        safe_names = [RiskLevel(i).name for i in safe]
        trans_names = [RiskLevel(i).name for i in transition]
        danger_names = [RiskLevel(i).name for i in danger]
    else:
        safe_names = [str(i) for i in safe[:5]]
        trans_names = [str(i) for i in transition[:5]]
        danger_names = [str(i) for i in danger[:5]]

    parts = []
    if danger_names:
        parts.append(f"Danger zone (>{85}%): {', '.join(danger_names)}")
    if trans_names:
        parts.append(f"Transition zone (30-85%): {', '.join(trans_names)}")
    if safe_names:
        parts.append(f"Safe zone (<30%): {', '.join(safe_names)}")

    return "; ".join(parts) if parts else "No clear cliff boundary found"


def analyze_per_category(
    traces: list[Trace],
    granularity: Literal["coarse", "fine"] = "coarse",
    smoothing: float = 0.0,
    threshold: float = 0.85,
    horizon: int = 5,
    horizons: list[int] | None = None,
) -> list[PerCategoryResult]:
    """Run full analysis separately for each scenario category."""
    if horizons is None:
        horizons = [1, 3, 5, 10]

    categories: dict[str, list[Trace]] = {}
    for trace in traces:
        cat = trace.metadata.scenario_category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(trace)

    results = []
    for cat, cat_traces in sorted(categories.items()):
        # Per-category data is sparser — use at least minimal smoothing to avoid singular matrices
        cat_smoothing = max(smoothing, 1e-6)
        tm = estimate_transition_matrix(cat_traces, granularity, cat_smoothing)
        ar = analyze_absorption(tm, horizons)

        num_violated = sum(1 for t in cat_traces if t.metadata.reached_violation)

        ponr = find_points_of_no_return(ar, threshold, horizon)

        results.append(PerCategoryResult(
            category=cat,
            num_traces=len(cat_traces),
            num_violated=num_violated,
            violation_rate=num_violated / len(cat_traces) if cat_traces else 0.0,
            transition_matrix=tm,
            absorption_result=ar,
            points_of_no_return=ponr,
        ))

    return results


def compare_category_absorption(
    per_category_results: list[PerCategoryResult],
    horizon: int = 5,
) -> dict[str, Any]:
    """Compare finite-horizon violation probabilities across categories."""
    comparison = {}
    for pcr in per_category_results:
        fh = pcr.absorption_result.finite_horizon_probs.get(horizon)
        if fh is not None:
            if pcr.absorption_result.granularity == "coarse":
                state_probs = {RiskLevel(i).name: float(fh[i]) for i in range(min(5, len(fh)))}
            else:
                state_probs = {str(i): float(fh[i]) for i in range(len(fh)) if fh[i] > 0.01}
            comparison[pcr.category] = {
                "violation_rate": pcr.violation_rate,
                "finite_horizon_probs": state_probs,
                "points_of_no_return": [p.risk_level_name for p in pcr.points_of_no_return.points_of_no_return],
            }

    return comparison
