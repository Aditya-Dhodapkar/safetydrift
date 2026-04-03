"""Markov-based runtime monitor using pre-computed absorption probabilities.

The core contribution: a lookup-table monitor that predicts violations
before they happen, at sub-millisecond overhead per step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from safetydrift.core.safety_state import SafetyState
from safetydrift.monitor.base import BaseMonitor, MonitorVerdict
from safetydrift.traces.models import Step


class MarkovMonitor(BaseMonitor):
    """Lookup-table monitor using finite-horizon absorption probabilities."""

    def __init__(
        self,
        lookup: dict[int, float],
        threshold: float = 0.40,
        horizon: int = 5,
    ):
        self._lookup = lookup
        self._threshold = threshold
        self._horizon = horizon

    @classmethod
    def from_json(cls, path: str | Path, threshold: float = 0.40) -> MarkovMonitor:
        """Load from the absorption_lookup.json produced by Phase 3."""
        data = json.loads(Path(path).read_text())
        lookup = {int(k): float(v) for k, v in data["lookup"].items()}
        return cls(lookup=lookup, threshold=threshold, horizon=data.get("horizon", 5))

    def check_step(self, step: Step, current_state: SafetyState) -> MonitorVerdict:
        t0 = time.perf_counter_ns()
        prob = self._lookup.get(current_state.risk_level.value, 0.0)
        flagged = prob >= self._threshold
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return MonitorVerdict(
            should_flag=flagged,
            risk_score=prob,
            reason=f"P(VIOLATED in {self._horizon})={prob:.3f} {'≥' if flagged else '<'} {self._threshold}",
            latency_ms=elapsed_ms,
        )

    @property
    def name(self) -> str:
        return f"Markov (h={self._horizon}, t={self._threshold})"

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value


class Order2PerCategoryMarkovMonitor(BaseMonitor):
    """2nd-order category-aware monitor using (prev, curr) state pairs.

    Tracks the previous risk level and looks up absorption probability
    conditioned on both previous and current state.
    """

    def __init__(
        self,
        category_lookups: dict[str, dict[tuple[int, int], float]],
        fallback_lookup: dict[tuple[int, int], float],
        order1_fallback: dict[str, dict[int, float]] | None = None,
        threshold: float = 0.50,
        horizon: int = 5,
    ):
        self._category_lookups = category_lookups
        self._fallback = fallback_lookup
        self._order1_fallback = order1_fallback or {}
        self._threshold = threshold
        self._horizon = horizon
        self._current_category: str | None = None
        self._prev_risk: int | None = None

    def set_category(self, category: str) -> None:
        self._current_category = category

    def reset(self) -> None:
        self._current_category = None
        self._prev_risk = None

    def check_step(self, step: Step, current_state: SafetyState) -> MonitorVerdict:
        t0 = time.perf_counter_ns()
        curr = current_state.risk_level.value

        if self._prev_risk is not None:
            key = (self._prev_risk, curr)
            lookup = self._category_lookups.get(self._current_category, self._fallback)
            prob = lookup.get(key, 0.0)
        else:
            # First step: no previous state, fall back to 1st-order
            o1 = self._order1_fallback.get(self._current_category, {})
            prob = o1.get(curr, 0.0)

        self._prev_risk = curr

        flagged = prob >= self._threshold
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return MonitorVerdict(
            should_flag=flagged,
            risk_score=prob,
            reason=f"O2 P(VIOLATED|{self._current_category})={prob:.3f}",
            latency_ms=elapsed_ms,
        )

    @property
    def name(self) -> str:
        return f"Markov-Cat-O2 (h={self._horizon}, t={self._threshold})"

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value


class PerCategoryMarkovMonitor(BaseMonitor):
    """Category-aware monitor using per-category absorption lookup tables.

    Uses the scenario category to select the right transition model,
    giving much more precise risk scores than the aggregate model.
    """

    def __init__(
        self,
        category_lookups: dict[str, dict[int, float]],
        fallback_lookup: dict[int, float],
        threshold: float = 0.50,
        horizon: int = 5,
    ):
        self._category_lookups = category_lookups
        self._fallback = fallback_lookup
        self._threshold = threshold
        self._horizon = horizon
        self._current_category: str | None = None

    def set_category(self, category: str) -> None:
        """Set the current scenario category (called before replaying a trace)."""
        self._current_category = category

    @classmethod
    def from_json(cls, path: str | Path, threshold: float = 0.50) -> PerCategoryMarkovMonitor:
        """Load from a per-category lookup JSON file."""
        data = json.loads(Path(path).read_text())
        cat_lookups = {}
        for cat, lookup in data.get("category_lookups", {}).items():
            cat_lookups[cat] = {int(k): float(v) for k, v in lookup.items()}
        fallback = {int(k): float(v) for k, v in data.get("fallback", {}).items()}
        return cls(
            category_lookups=cat_lookups,
            fallback_lookup=fallback,
            threshold=threshold,
            horizon=data.get("horizon", 5),
        )

    def check_step(self, step: Step, current_state: SafetyState) -> MonitorVerdict:
        t0 = time.perf_counter_ns()
        lookup = self._category_lookups.get(self._current_category, self._fallback)
        prob = lookup.get(current_state.risk_level.value, 0.0)
        flagged = prob >= self._threshold
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return MonitorVerdict(
            should_flag=flagged,
            risk_score=prob,
            reason=f"P(VIOLATED|{self._current_category})={prob:.3f} {'≥' if flagged else '<'} {self._threshold}",
            latency_ms=elapsed_ms,
        )

    @property
    def name(self) -> str:
        return f"Markov-Cat (h={self._horizon}, t={self._threshold})"

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def reset(self) -> None:
        self._current_category = None
