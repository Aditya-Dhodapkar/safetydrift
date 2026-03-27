"""Frozen, immutable safety state representation.

SafetyState is the core data structure of the entire project. It captures where an agent
sits on the safety spectrum at any point during execution. Key properties:

- Frozen (immutable): transitions return new instances
- risk_level is always auto-computed from the other three dimensions
- data_exposure and tool_escalation are monotonically non-decreasing via transition()
- VIOLATED is an absorbing state (transition() returns self)
- to_index()/from_index() provide dense [0, 59] indexing for Markov matrices
"""

from __future__ import annotations

from itertools import product
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.risk_synthesis import synthesize_fast


class SafetyState(BaseModel):
    """Immutable safety state of an agent at a point in time."""

    model_config = ConfigDict(frozen=True)

    data_exposure: DataExposure
    tool_escalation: ToolEscalation
    reversibility: Reversibility
    risk_level: RiskLevel = RiskLevel.SAFE  # overwritten by validator

    @model_validator(mode="after")
    def _sync_risk_level(self) -> Self:
        """Ensure risk_level always matches the synthesis function."""
        expected = synthesize_fast(self.data_exposure, self.tool_escalation, self.reversibility)
        if self.risk_level != expected:
            # Bypass frozen model to correct risk_level
            object.__setattr__(self, "risk_level", expected)
        return self

    @staticmethod
    def initial() -> SafetyState:
        """Starting state: all at minimum → SAFE."""
        return SafetyState(
            data_exposure=DataExposure.NONE,
            tool_escalation=ToolEscalation.READ_ONLY,
            reversibility=Reversibility.FULLY_REVERSIBLE,
        )

    def transition(
        self,
        new_exposure: DataExposure | None = None,
        new_escalation: ToolEscalation | None = None,
        new_reversibility: Reversibility | None = None,
    ) -> SafetyState:
        """Return a new state with monotonicity enforced.

        - data_exposure: max(current, new) — can only increase
        - tool_escalation: max(current, new) — can only increase
        - reversibility: takes new value directly (not monotonic)
        - If current state is VIOLATED, returns self (absorbing)
        """
        if self.is_absorbing:
            return self

        d = max(self.data_exposure, new_exposure) if new_exposure is not None else self.data_exposure
        t = max(self.tool_escalation, new_escalation) if new_escalation is not None else self.tool_escalation
        r = new_reversibility if new_reversibility is not None else self.reversibility

        return SafetyState(
            data_exposure=d,
            tool_escalation=t,
            reversibility=r,
        )

    @property
    def is_absorbing(self) -> bool:
        """True if this is a VIOLATED state (terminal — cannot leave)."""
        return self.risk_level == RiskLevel.VIOLATED

    def to_index(self) -> int:
        """Deterministic mapping to [0, state_space_size()-1] for Markov matrix indexing.

        Formula: d * (|T| * |R|) + t * |R| + r = d * 12 + t * 3 + r
        """
        return (
            self.data_exposure.value * len(ToolEscalation) * len(Reversibility)
            + self.tool_escalation.value * len(Reversibility)
            + self.reversibility.value
        )

    @staticmethod
    def from_index(idx: int) -> SafetyState:
        """Inverse of to_index. Reconstructs SafetyState from a dense index."""
        n_t = len(ToolEscalation)
        n_r = len(Reversibility)

        d, remainder = divmod(idx, n_t * n_r)
        t, r = divmod(remainder, n_r)

        return SafetyState(
            data_exposure=DataExposure(d),
            tool_escalation=ToolEscalation(t),
            reversibility=Reversibility(r),
        )

    @staticmethod
    def enumerate_all() -> list[SafetyState]:
        """Return all valid safety states (60 total)."""
        states = []
        for d, t, r in product(DataExposure, ToolEscalation, Reversibility):
            states.append(SafetyState(data_exposure=d, tool_escalation=t, reversibility=r))
        return states

    @staticmethod
    def state_space_size() -> int:
        """Total number of distinct safety states."""
        return len(DataExposure) * len(ToolEscalation) * len(Reversibility)

    def __repr__(self) -> str:
        return (
            f"SafetyState(d={self.data_exposure.name}, t={self.tool_escalation.name}, "
            f"r={self.reversibility.name} → {self.risk_level.name})"
        )
