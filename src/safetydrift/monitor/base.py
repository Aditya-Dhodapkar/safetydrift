"""Abstract base for all runtime monitors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from safetydrift.core.safety_state import SafetyState
from safetydrift.traces.models import Step


@dataclass
class MonitorVerdict:
    """Result of a single monitor check."""

    should_flag: bool
    risk_score: float  # 0.0-1.0
    reason: str = ""
    latency_ms: float = 0.0


class BaseMonitor(ABC):
    """Protocol for all runtime monitors."""

    @abstractmethod
    def check_step(self, step: Step, current_state: SafetyState) -> MonitorVerdict:
        """Evaluate a single step synchronously."""
        ...

    async def check_step_async(self, step: Step, current_state: SafetyState) -> MonitorVerdict:
        """Async variant. Default delegates to sync."""
        return self.check_step(step, current_state)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for results tables."""
        ...

    def reset(self) -> None:
        """Reset internal state between traces. Default: no-op."""
        pass
