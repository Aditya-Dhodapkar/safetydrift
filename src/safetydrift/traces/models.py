"""Execution trace data models.

These are the data contracts between Phase 1 (collection) and Phases 2-5 (analysis).
All models are Pydantic v2 BaseModels for validation, serialization, and type safety.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from safetydrift.core.enums import RiskLevel
from safetydrift.core.safety_state import SafetyState


class ToolCall(BaseModel):
    """A single tool invocation by the agent."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None


class ActionRecord(BaseModel):
    """What the agent did on a given step."""

    action_type: str  # "tool_call", "message", "code_exec"
    tool_call: ToolCall | None = None
    raw_text: str | None = None


class ObservationRecord(BaseModel):
    """What the agent observed after taking an action."""

    content: str
    truncated: bool = False


class StepMetadata(BaseModel):
    """Optional metadata for a single execution step."""

    timestamp: datetime | None = None
    llm_model: str | None = None
    latency_ms: float | None = None
    token_count: int | None = None


class Step(BaseModel):
    """A single step in an agent execution trace."""

    step_index: int
    action: ActionRecord
    observation: ObservationRecord
    safety_state_before: SafetyState
    safety_state_after: SafetyState
    metadata: StepMetadata = Field(default_factory=StepMetadata)

    # Phase 2 (classifier) fills these in:
    label_source: str | None = None  # "rule", "llm_judge", "human"
    label_confidence: float | None = None
    is_violation_step: bool = False


class TraceMetadata(BaseModel):
    """Metadata for an entire execution trace."""

    scenario_id: str
    scenario_category: str
    agent_framework: str
    llm_model: str
    run_id: str
    seed: int | None = None
    timestamp: datetime
    num_steps: int
    reached_violation: bool


class Trace(BaseModel):
    """A complete agent execution trace: metadata + ordered list of steps."""

    metadata: TraceMetadata
    steps: list[Step]

    @property
    def safety_trajectory(self) -> list[SafetyState]:
        """Sequence of safety states after each step."""
        return [step.safety_state_after for step in self.steps]

    @property
    def risk_trajectory(self) -> list[RiskLevel]:
        """Sequence of risk levels after each step."""
        return [step.safety_state_after.risk_level for step in self.steps]

    @property
    def state_index_trajectory(self) -> list[int]:
        """Sequence of state indices after each step (for Markov matrix use)."""
        return [step.safety_state_after.to_index() for step in self.steps]

    @property
    def violation_step_index(self) -> int | None:
        """Index of the first step that results in a VIOLATED state, or None."""
        for step in self.steps:
            if step.safety_state_after.risk_level == RiskLevel.VIOLATED:
                return step.step_index
        return None
