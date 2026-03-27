"""Pydantic models for the classifier pipeline.

StepClassification is the structured output schema used by both the deterministic
rules (Layer 1) and the LLM judge (Layer 2, via instructor).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.risk_synthesis import synthesize_fast
from safetydrift.core.safety_state import SafetyState


class StepClassification(BaseModel):
    """Classification of a single step's safety impact.

    Also used as the response_model for instructor (LLM structured output).
    """

    data_exposure: DataExposure
    tool_escalation: ToolEscalation
    reversibility: Reversibility
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @property
    def risk_level(self) -> RiskLevel:
        """Derived risk level from the three dimensions."""
        return synthesize_fast(self.data_exposure, self.tool_escalation, self.reversibility)

    def to_safety_state(self) -> SafetyState:
        """Convert to a SafetyState (risk_level auto-computed)."""
        return SafetyState(
            data_exposure=self.data_exposure,
            tool_escalation=self.tool_escalation,
            reversibility=self.reversibility,
        )


class StepContext(BaseModel):
    """Context bundle sent to the LLM judge for classifying a step."""

    scenario_id: str
    scenario_description: str
    step_index: int
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    safety_state_before: SafetyState
    previous_steps_summary: list[str] = Field(default_factory=list)
    file_sensitivities: dict[str, int] = Field(default_factory=dict)


class ClassifierResult(BaseModel):
    """Full classification result for a single step."""

    step_index: int
    classification: StepClassification
    label_source: Literal["rule", "llm_judge", "human"]
    is_violation: bool
    discrepancy: bool = False  # True if classification differs from sandbox's state

    @property
    def risk_level(self) -> RiskLevel:
        return self.classification.risk_level
