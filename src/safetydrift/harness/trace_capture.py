"""Trace capture middleware.

Sits between the agent runner and sandbox, recording every step into a Trace.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from safetydrift.core.enums import DataExposure, Reversibility, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.harness.base import LLMResponse, SafetyDelta, ToolExecutor, ToolResult
from safetydrift.scenarios.schema import ScenarioConfig
from safetydrift.traces.models import (
    ActionRecord,
    ObservationRecord,
    Step,
    StepMetadata,
    ToolCall,
    Trace,
    TraceMetadata,
)


class TraceCapture:
    """Records agent execution steps into a structured Trace."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        agent_framework: str,
        llm_model: str,
        seed: int | None = None,
    ):
        self._scenario = scenario
        self._agent_framework = agent_framework
        self._llm_model = llm_model
        self._seed = seed
        self._run_id = str(uuid4())[:8]
        self._steps: list[Step] = []
        self._current_state = SafetyState.initial()
        self._start_time = datetime.now(timezone.utc)

    @property
    def current_state(self) -> SafetyState:
        return self._current_state

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def record_step(
        self,
        llm_response: LLMResponse,
        tool_result: ToolResult | None = None,
    ) -> Step:
        """Record a single agent step.

        Args:
            llm_response: The LLM's output for this step.
            tool_result: Result of tool execution, if any.

        Returns:
            The recorded Step.
        """
        state_before = self._current_state

        # Determine action
        if llm_response.tool_calls:
            tc = llm_response.tool_calls[0]  # primary tool call
            action = ActionRecord(
                action_type="tool_call",
                tool_call=ToolCall(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    result=tool_result.output if tool_result else None,
                ),
                raw_text=llm_response.content or None,
            )
        else:
            action = ActionRecord(
                action_type="message",
                raw_text=llm_response.content,
            )

        # Determine observation
        if tool_result:
            observation = ObservationRecord(
                content=tool_result.output,
                truncated=len(tool_result.output) > 10000,
            )
        else:
            observation = ObservationRecord(content=llm_response.content)

        # Compute state transition from safety delta
        if tool_result and tool_result.safety_delta:
            state_after = self._apply_delta(state_before, tool_result.safety_delta)
        else:
            state_after = state_before

        self._current_state = state_after

        step = Step(
            step_index=len(self._steps),
            action=action,
            observation=observation,
            safety_state_before=state_before,
            safety_state_after=state_after,
            metadata=StepMetadata(
                timestamp=datetime.now(timezone.utc),
                llm_model=llm_response.model or self._llm_model,
                token_count=llm_response.input_tokens + llm_response.output_tokens,
            ),
        )
        self._steps.append(step)
        return step

    def finalize(self) -> Trace:
        """Build the final Trace from all recorded steps."""
        reached_violation = any(
            s.safety_state_after.is_absorbing for s in self._steps
        )

        return Trace(
            metadata=TraceMetadata(
                scenario_id=self._scenario.id,
                scenario_category=self._scenario.category,
                agent_framework=self._agent_framework,
                llm_model=self._llm_model,
                run_id=self._run_id,
                seed=self._seed,
                timestamp=self._start_time,
                num_steps=len(self._steps),
                reached_violation=reached_violation,
            ),
            steps=self._steps,
        )

    @staticmethod
    def _apply_delta(state: SafetyState, delta: SafetyDelta) -> SafetyState:
        """Apply a safety delta to produce a new state."""
        new_exposure = DataExposure(delta.data_exposure) if delta.data_exposure is not None else None
        new_escalation = ToolEscalation(delta.tool_escalation) if delta.tool_escalation is not None else None
        new_reversibility = Reversibility(delta.reversibility) if delta.reversibility is not None else None

        return state.transition(
            new_exposure=new_exposure,
            new_escalation=new_escalation,
            new_reversibility=new_reversibility,
        )
