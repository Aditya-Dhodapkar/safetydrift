"""Shared test fixtures for SafetyDrift tests."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from safetydrift.core.enums import DataExposure, Reversibility, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.traces.models import (
    ActionRecord,
    ObservationRecord,
    Step,
    StepMetadata,
    ToolCall,
    Trace,
    TraceMetadata,
)


def make_step(
    index: int,
    state_before: SafetyState,
    state_after: SafetyState,
    tool_name: str = "read_file",
    action_type: str = "tool_call",
) -> Step:
    """Helper to create a Step with minimal boilerplate."""
    return Step(
        step_index=index,
        action=ActionRecord(
            action_type=action_type,
            tool_call=ToolCall(tool_name=tool_name, arguments={"path": f"/file_{index}.txt"}),
            raw_text=f"Step {index} action",
        ),
        observation=ObservationRecord(content=f"Result of step {index}"),
        safety_state_before=state_before,
        safety_state_after=state_after,
        metadata=StepMetadata(llm_model="mock"),
    )


def make_trace(
    steps: list[Step],
    scenario_id: str = "test_scenario",
    category: str = "data_handling",
    reached_violation: bool = False,
) -> Trace:
    """Helper to create a Trace from a list of Steps."""
    return Trace(
        metadata=TraceMetadata(
            scenario_id=scenario_id,
            scenario_category=category,
            agent_framework="mock",
            llm_model="mock",
            run_id=str(uuid4())[:8],
            seed=42,
            timestamp=datetime.now(timezone.utc),
            num_steps=len(steps),
            reached_violation=reached_violation,
        ),
        steps=steps,
    )


@pytest.fixture
def safe_trace() -> Trace:
    """A 3-step trace that stays in safe territory."""
    s0 = SafetyState.initial()
    s1 = s0.transition(new_exposure=DataExposure.PUBLIC)
    s2 = s1.transition(new_escalation=ToolEscalation.FILE_WRITE)

    steps = [
        make_step(0, s0, s1, tool_name="read_file"),
        make_step(1, s1, s1, tool_name="read_file"),
        make_step(2, s1, s2, tool_name="write_file"),
    ]
    return make_trace(steps, reached_violation=False)


@pytest.fixture
def violating_trace() -> Trace:
    """A 4-step trace that drifts into a violation."""
    s0 = SafetyState.initial()
    s1 = s0.transition(new_exposure=DataExposure.INTERNAL)
    s2 = s1.transition(new_exposure=DataExposure.CREDENTIALS)
    s3 = s2.transition(
        new_escalation=ToolEscalation.NETWORK,
        new_reversibility=Reversibility.IRREVERSIBLE,
    )

    steps = [
        make_step(0, s0, s1, tool_name="read_file"),
        make_step(1, s1, s2, tool_name="read_file"),
        make_step(2, s2, s2, tool_name="execute_code"),
        make_step(3, s2, s3, tool_name="send_email"),
    ]
    return make_trace(steps, reached_violation=True)
