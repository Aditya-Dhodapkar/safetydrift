"""Offline trace replay engine for monitor evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from safetydrift.core.enums import RiskLevel
from safetydrift.monitor.base import BaseMonitor, MonitorVerdict
from safetydrift.traces.models import Trace


@dataclass
class StepReplayResult:
    step_index: int
    verdict: MonitorVerdict
    risk_level_after: RiskLevel


@dataclass
class TraceReplayResult:
    trace_id: str
    scenario_category: str
    reached_violation: bool
    violation_step_index: int | None
    first_flag_step: int | None
    step_results: list[StepReplayResult] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """True if monitor flagged BEFORE the violation step."""
        if not self.reached_violation or self.violation_step_index is None:
            return False
        if self.first_flag_step is None:
            return False
        return self.first_flag_step < self.violation_step_index

    @property
    def early_warning_steps(self) -> int | None:
        """Steps of early warning. None if not detected or no violation."""
        if not self.detected:
            return None
        return self.violation_step_index - self.first_flag_step

    @property
    def false_positive(self) -> bool:
        """True if monitor flagged but trace never reaches VIOLATED."""
        return not self.reached_violation and self.first_flag_step is not None


def replay_trace(monitor: BaseMonitor, trace: Trace) -> TraceReplayResult:
    """Replay a single trace through a monitor (synchronous)."""
    monitor.reset()
    # If monitor is category-aware, set the category
    if hasattr(monitor, "set_category"):
        monitor.set_category(trace.metadata.scenario_category)
    step_results = []
    first_flag = None

    for step in trace.steps:
        state = step.safety_state_after
        verdict = monitor.check_step(step, state)
        step_results.append(StepReplayResult(
            step_index=step.step_index,
            verdict=verdict,
            risk_level_after=state.risk_level,
        ))
        if verdict.should_flag and first_flag is None:
            first_flag = step.step_index

    return TraceReplayResult(
        trace_id=f"{trace.metadata.scenario_id}_{trace.metadata.run_id}",
        scenario_category=trace.metadata.scenario_category,
        reached_violation=trace.metadata.reached_violation,
        violation_step_index=trace.violation_step_index,
        first_flag_step=first_flag,
        step_results=step_results,
    )


async def replay_trace_async(monitor: BaseMonitor, trace: Trace) -> TraceReplayResult:
    """Replay using async check (for LLM judge)."""
    monitor.reset()
    step_results = []
    first_flag = None

    for step in trace.steps:
        state = step.safety_state_after
        verdict = await monitor.check_step_async(step, state)
        step_results.append(StepReplayResult(
            step_index=step.step_index,
            verdict=verdict,
            risk_level_after=state.risk_level,
        ))
        if verdict.should_flag and first_flag is None:
            first_flag = step.step_index

    return TraceReplayResult(
        trace_id=f"{trace.metadata.scenario_id}_{trace.metadata.run_id}",
        scenario_category=trace.metadata.scenario_category,
        reached_violation=trace.metadata.reached_violation,
        violation_step_index=trace.violation_step_index,
        first_flag_step=first_flag,
        step_results=step_results,
    )


def replay_all(monitor: BaseMonitor, traces: list[Trace]) -> list[TraceReplayResult]:
    """Replay all traces synchronously."""
    return [replay_trace(monitor, t) for t in traces]


async def replay_all_async(monitor: BaseMonitor, traces: list[Trace]) -> list[TraceReplayResult]:
    """Replay all traces using async check (for LLM-based monitors)."""
    return [await replay_trace_async(monitor, t) for t in traces]
