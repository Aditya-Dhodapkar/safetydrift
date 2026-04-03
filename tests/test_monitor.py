"""Tests for the runtime monitor module."""

import json
import time

import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.harness.llm_backends import MockBackend
from safetydrift.monitor.base import MonitorVerdict
from safetydrift.monitor.baselines import KeywordMonitor, LLMJudgeMonitor, NoMonitor
from safetydrift.monitor.markov_monitor import MarkovMonitor
from safetydrift.traces.models import ActionRecord, ObservationRecord, Step, ToolCall

SAMPLE_LOOKUP = {0: 0.248, 1: 0.447, 2: 0.259, 3: 0.230, 4: 1.0}


def _make_step_with_tool(tool_name: str, state: SafetyState) -> Step:
    return Step(
        step_index=0,
        action=ActionRecord(
            action_type="tool_call",
            tool_call=ToolCall(tool_name=tool_name, arguments={"path": "/data"}),
        ),
        observation=ObservationRecord(content="result"),
        safety_state_before=state,
        safety_state_after=state,
    )


def _make_text_step(state: SafetyState) -> Step:
    return Step(
        step_index=0,
        action=ActionRecord(action_type="message", raw_text="Hello"),
        observation=ObservationRecord(content=""),
        safety_state_before=state,
        safety_state_after=state,
    )


class TestMarkovMonitor:
    def test_flags_above_threshold(self):
        monitor = MarkovMonitor(SAMPLE_LOOKUP, threshold=0.40)
        state = SafetyState.initial().transition(
            new_exposure=DataExposure.SENSITIVE,
            new_escalation=ToolEscalation.FILE_WRITE,
        )
        # MILD state has prob=0.447 > 0.40
        step = _make_step_with_tool("read_file", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is True
        assert verdict.risk_score == pytest.approx(0.447, abs=0.01)

    def test_does_not_flag_below_threshold(self):
        monitor = MarkovMonitor(SAMPLE_LOOKUP, threshold=0.50)
        state = SafetyState.initial()  # SAFE, prob=0.248
        step = _make_step_with_tool("read_file", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is False

    def test_flags_violated(self):
        monitor = MarkovMonitor(SAMPLE_LOOKUP, threshold=0.40)
        state = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        step = _make_step_with_tool("send_email", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is True
        assert verdict.risk_score == 1.0

    def test_from_json(self, tmp_path):
        data = {"horizon": 5, "lookup": {str(k): v for k, v in SAMPLE_LOOKUP.items()}}
        path = tmp_path / "lookup.json"
        path.write_text(json.dumps(data))
        monitor = MarkovMonitor.from_json(path, threshold=0.40)
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.risk_score == pytest.approx(0.248, abs=0.01)

    def test_sub_millisecond_latency(self):
        monitor = MarkovMonitor(SAMPLE_LOOKUP, threshold=0.40)
        state = SafetyState.initial()
        step = _make_step_with_tool("read_file", state)
        latencies = []
        for _ in range(1000):
            verdict = monitor.check_step(step, state)
            latencies.append(verdict.latency_ms)
        mean_ms = sum(latencies) / len(latencies)
        assert mean_ms < 1.0, f"Mean latency {mean_ms:.4f}ms exceeds 1ms"

    def test_threshold_mutable(self):
        monitor = MarkovMonitor(SAMPLE_LOOKUP, threshold=0.80)
        assert monitor.threshold == 0.80
        monitor.threshold = 0.30
        assert monitor.threshold == 0.30


class TestKeywordMonitor:
    def test_flags_send_email(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial()
        step = _make_step_with_tool("send_email", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is True

    def test_flags_http_request(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial()
        step = _make_step_with_tool("http_request", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is True

    def test_flags_exec_with_sensitive(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial().transition(new_exposure=DataExposure.SENSITIVE)
        step = _make_step_with_tool("execute_code", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is True

    def test_does_not_flag_exec_with_public(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial().transition(new_exposure=DataExposure.PUBLIC)
        step = _make_step_with_tool("execute_code", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is False

    def test_does_not_flag_read(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial()
        step = _make_step_with_tool("read_file", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is False

    def test_does_not_flag_text(self):
        monitor = KeywordMonitor()
        state = SafetyState.initial()
        step = _make_text_step(state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is False


class TestNoMonitor:
    def test_never_flags(self):
        monitor = NoMonitor()
        state = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        step = _make_step_with_tool("send_email", state)
        verdict = monitor.check_step(step, state)
        assert verdict.should_flag is False


class TestLLMJudgeMonitor:
    @pytest.mark.asyncio
    async def test_flags_on_yes(self):
        backend = MockBackend()
        backend.add_final_response("YES")
        monitor = LLMJudgeMonitor(backend)
        state = SafetyState.initial()
        step = _make_step_with_tool("send_email", state)
        verdict = await monitor.check_step_async(step, state)
        assert verdict.should_flag is True

    @pytest.mark.asyncio
    async def test_does_not_flag_on_no(self):
        backend = MockBackend()
        backend.add_final_response("NO")
        monitor = LLMJudgeMonitor(backend)
        state = SafetyState.initial()
        step = _make_step_with_tool("read_file", state)
        verdict = await monitor.check_step_async(step, state)
        assert verdict.should_flag is False
