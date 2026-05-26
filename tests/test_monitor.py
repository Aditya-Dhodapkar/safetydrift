"""Tests for the runtime monitor module."""

import json
import time

import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState
from safetydrift.harness.llm_backends import MockBackend
from safetydrift.monitor.base import MonitorVerdict
from safetydrift.monitor.baselines import (
    CategoryPriorMonitor,
    KeywordMonitor,
    LLMJudgeMonitor,
    NoMonitor,
    Pro2GuardStyleMonitor,
    extract_pro2guard_bits,
)
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


class TestPro2GuardStyleMonitor:
    def test_extract_bits_safe_state(self):
        state = SafetyState.initial()
        step = _make_step_with_tool("read_file", state)
        assert extract_pro2guard_bits(step) == 0

    def test_extract_bits_sensitive_and_network(self):
        state = SafetyState.initial().transition(
            new_exposure=DataExposure.SENSITIVE,
            new_escalation=ToolEscalation.NETWORK,
        )
        step = _make_step_with_tool("send_email", state)
        bits = extract_pro2guard_bits(step)
        # bit 0 (sensitive) and bit 4 (network) should be set
        assert bits & (1 << 0)
        assert bits & (1 << 4)
        # bit 1 (credentials) should NOT be set
        assert not (bits & (1 << 1))

    def test_cumulative_bits_across_steps(self):
        # Lookup: flag any state that has both sensitive (bit 0) and network (bit 4) set.
        def has_sens_and_net(i: int) -> bool:
            return bool(i & (1 << 0)) and bool(i & (1 << 4))

        lookup = {i: (0.9 if has_sens_and_net(i) else 0.0) for i in range(64)}
        monitor = Pro2GuardStyleMonitor(lookup, threshold=0.5)

        sens_state = SafetyState.initial().transition(new_exposure=DataExposure.SENSITIVE)
        net_state = sens_state.transition(new_escalation=ToolEscalation.NETWORK)

        # Step 1: sensitive only — no network bit yet — no flag
        step1 = _make_step_with_tool("read_file", sens_state)
        v1 = monitor.check_step(step1, sens_state)
        assert v1.should_flag is False

        # Step 2: network — cum bits now includes both — flag
        step2 = _make_step_with_tool("send_email", net_state)
        v2 = monitor.check_step(step2, net_state)
        assert v2.should_flag is True
        assert v2.risk_score == pytest.approx(0.9)

    def test_reset_clears_history(self):
        sens_and_net = (1 << 0) | (1 << 4)
        lookup = {i: (0.9 if i == sens_and_net else 0.0) for i in range(64)}
        monitor = Pro2GuardStyleMonitor(lookup, threshold=0.5)

        # Drive monitor into flagged state
        net_state = SafetyState.initial().transition(
            new_exposure=DataExposure.SENSITIVE,
            new_escalation=ToolEscalation.NETWORK,
        )
        monitor.check_step(_make_step_with_tool("send_email", net_state), net_state)

        # Reset, then a safe step should not flag
        monitor.reset()
        safe_state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", safe_state), safe_state)
        assert verdict.should_flag is False

    def test_from_json_roundtrip(self, tmp_path):
        data = {
            "horizon": 5,
            "lookup": {str(i): (0.9 if i == 17 else 0.1) for i in range(64)},
        }
        path = tmp_path / "p2g_lookup.json"
        path.write_text(json.dumps(data))
        monitor = Pro2GuardStyleMonitor.from_json(path, threshold=0.5)
        assert monitor.threshold == 0.5

    def test_threshold_mutable(self):
        monitor = Pro2GuardStyleMonitor({i: 0.0 for i in range(64)}, threshold=0.50)
        monitor.threshold = 0.30
        assert monitor.threshold == 0.30


class TestCategoryPriorMonitor:
    """Confound-check monitor: flags by scenario category, ignores trajectory."""

    def test_flags_in_high_rate_category(self):
        monitor = CategoryPriorMonitor()
        monitor.set_category("research_comms")
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.should_flag is True
        assert verdict.risk_score == 1.0

    def test_flags_data_handling(self):
        monitor = CategoryPriorMonitor()
        monitor.set_category("data_handling")
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.should_flag is True

    def test_does_not_flag_low_rate_category(self):
        monitor = CategoryPriorMonitor()
        monitor.set_category("sysadmin")
        violating_state = SafetyState.initial().transition(
            new_exposure=DataExposure.CREDENTIALS,
            new_escalation=ToolEscalation.NETWORK,
            new_reversibility=Reversibility.IRREVERSIBLE,
        )
        # Even at a VIOLATED state, the category-prior monitor must not flag
        # in low-rate categories — this is the whole point of the baseline.
        verdict = monitor.check_step(_make_step_with_tool("http_request", violating_state), violating_state)
        assert verdict.should_flag is False
        assert verdict.risk_score == 0.0

    def test_does_not_flag_code_debugging(self):
        monitor = CategoryPriorMonitor()
        monitor.set_category("code_debugging")
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.should_flag is False

    def test_ignores_trajectory_within_category(self):
        """Confirms it is a *prior*: identical decision regardless of step state."""
        monitor = CategoryPriorMonitor()
        monitor.set_category("research_comms")
        safe = SafetyState.initial()
        risky = safe.transition(
            new_exposure=DataExposure.SENSITIVE,
            new_escalation=ToolEscalation.FILE_WRITE,
        )
        v1 = monitor.check_step(_make_step_with_tool("read_file", safe), safe)
        v2 = monitor.check_step(_make_step_with_tool("send_email", risky), risky)
        assert v1.should_flag == v2.should_flag

    def test_reset_clears_category(self):
        monitor = CategoryPriorMonitor()
        monitor.set_category("research_comms")
        monitor.reset()
        # After reset, no category → not in high-rate set → no flag
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.should_flag is False

    def test_custom_high_rate_set(self):
        monitor = CategoryPriorMonitor(high_rate_categories={"sysadmin"})
        monitor.set_category("sysadmin")
        state = SafetyState.initial()
        verdict = monitor.check_step(_make_step_with_tool("read_file", state), state)
        assert verdict.should_flag is True

    def test_name(self):
        assert CategoryPriorMonitor().name == "Category Prior"
