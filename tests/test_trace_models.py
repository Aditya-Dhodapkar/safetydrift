"""Tests for trace data models."""

from safetydrift.core.enums import RiskLevel
from safetydrift.traces.models import Trace


class TestTraceConstruction:
    def test_safe_trace_has_steps(self, safe_trace: Trace):
        assert len(safe_trace.steps) == 3

    def test_violating_trace_has_steps(self, violating_trace: Trace):
        assert len(violating_trace.steps) == 4

    def test_metadata_matches(self, safe_trace: Trace):
        assert safe_trace.metadata.num_steps == 3
        assert safe_trace.metadata.reached_violation is False

    def test_violating_metadata(self, violating_trace: Trace):
        assert violating_trace.metadata.reached_violation is True


class TestConvenienceProperties:
    def test_safety_trajectory_length(self, safe_trace: Trace):
        assert len(safe_trace.safety_trajectory) == 3

    def test_risk_trajectory(self, safe_trace: Trace):
        risks = safe_trace.risk_trajectory
        assert all(r == RiskLevel.SAFE for r in risks)

    def test_violating_risk_trajectory_ends_violated(self, violating_trace: Trace):
        risks = violating_trace.risk_trajectory
        assert risks[-1] == RiskLevel.VIOLATED

    def test_state_index_trajectory_length(self, safe_trace: Trace):
        assert len(safe_trace.state_index_trajectory) == 3

    def test_state_index_trajectory_values_are_ints(self, safe_trace: Trace):
        for idx in safe_trace.state_index_trajectory:
            assert isinstance(idx, int)
            assert 0 <= idx < 60

    def test_violation_step_index_none_for_safe(self, safe_trace: Trace):
        assert safe_trace.violation_step_index is None

    def test_violation_step_index_for_violating(self, violating_trace: Trace):
        assert violating_trace.violation_step_index == 3


class TestSerialization:
    def test_json_round_trip(self, safe_trace: Trace):
        j = safe_trace.model_dump_json()
        t2 = Trace.model_validate_json(j)
        assert len(t2.steps) == len(safe_trace.steps)
        assert t2.metadata.scenario_id == safe_trace.metadata.scenario_id

    def test_dict_round_trip(self, violating_trace: Trace):
        d = violating_trace.model_dump()
        t2 = Trace.model_validate(d)
        assert t2.metadata.reached_violation is True
        assert len(t2.steps) == 4

    def test_step_safety_states_survive_round_trip(self, violating_trace: Trace):
        j = violating_trace.model_dump_json()
        t2 = Trace.model_validate_json(j)
        assert t2.steps[-1].safety_state_after.risk_level == RiskLevel.VIOLATED
        assert t2.violation_step_index == 3
