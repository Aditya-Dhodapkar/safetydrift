"""Tests for SafetyState: transitions, monotonicity, absorbing, indexing, serialization."""

import json

import pytest

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.safety_state import SafetyState


class TestInitial:
    def test_initial_is_safe(self):
        s = SafetyState.initial()
        assert s.risk_level == RiskLevel.SAFE
        assert s.data_exposure == DataExposure.NONE
        assert s.tool_escalation == ToolEscalation.READ_ONLY
        assert s.reversibility == Reversibility.FULLY_REVERSIBLE

    def test_initial_is_not_absorbing(self):
        assert not SafetyState.initial().is_absorbing


class TestRiskLevelAutoSync:
    def test_risk_level_auto_computed(self):
        s = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        assert s.risk_level == RiskLevel.VIOLATED

    def test_wrong_risk_level_corrected(self):
        s = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
            risk_level=RiskLevel.SAFE,  # intentionally wrong
        )
        assert s.risk_level == RiskLevel.VIOLATED


class TestTransition:
    def test_basic_transition(self):
        s = SafetyState.initial()
        s2 = s.transition(new_exposure=DataExposure.INTERNAL)
        assert s2.data_exposure == DataExposure.INTERNAL
        assert s2.tool_escalation == ToolEscalation.READ_ONLY  # unchanged

    def test_monotonicity_data_exposure(self):
        s = SafetyState.initial().transition(new_exposure=DataExposure.SENSITIVE)
        s2 = s.transition(new_exposure=DataExposure.PUBLIC)  # try to decrease
        assert s2.data_exposure == DataExposure.SENSITIVE  # stays at max

    def test_monotonicity_tool_escalation(self):
        s = SafetyState.initial().transition(new_escalation=ToolEscalation.CODE_EXEC)
        s2 = s.transition(new_escalation=ToolEscalation.READ_ONLY)  # try to decrease
        assert s2.tool_escalation == ToolEscalation.CODE_EXEC  # stays at max

    def test_reversibility_not_monotonic(self):
        s = SafetyState.initial().transition(new_reversibility=Reversibility.IRREVERSIBLE)
        s2 = s.transition(new_reversibility=Reversibility.FULLY_REVERSIBLE)
        assert s2.reversibility == Reversibility.FULLY_REVERSIBLE  # can go back

    def test_multiple_dimensions(self):
        s = SafetyState.initial().transition(
            new_exposure=DataExposure.SENSITIVE,
            new_escalation=ToolEscalation.NETWORK,
            new_reversibility=Reversibility.IRREVERSIBLE,
        )
        assert s.risk_level == RiskLevel.VIOLATED

    def test_none_args_leave_unchanged(self):
        s = SafetyState.initial()
        s2 = s.transition()  # all None
        assert s2 == s

    def test_transition_returns_new_instance(self):
        s = SafetyState.initial()
        s2 = s.transition(new_exposure=DataExposure.PUBLIC)
        assert s is not s2


class TestAbsorbing:
    def test_violated_is_absorbing(self):
        s = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        assert s.is_absorbing

    def test_absorbing_transition_returns_self(self):
        s = SafetyState(
            data_exposure=DataExposure.CREDENTIALS,
            tool_escalation=ToolEscalation.NETWORK,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        s2 = s.transition(new_exposure=DataExposure.NONE)
        assert s2 is s

    def test_non_violated_not_absorbing(self):
        s = SafetyState.initial()
        assert not s.is_absorbing


class TestFrozen:
    def test_cannot_mutate(self):
        s = SafetyState.initial()
        with pytest.raises(Exception):
            s.data_exposure = DataExposure.CREDENTIALS


class TestIndexing:
    def test_initial_index_is_zero(self):
        assert SafetyState.initial().to_index() == 0

    def test_round_trip_all_states(self):
        for i in range(SafetyState.state_space_size()):
            s = SafetyState.from_index(i)
            assert s.to_index() == i

    def test_enumerate_all_indices_unique(self):
        states = SafetyState.enumerate_all()
        indices = [s.to_index() for s in states]
        assert len(set(indices)) == len(indices)

    def test_enumerate_all_indices_are_dense(self):
        states = SafetyState.enumerate_all()
        indices = sorted(s.to_index() for s in states)
        assert indices == list(range(60))

    def test_from_index_invalid_raises(self):
        with pytest.raises((ValueError, IndexError)):
            SafetyState.from_index(60)

    def test_from_index_negative_raises(self):
        with pytest.raises((ValueError, IndexError)):
            SafetyState.from_index(-1)


class TestEnumerateAll:
    def test_count(self):
        assert len(SafetyState.enumerate_all()) == 60

    def test_state_space_size(self):
        assert SafetyState.state_space_size() == 60

    def test_no_duplicates(self):
        states = SafetyState.enumerate_all()
        assert len(set(states)) == len(states)


class TestSerialization:
    def test_json_round_trip(self):
        s = SafetyState(
            data_exposure=DataExposure.SENSITIVE,
            tool_escalation=ToolEscalation.CODE_EXEC,
            reversibility=Reversibility.PARTIALLY,
        )
        j = s.model_dump_json()
        s2 = SafetyState.model_validate_json(j)
        assert s == s2

    def test_dict_round_trip(self):
        s = SafetyState.initial()
        d = s.model_dump()
        s2 = SafetyState.model_validate(d)
        assert s == s2

    def test_json_contains_expected_fields(self):
        s = SafetyState.initial()
        d = json.loads(s.model_dump_json())
        assert "data_exposure" in d
        assert "tool_escalation" in d
        assert "reversibility" in d
        assert "risk_level" in d
