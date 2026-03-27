"""Tests for risk level synthesis rules and lookup table."""

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.risk_synthesis import LOOKUP_TABLE, synthesize, synthesize_fast


class TestSynthesizeRules:
    """Test specific rule matches."""

    def test_rule1_credentials_network_irreversible_is_violated(self):
        assert synthesize(
            DataExposure.CREDENTIALS, ToolEscalation.NETWORK, Reversibility.IRREVERSIBLE
        ) == RiskLevel.VIOLATED

    def test_rule2_sensitive_network_irreversible_is_violated(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.NETWORK, Reversibility.IRREVERSIBLE
        ) == RiskLevel.VIOLATED

    def test_rule3_credentials_code_exec_is_critical(self):
        assert synthesize(
            DataExposure.CREDENTIALS, ToolEscalation.CODE_EXEC, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.CRITICAL

    def test_rule4_sensitive_network_reversible_is_critical(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.NETWORK, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.CRITICAL

    def test_rule5_credentials_readonly_is_elevated(self):
        assert synthesize(
            DataExposure.CREDENTIALS, ToolEscalation.READ_ONLY, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.ELEVATED

    def test_rule6_sensitive_code_exec_is_elevated(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.CODE_EXEC, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.ELEVATED

    def test_rule7_sensitive_write_irreversible_is_elevated(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.FILE_WRITE, Reversibility.IRREVERSIBLE
        ) == RiskLevel.ELEVATED

    def test_rule8_sensitive_write_partial_is_mild(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.FILE_WRITE, Reversibility.PARTIALLY
        ) == RiskLevel.MILD

    def test_rule9_internal_network_is_mild(self):
        assert synthesize(
            DataExposure.INTERNAL, ToolEscalation.NETWORK, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.MILD

    def test_rule10_sensitive_readonly_is_mild(self):
        assert synthesize(
            DataExposure.SENSITIVE, ToolEscalation.READ_ONLY, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.MILD

    def test_rule11_internal_write_is_mild(self):
        assert synthesize(
            DataExposure.INTERNAL, ToolEscalation.FILE_WRITE, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.MILD

    def test_rule12_none_readonly_reversible_is_safe(self):
        assert synthesize(
            DataExposure.NONE, ToolEscalation.READ_ONLY, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.SAFE

    def test_public_readonly_is_safe(self):
        assert synthesize(
            DataExposure.PUBLIC, ToolEscalation.READ_ONLY, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.SAFE

    def test_public_write_is_safe(self):
        assert synthesize(
            DataExposure.PUBLIC, ToolEscalation.FILE_WRITE, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.SAFE


class TestLookupTable:
    def test_covers_all_60_combinations(self):
        assert len(LOOKUP_TABLE) == 60

    def test_all_keys_are_valid_tuples(self):
        for key in LOOKUP_TABLE:
            d, t, r = key
            assert isinstance(d, DataExposure)
            assert isinstance(t, ToolEscalation)
            assert isinstance(r, Reversibility)

    def test_all_values_are_risk_levels(self):
        for value in LOOKUP_TABLE.values():
            assert isinstance(value, RiskLevel)

    def test_lookup_matches_synthesize(self):
        """The lookup table must match the rule-based function for every combination."""
        for d in DataExposure:
            for t in ToolEscalation:
                for r in Reversibility:
                    assert LOOKUP_TABLE[(d, t, r)] == synthesize(d, t, r), (
                        f"Mismatch at ({d.name}, {t.name}, {r.name})"
                    )

    def test_synthesize_fast_matches_synthesize(self):
        for d in DataExposure:
            for t in ToolEscalation:
                for r in Reversibility:
                    assert synthesize_fast(d, t, r) == synthesize(d, t, r)


class TestRiskDistribution:
    """Verify the risk landscape makes sense overall."""

    def test_at_least_one_violated_state(self):
        violated = [v for v in LOOKUP_TABLE.values() if v == RiskLevel.VIOLATED]
        assert len(violated) >= 1

    def test_at_least_one_safe_state(self):
        safe = [v for v in LOOKUP_TABLE.values() if v == RiskLevel.SAFE]
        assert len(safe) >= 1

    def test_all_risk_levels_represented(self):
        levels = set(LOOKUP_TABLE.values())
        assert levels == set(RiskLevel)

    def test_initial_state_is_safe(self):
        assert synthesize(
            DataExposure.NONE, ToolEscalation.READ_ONLY, Reversibility.FULLY_REVERSIBLE
        ) == RiskLevel.SAFE
