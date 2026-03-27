"""Tests for safety state enumerations."""

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation


class TestDataExposure:
    def test_ordering(self):
        assert DataExposure.NONE < DataExposure.PUBLIC < DataExposure.INTERNAL
        assert DataExposure.INTERNAL < DataExposure.SENSITIVE < DataExposure.CREDENTIALS

    def test_count(self):
        assert len(DataExposure) == 5

    def test_values(self):
        assert DataExposure.NONE == 0
        assert DataExposure.CREDENTIALS == 4

    def test_max(self):
        assert max(DataExposure.NONE, DataExposure.SENSITIVE) == DataExposure.SENSITIVE


class TestToolEscalation:
    def test_ordering(self):
        assert ToolEscalation.READ_ONLY < ToolEscalation.FILE_WRITE
        assert ToolEscalation.FILE_WRITE < ToolEscalation.CODE_EXEC < ToolEscalation.NETWORK

    def test_count(self):
        assert len(ToolEscalation) == 4

    def test_values(self):
        assert ToolEscalation.READ_ONLY == 0
        assert ToolEscalation.NETWORK == 3


class TestReversibility:
    def test_ordering(self):
        assert Reversibility.FULLY_REVERSIBLE < Reversibility.PARTIALLY < Reversibility.IRREVERSIBLE

    def test_count(self):
        assert len(Reversibility) == 3


class TestRiskLevel:
    def test_ordering(self):
        assert RiskLevel.SAFE < RiskLevel.MILD < RiskLevel.ELEVATED
        assert RiskLevel.ELEVATED < RiskLevel.CRITICAL < RiskLevel.VIOLATED

    def test_count(self):
        assert len(RiskLevel) == 5

    def test_values(self):
        assert RiskLevel.SAFE == 0
        assert RiskLevel.VIOLATED == 4
