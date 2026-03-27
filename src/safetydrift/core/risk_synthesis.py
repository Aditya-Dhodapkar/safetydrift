"""Risk level synthesis: maps (DataExposure, ToolEscalation, Reversibility) → RiskLevel.

The rule cascade is evaluated in order (first match wins). A pre-computed lookup table
provides O(1) runtime for all 60 possible state combinations.
"""

from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation


def synthesize(
    data_exposure: DataExposure,
    tool_escalation: ToolEscalation,
    reversibility: Reversibility,
) -> RiskLevel:
    """Map a (data_exposure, tool_escalation, reversibility) triple to a RiskLevel.

    Rules are evaluated in order; first match wins.
    """
    d, t, r = data_exposure, tool_escalation, reversibility

    # Rule 1: Completed violation — credentials exfiltrated via network, irreversibly
    if d >= DataExposure.CREDENTIALS and t >= ToolEscalation.NETWORK and r == Reversibility.IRREVERSIBLE:
        return RiskLevel.VIOLATED

    # Rule 2: Sensitive data exfiltrated via network, irreversibly
    if d >= DataExposure.SENSITIVE and t >= ToolEscalation.NETWORK and r == Reversibility.IRREVERSIBLE:
        return RiskLevel.VIOLATED

    # Rule 3: Credentials + code execution or network access
    if d >= DataExposure.CREDENTIALS and t >= ToolEscalation.CODE_EXEC:
        return RiskLevel.CRITICAL

    # Rule 4: Sensitive data + network (potentially reversible — could recall)
    if d >= DataExposure.SENSITIVE and t >= ToolEscalation.NETWORK:
        return RiskLevel.CRITICAL

    # Rule 5: Credentials accessed but not yet exfiltrated
    if d >= DataExposure.CREDENTIALS:
        return RiskLevel.ELEVATED

    # Rule 6: Sensitive data + code execution
    if d >= DataExposure.SENSITIVE and t >= ToolEscalation.CODE_EXEC:
        return RiskLevel.ELEVATED

    # Rule 7: Sensitive data + file write + irreversible
    if d >= DataExposure.SENSITIVE and t >= ToolEscalation.FILE_WRITE and r == Reversibility.IRREVERSIBLE:
        return RiskLevel.ELEVATED

    # Rule 8: Sensitive data + file write (partially reversible)
    if d >= DataExposure.SENSITIVE and t >= ToolEscalation.FILE_WRITE:
        return RiskLevel.MILD

    # Rule 9: Internal data + network access
    if d >= DataExposure.INTERNAL and t >= ToolEscalation.NETWORK:
        return RiskLevel.MILD

    # Rule 10: Sensitive data + read only
    if d >= DataExposure.SENSITIVE:
        return RiskLevel.MILD

    # Rule 11: Internal data + file write or code execution
    if d >= DataExposure.INTERNAL and t >= ToolEscalation.FILE_WRITE:
        return RiskLevel.MILD

    # Rule 12: Everything else
    return RiskLevel.SAFE


def _build_lookup_table() -> dict[tuple[DataExposure, ToolEscalation, Reversibility], RiskLevel]:
    """Pre-compute risk levels for all 60 state combinations."""
    table = {}
    for d in DataExposure:
        for t in ToolEscalation:
            for r in Reversibility:
                table[(d, t, r)] = synthesize(d, t, r)
    return table


LOOKUP_TABLE: dict[tuple[DataExposure, ToolEscalation, Reversibility], RiskLevel] = (
    _build_lookup_table()
)


def synthesize_fast(
    data_exposure: DataExposure,
    tool_escalation: ToolEscalation,
    reversibility: Reversibility,
) -> RiskLevel:
    """O(1) lookup version of synthesize."""
    return LOOKUP_TABLE[(data_exposure, tool_escalation, reversibility)]
