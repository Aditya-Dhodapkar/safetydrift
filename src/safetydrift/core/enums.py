"""Safety state dimension enumerations.

Four IntEnum classes representing the dimensions of an agent's safety state.
IntEnum enables ordering via <, >, max() which is essential for monotonicity enforcement.
"""

from enum import IntEnum


class DataExposure(IntEnum):
    """Sensitivity level of data the agent has accessed. Monotonically non-decreasing."""

    NONE = 0
    PUBLIC = 1
    INTERNAL = 2
    SENSITIVE = 3
    CREDENTIALS = 4


class ToolEscalation(IntEnum):
    """Most powerful capability the agent has used. Monotonically non-decreasing."""

    READ_ONLY = 0
    FILE_WRITE = 1
    CODE_EXEC = 2
    NETWORK = 3


class Reversibility(IntEnum):
    """Whether the agent's cumulative actions can be undone. NOT monotonic."""

    FULLY_REVERSIBLE = 0
    PARTIALLY = 1
    IRREVERSIBLE = 2


class RiskLevel(IntEnum):
    """Synthesized discrete risk tier. VIOLATED is an absorbing state."""

    SAFE = 0
    MILD = 1
    ELEVATED = 2
    CRITICAL = 3
    VIOLATED = 4
