"""Extended sandbox executor that handles non-standard scenario tools.

Wraps SimulatedSandboxExecutor (Phase 1, untouched) and adds fallback handling
for tools not in the base 7 (read_file, write_file, execute_code, http_request,
send_email, list_files, search_files). Uses ToolRiskProfile metadata from
scenario definitions to generate safety deltas for non-standard tools.
"""

from __future__ import annotations

import logging
from typing import Any

from safetydrift.core.enums import Reversibility, ToolEscalation
from safetydrift.harness.base import SafetyDelta, ToolExecutor, ToolResult
from safetydrift.harness.sandbox import SimulatedSandboxExecutor
from safetydrift.scenarios.schema import ScenarioConfig, ToolRiskProfile

logger = logging.getLogger(__name__)


class ExtendedSandboxExecutor(ToolExecutor):
    """Wraps SimulatedSandboxExecutor with fallback for non-standard tools.

    Standard tools (read_file, write_file, etc.) are handled by the base executor.
    Non-standard tools (run_command, manage_firewall, execute_sql, etc.) use the
    scenario's ToolRiskProfile to generate appropriate safety deltas.
    """

    def __init__(self, base: SimulatedSandboxExecutor | None = None):
        self._base = base or SimulatedSandboxExecutor()
        self._tool_risk_profiles: dict[str, ToolRiskProfile] = {}
        self._tool_descriptions: dict[str, str] = {}

    def setup(self, scenario: ScenarioConfig) -> None:
        """Set up the base sandbox and index tool risk profiles."""
        self._base.setup(scenario)
        self._tool_risk_profiles.clear()
        self._tool_descriptions.clear()
        for tool_def in scenario.tools:
            self._tool_risk_profiles[tool_def.name] = tool_def.risk_profile
            self._tool_descriptions[tool_def.name] = tool_def.description

    def teardown(self) -> None:
        """Clean up the base sandbox."""
        self._base.teardown()
        self._tool_risk_profiles.clear()
        self._tool_descriptions.clear()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call, falling back to risk-profile simulation for unknown tools."""
        result = self._base.execute(tool_name, arguments)

        # If base handled it successfully, return as-is
        if result.error is None:
            return result

        # Try risk-profile fallback for non-standard tools
        profile = self._tool_risk_profiles.get(tool_name)
        if profile is None:
            logger.warning(f"Unknown tool '{tool_name}' with no risk profile")
            return result  # truly unknown — return base's error

        return self._simulate_from_profile(tool_name, arguments, profile)

    def _simulate_from_profile(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        profile: ToolRiskProfile,
    ) -> ToolResult:
        """Generate a simulated tool result from a ToolRiskProfile."""
        # Build a plausible output
        desc = self._tool_descriptions.get(tool_name, tool_name)
        args_summary = ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:3])
        output = f"[Simulated {tool_name}({args_summary})]\n{desc}\nCompleted successfully."

        # Derive safety delta from risk profile
        delta = SafetyDelta(
            data_exposure=profile.data_exposure.value if profile.data_exposure else None,
            tool_escalation=profile.tool_escalation.value if profile.tool_escalation else None,
        )

        # Determine reversibility from profile
        if profile.can_be_irreversible:
            delta.reversibility = Reversibility.IRREVERSIBLE.value
        elif profile.tool_escalation >= ToolEscalation.FILE_WRITE:
            delta.reversibility = Reversibility.PARTIALLY.value

        return ToolResult(
            tool_call_id="",
            tool_name=tool_name,
            output=output,
            safety_delta=delta,
        )
