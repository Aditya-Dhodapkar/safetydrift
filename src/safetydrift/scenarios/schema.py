"""Scenario configuration schema.

Each scenario defines a realistic multi-step task that an agent will execute,
along with metadata about tools, environment, expected safe paths, and drift opportunities.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from safetydrift.core.enums import DataExposure, RiskLevel, ToolEscalation


class ToolRiskProfile(BaseModel):
    """What safety state changes this tool can cause."""

    data_exposure: DataExposure = DataExposure.NONE
    tool_escalation: ToolEscalation = ToolEscalation.READ_ONLY
    can_be_irreversible: bool = False


class ToolDefinition(BaseModel):
    """A tool available to the agent in this scenario."""

    name: str
    description: str
    parameters: dict[str, dict] = Field(default_factory=dict)
    risk_profile: ToolRiskProfile = Field(default_factory=ToolRiskProfile)


class EnvironmentFile(BaseModel):
    """A file present in the scenario's simulated environment."""

    path: str
    content_file: str | None = None  # path relative to data/environments/
    inline_content: str | None = None
    sensitivity: DataExposure = DataExposure.NONE
    description: str = ""


class DriftOpportunity(BaseModel):
    """A point where the agent might drift toward unsafe behavior."""

    step_description: str
    risk_if_taken: RiskLevel
    data_exposure_if_taken: DataExposure = DataExposure.NONE
    tool_escalation_if_taken: ToolEscalation = ToolEscalation.READ_ONLY


class ExpectedPath(BaseModel):
    """The expected safe way to complete the task."""

    description: str
    steps: list[str]
    expected_max_risk: RiskLevel = RiskLevel.MILD


class MockResponse(BaseModel):
    """A canned response for simulated external services."""

    endpoint: str
    method: str = "GET"
    response: str = ""
    status_code: int = 200


class ScenarioConfig(BaseModel):
    """Complete configuration for a single evaluation scenario."""

    id: str
    category: str  # data_handling | sysadmin | research_comms | code_debugging
    name: str
    description: str
    task_prompt: str
    tools: list[ToolDefinition]
    environment: list[EnvironmentFile] = Field(default_factory=list)
    mock_responses: list[MockResponse] = Field(default_factory=list)
    expected_safe_path: ExpectedPath
    drift_opportunities: list[DriftOpportunity]
    ground_truth_can_violate: bool
    difficulty: str = "medium"  # easy | medium | hard
    tags: list[str] = Field(default_factory=list)
