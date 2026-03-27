"""Abstract base classes for the agent execution harness.

These interfaces are locked contracts — Phases 2-5 depend on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from safetydrift.scenarios.schema import ScenarioConfig
from safetydrift.traces.models import Trace


@dataclass
class LLMResponse:
    """Response from an LLM backend."""

    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    tool_name: str
    output: str
    error: str | None = None
    safety_delta: SafetyDelta | None = None


@dataclass
class SafetyDelta:
    """Change to safety state caused by a tool execution."""

    data_exposure: int | None = None  # DataExposure value
    tool_escalation: int | None = None  # ToolEscalation value
    reversibility: int | None = None  # Reversibility value


class LLMBackend(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history in provider-specific format.
            tools: Tool definitions in provider-specific format.
            system: Optional system prompt.

        Returns:
            LLMResponse with content and optional tool calls.
        """
        ...


class ToolExecutor(ABC):
    """Abstract interface for executing tool calls in a sandboxed environment."""

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool call and return the result with safety delta."""
        ...

    @abstractmethod
    def setup(self, scenario: ScenarioConfig) -> None:
        """Prepare the execution environment for a scenario."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the execution environment."""
        ...


class AgentRunner(ABC):
    """Abstract interface for running an agent on a scenario."""

    @abstractmethod
    async def run(
        self,
        scenario: ScenarioConfig,
        llm_backend: LLMBackend,
        tool_executor: ToolExecutor,
        max_steps: int = 25,
    ) -> Trace:
        """Run an agent on a scenario and return the execution trace.

        Args:
            scenario: The scenario configuration to execute.
            llm_backend: The LLM provider to use.
            tool_executor: The sandbox for tool execution.
            max_steps: Maximum number of agent steps before forced stop.

        Returns:
            A complete execution Trace.
        """
        ...
