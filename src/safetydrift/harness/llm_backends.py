"""LLM backend implementations.

AnthropicBackend for real API calls, MockBackend for testing.
"""

from __future__ import annotations

import os
from typing import Any

from safetydrift.harness.base import LLMBackend, LLMResponse, LLMToolCall


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)

        tool_calls = []
        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    LLMToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class MockBackend(LLMBackend):
    """Mock LLM backend that returns canned responses for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses) if responses else []
        self._call_count = 0

    def add_response(self, response: LLMResponse) -> None:
        """Add a response to the queue."""
        self._responses.append(response)

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> None:
        """Convenience: add a response that makes a single tool call."""
        self._responses.append(
            LLMResponse(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id=f"mock_call_{self._call_count + len(self._responses)}",
                        name=tool_name,
                        arguments=arguments or {},
                    )
                ],
                stop_reason="tool_use",
                model="mock",
            )
        )

    def add_final_response(self, content: str = "Task complete.") -> None:
        """Convenience: add a final text-only response (no tool calls)."""
        self._responses.append(
            LLMResponse(
                content=content,
                tool_calls=[],
                stop_reason="end_turn",
                model="mock",
            )
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(
                content="No more mock responses available.",
                stop_reason="end_turn",
                model="mock",
            )
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def create_backend(provider: str = "anthropic", **kwargs: Any) -> LLMBackend:
    """Factory function to create an LLM backend from config."""
    if provider == "anthropic":
        return AnthropicBackend(**kwargs)
    elif provider == "mock":
        return MockBackend()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
