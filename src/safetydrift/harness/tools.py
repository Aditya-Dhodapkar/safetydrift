"""Tool schema conversion for Anthropic API format."""

from __future__ import annotations

from safetydrift.scenarios.schema import ScenarioConfig, ToolDefinition


def to_anthropic_tools(scenario: ScenarioConfig) -> list[dict]:
    """Convert scenario tool definitions to Anthropic tool format."""
    return [_tool_to_anthropic(t) for t in scenario.tools]


def _tool_to_anthropic(tool: ToolDefinition) -> dict:
    """Convert a single ToolDefinition to Anthropic format."""
    properties = {}
    required = []

    for param_name, param_spec in tool.parameters.items():
        properties[param_name] = {
            "type": param_spec.get("type", "string"),
            "description": param_spec.get("description", ""),
        }
        if param_spec.get("required", False):
            required.append(param_name)

    schema: dict = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }
