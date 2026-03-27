"""Tests for Layer 2 LLM judge classifier (mocked — no real API calls)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from safetydrift.classifier.llm_judge import LLMJudge, MAX_OBSERVATION_LENGTH
from safetydrift.classifier.models import ClassifierResult, StepClassification, StepContext
from safetydrift.core.enums import DataExposure, Reversibility, ToolEscalation
from safetydrift.core.safety_state import SafetyState


@pytest.fixture
def sample_context() -> StepContext:
    return StepContext(
        scenario_id="test_scenario",
        scenario_description="Test scenario for judge testing",
        step_index=2,
        tool_name="read_file",
        tool_arguments={"path": "/etc/shadow"},
        tool_result="root:$6$hash:18000:0:99999:7:::",
        safety_state_before=SafetyState.initial(),
        previous_steps_summary=["Step 0: read_file(/etc/passwd)", "Step 1: write_file(output.txt)"],
        file_sensitivities={"data/public.txt": 1},
    )


@pytest.fixture
def mock_classification() -> StepClassification:
    return StepClassification(
        data_exposure=DataExposure.CREDENTIALS,
        tool_escalation=ToolEscalation.READ_ONLY,
        reversibility=Reversibility.FULLY_REVERSIBLE,
        confidence=0.92,
        reasoning="Reading /etc/shadow exposes credential hashes",
    )


def _make_mock_client(return_value=None, side_effect=None):
    """Create a mock instructor client with messages.create mocked."""
    client = MagicMock()
    messages = MagicMock()
    messages.create = AsyncMock(return_value=return_value, side_effect=side_effect)
    client.messages = messages
    return client


class TestLLMJudgeClassifyStep:
    @pytest.mark.asyncio
    async def test_returns_valid_result(self, sample_context, mock_classification):
        mock_client = _make_mock_client(return_value=mock_classification)
        judge = LLMJudge(client=mock_client)

        result = await judge.classify_step(sample_context)

        assert isinstance(result, ClassifierResult)
        assert result.label_source == "llm_judge"
        assert result.classification.data_exposure == DataExposure.CREDENTIALS
        assert result.classification.confidence == 0.92
        assert result.step_index == 2

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self, sample_context):
        mock_client = _make_mock_client(side_effect=Exception("API error"))
        judge = LLMJudge(max_retries=0, client=mock_client)

        result = await judge.classify_step(sample_context)

        assert isinstance(result, ClassifierResult)
        assert result.classification.confidence == 0.3  # fallback confidence
        assert result.discrepancy is True


class TestPromptBuilding:
    def test_prompt_contains_context(self, sample_context):
        judge = LLMJudge()
        prompt = judge._build_prompt(sample_context)
        assert "Test scenario" in prompt
        assert "read_file" in prompt
        assert "/etc/shadow" in prompt
        assert "Step 0" in prompt

    def test_prompt_truncates_long_observation(self):
        ctx = StepContext(
            scenario_id="test",
            scenario_description="Test",
            step_index=0,
            tool_name="read_file",
            tool_arguments={"path": "big.txt"},
            tool_result="x" * 5000,
            safety_state_before=SafetyState.initial(),
        )
        judge = LLMJudge()
        prompt = judge._build_prompt(ctx)
        assert "[truncated]" in prompt
        assert len(ctx.tool_result) > MAX_OBSERVATION_LENGTH


class TestBatchClassification:
    @pytest.mark.asyncio
    async def test_batch_classifies_all(self, sample_context, mock_classification):
        mock_client = _make_mock_client(return_value=mock_classification)
        judge = LLMJudge(client=mock_client)

        contexts = [sample_context] * 5
        results = await judge.classify_batch(contexts, concurrency=2)

        assert len(results) == 5
        assert all(r.label_source == "llm_judge" for r in results)
