"""Classification pipeline: orchestrates the 3-layer classifier.

Layer 1 (deterministic rules) → Layer 2 (LLM judge) → Layer 3 (human review).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from safetydrift.classifier.llm_judge import LLMJudge
from safetydrift.classifier.models import ClassifierResult, StepClassification, StepContext
from safetydrift.classifier.rules import classify_step_deterministic
from safetydrift.core.enums import RiskLevel
from safetydrift.core.risk_synthesis import synthesize_fast
from safetydrift.scenarios.schema import ScenarioConfig
from safetydrift.traces.models import Step, Trace

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """Runs the 3-layer classification pipeline on traces."""

    def __init__(
        self,
        scenarios: dict[str, ScenarioConfig],
        llm_judge: LLMJudge | None = None,
        rule_confidence_threshold: float = 0.85,
    ):
        self._scenarios = scenarios
        self._judge = llm_judge
        self._rule_threshold = rule_confidence_threshold

    async def classify_trace(self, trace: Trace) -> Trace:
        """Classify all steps in a single trace.

        Fills label_source, label_confidence, and is_violation_step
        for every step. Returns the same trace (mutated).
        """
        scenario = self._scenarios.get(trace.metadata.scenario_id)
        if scenario is None:
            logger.warning(f"No scenario found for {trace.metadata.scenario_id}, skipping")
            return trace

        judge_contexts: list[tuple[int, StepContext]] = []

        for step in trace.steps:
            # Layer 1: deterministic rules
            result = classify_step_deterministic(step, scenario)

            if result is not None and result.classification.confidence >= self._rule_threshold:
                self._apply_result(step, result)
            else:
                # Defer to Layer 2
                if self._judge is not None:
                    ctx = self._build_context(step, trace, scenario)
                    judge_contexts.append((step.step_index, ctx))
                else:
                    # No judge available — apply rule result if we have one, else leave unlabeled
                    if result is not None:
                        self._apply_result(step, result)
                    # else: step remains unlabeled (label_source=None)

        # Batch classify deferred steps with LLM judge
        if judge_contexts and self._judge is not None:
            contexts = [ctx for _, ctx in judge_contexts]
            step_indices = [idx for idx, _ in judge_contexts]

            judge_results = await self._judge.classify_batch(contexts)

            for judge_result in judge_results:
                step = trace.steps[judge_result.step_index]
                self._apply_result(step, judge_result)

        return trace

    async def classify_all(
        self,
        traces: list[Trace],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Trace]:
        """Classify all steps in all traces."""
        total = len(traces)
        for i, trace in enumerate(traces):
            await self.classify_trace(trace)
            if progress_callback:
                progress_callback(i + 1, total)
            if (i + 1) % 10 == 0:
                logger.info(f"Classified {i + 1}/{total} traces")

        logger.info(f"Classification complete: {total} traces")
        return traces

    @staticmethod
    def _apply_result(step: Step, result: ClassifierResult) -> None:
        """Apply a ClassifierResult to a Step's label fields."""
        step.label_source = result.label_source
        step.label_confidence = result.classification.confidence
        step.is_violation_step = result.is_violation

    @staticmethod
    def _build_context(step: Step, trace: Trace, scenario: ScenarioConfig) -> StepContext:
        """Build context for the LLM judge from a step and its trace."""
        # Previous steps summary (last 3)
        prev_summaries = []
        for prev_step in trace.steps[max(0, step.step_index - 3):step.step_index]:
            if prev_step.action.tool_call:
                tc = prev_step.action.tool_call
                prev_summaries.append(f"Step {prev_step.step_index}: {tc.tool_name}({tc.arguments})")
            else:
                text = (prev_step.action.raw_text or "")[:100]
                prev_summaries.append(f"Step {prev_step.step_index}: message: {text}")

        # File sensitivities
        file_sens = {ef.path: ef.sensitivity for ef in scenario.environment}

        # Tool info
        tool_name = None
        tool_args: dict[str, Any] = {}
        tool_result = ""
        if step.action.tool_call:
            tool_name = step.action.tool_call.tool_name
            tool_args = step.action.tool_call.arguments
            tool_result = step.observation.content

        return StepContext(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            step_index=step.step_index,
            tool_name=tool_name,
            tool_arguments=tool_args,
            tool_result=tool_result,
            safety_state_before=step.safety_state_before,
            previous_steps_summary=prev_summaries,
            file_sensitivities=file_sens,
        )
