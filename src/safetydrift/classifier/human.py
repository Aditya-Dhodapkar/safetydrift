"""Layer 3: Human validation export/import.

Exports a stratified random sample of classified steps for human annotation,
and imports human labels back into traces.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from safetydrift.classifier.models import StepClassification
from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.scenarios.schema import ScenarioConfig
from safetydrift.traces.models import Trace


def export_for_review(
    traces: list[Trace],
    scenarios: dict[str, ScenarioConfig],
    sample_rate: float = 0.12,
    output_dir: str | Path = "results/human_review",
    seed: int = 42,
    oversample_discrepancies: bool = True,
    oversample_transitions: bool = True,
) -> tuple[Path, Path]:
    """Export a stratified random sample of steps for human annotation.

    Args:
        traces: Labeled traces (after pipeline classification).
        scenarios: Scenario configs keyed by ID.
        sample_rate: Base sampling rate (0.12 = 12%).
        output_dir: Where to save export files.
        seed: Random seed for reproducibility.
        oversample_discrepancies: 2x sampling for steps with discrepancies.
        oversample_transitions: 2x sampling for steps where risk_level changed.

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Collect all candidate steps with weights
    candidates: list[dict[str, Any]] = []
    for trace in traces:
        scenario = scenarios.get(trace.metadata.scenario_id)
        for step in trace.steps:
            weight = 1.0

            # Oversample discrepancies (steps where label and sandbox may differ)
            if oversample_discrepancies and step.label_source == "llm_judge":
                weight *= 2.0

            # Oversample risk transitions
            if oversample_transitions:
                if step.safety_state_before.risk_level != step.safety_state_after.risk_level:
                    weight *= 2.0

            # Build sample entry
            tool_name = step.action.tool_call.tool_name if step.action.tool_call else None
            tool_args = step.action.tool_call.arguments if step.action.tool_call else {}

            candidates.append({
                "scenario_id": trace.metadata.scenario_id,
                "run_id": trace.metadata.run_id,
                "step_index": step.step_index,
                "scenario_description": scenario.description if scenario else "",
                "tool_name": tool_name,
                "tool_arguments": tool_args,
                "observation_truncated": step.observation.content[:500],
                "auto_data_exposure": step.safety_state_after.data_exposure.value,
                "auto_tool_escalation": step.safety_state_after.tool_escalation.value,
                "auto_reversibility": step.safety_state_after.reversibility.value,
                "auto_risk_level": step.safety_state_after.risk_level.value,
                "auto_label_source": step.label_source,
                "auto_confidence": step.label_confidence,
                "weight": weight,
                # Human fills these in:
                "human_data_exposure": None,
                "human_tool_escalation": None,
                "human_reversibility": None,
                "human_notes": None,
            })

    # Weighted sampling
    effective_rate = sample_rate
    total_weight = sum(c["weight"] for c in candidates)
    target_count = int(len(candidates) * effective_rate)

    # Use weighted random selection
    sampled = []
    for c in candidates:
        if rng.random() < c["weight"] * effective_rate:
            sampled.append(c)

    # Cap at reasonable size
    if len(sampled) > target_count * 2:
        sampled = rng.sample(sampled, target_count)

    # Save JSON
    json_data = {
        "export_metadata": {
            "sample_rate": sample_rate,
            "total_steps": len(candidates),
            "sampled_steps": len(sampled),
            "seed": seed,
        },
        "samples": sampled,
    }
    json_path = output_dir / "review_data.json"
    json_path.write_text(json.dumps(json_data, indent=2, default=str))

    # Save CSV (flattened)
    csv_path = output_dir / "review_data.csv"
    if sampled:
        fieldnames = [
            "scenario_id", "run_id", "step_index", "tool_name",
            "observation_truncated", "auto_data_exposure", "auto_tool_escalation",
            "auto_reversibility", "auto_risk_level", "auto_label_source",
            "auto_confidence", "human_data_exposure", "human_tool_escalation",
            "human_reversibility", "human_notes",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in sampled:
                writer.writerow(s)

    return json_path, csv_path


def import_human_labels(
    review_path: str | Path,
) -> dict[tuple[str, str, int], StepClassification]:
    """Import human annotations from a completed review file.

    Args:
        review_path: Path to the reviewed JSON file.

    Returns:
        Dict keyed by (scenario_id, run_id, step_index) → StepClassification.
    """
    path = Path(review_path)
    data = json.loads(path.read_text())

    labels: dict[tuple[str, str, int], StepClassification] = {}
    for sample in data.get("samples", []):
        # Skip if human didn't annotate
        if sample.get("human_data_exposure") is None:
            continue

        key = (sample["scenario_id"], sample["run_id"], sample["step_index"])
        labels[key] = StepClassification(
            data_exposure=DataExposure(sample["human_data_exposure"]),
            tool_escalation=ToolEscalation(sample["human_tool_escalation"]),
            reversibility=Reversibility(sample["human_reversibility"]),
            confidence=1.0,  # human labels are ground truth
            reasoning=sample.get("human_notes", "Human annotation"),
        )

    return labels


def apply_human_labels(
    traces: list[Trace],
    human_labels: dict[tuple[str, str, int], StepClassification],
) -> list[Trace]:
    """Apply human labels to traces, overriding automated classifications.

    Sets label_source='human' for steps with human annotations.
    Returns the same traces (mutated).
    """
    applied = 0
    for trace in traces:
        for step in trace.steps:
            key = (trace.metadata.scenario_id, trace.metadata.run_id, step.step_index)
            if key in human_labels:
                human = human_labels[key]
                step.label_source = "human"
                step.label_confidence = 1.0
                step.is_violation_step = human.risk_level == RiskLevel.VIOLATED
                applied += 1

    return traces
