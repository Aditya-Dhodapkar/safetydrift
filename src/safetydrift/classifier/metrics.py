"""Classifier accuracy metrics for the paper.

Computes per-dimension accuracy, confusion matrices, Cohen's kappa,
and error analysis comparing automated labels to human annotations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from safetydrift.classifier.models import StepClassification
from safetydrift.core.enums import DataExposure, Reversibility, RiskLevel, ToolEscalation
from safetydrift.core.risk_synthesis import synthesize_fast
from safetydrift.traces.models import Trace


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa inter-annotator agreement.

    Args:
        labels_a: First annotator's labels (integers).
        labels_b: Second annotator's labels (integers).

    Returns:
        Kappa score in [-1, 1]. 1 = perfect agreement, 0 = chance agreement.
    """
    assert len(labels_a) == len(labels_b), "Label lists must be same length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement by chance
    all_labels = set(labels_a) | set(labels_b)
    pe = 0.0
    for label in all_labels:
        count_a = sum(1 for x in labels_a if x == label)
        count_b = sum(1 for x in labels_b if x == label)
        pe += (count_a / n) * (count_b / n)

    if pe == 1.0:
        return 1.0  # avoid division by zero when perfect marginal agreement

    return (po - pe) / (1 - pe)


def confusion_matrix_for_dimension(
    automated: list[int],
    human: list[int],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix for a single dimension.

    Returns:
        np.ndarray of shape (num_classes, num_classes) where [i][j] = count of
        (automated=i, human=j).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for a, h in zip(automated, human):
        if 0 <= a < num_classes and 0 <= h < num_classes:
            cm[a][h] += 1
    return cm


def compute_accuracy_report(
    traces: list[Trace],
    human_labels: dict[tuple[str, str, int], StepClassification],
) -> dict[str, Any]:
    """Compute comprehensive accuracy metrics.

    Compares automated labels (from steps) to human labels (ground truth).
    Only includes steps that have both automated and human labels.

    Returns dict with:
        - per_dimension_accuracy: {dimension: float}
        - confusion_matrices: {dimension: list[list[int]]}
        - cohens_kappa: {dimension: float}
        - overall_accuracy: float
        - total_compared: int
        - error_analysis: list of error pattern dicts
    """
    # Collect paired labels
    auto_d, human_d = [], []
    auto_t, human_t = [], []
    auto_r, human_r = [], []
    auto_risk, human_risk = [], []

    for trace in traces:
        for step in trace.steps:
            key = (trace.metadata.scenario_id, trace.metadata.run_id, step.step_index)
            if key not in human_labels:
                continue

            human = human_labels[key]
            auto_state = step.safety_state_after

            auto_d.append(auto_state.data_exposure.value)
            human_d.append(human.data_exposure.value)

            auto_t.append(auto_state.tool_escalation.value)
            human_t.append(human.tool_escalation.value)

            auto_r.append(auto_state.reversibility.value)
            human_r.append(human.reversibility.value)

            auto_risk.append(auto_state.risk_level.value)
            human_risk_val = synthesize_fast(human.data_exposure, human.tool_escalation, human.reversibility)
            human_risk.append(human_risk_val.value)

    total = len(auto_d)
    if total == 0:
        return {
            "per_dimension_accuracy": {},
            "confusion_matrices": {},
            "cohens_kappa": {},
            "overall_accuracy": 0.0,
            "total_compared": 0,
            "error_analysis": [],
        }

    # Per-dimension accuracy
    accuracy_d = sum(a == h for a, h in zip(auto_d, human_d)) / total
    accuracy_t = sum(a == h for a, h in zip(auto_t, human_t)) / total
    accuracy_r = sum(a == h for a, h in zip(auto_r, human_r)) / total
    accuracy_risk = sum(a == h for a, h in zip(auto_risk, human_risk)) / total

    # Confusion matrices
    cm_d = confusion_matrix_for_dimension(auto_d, human_d, len(DataExposure))
    cm_t = confusion_matrix_for_dimension(auto_t, human_t, len(ToolEscalation))
    cm_r = confusion_matrix_for_dimension(auto_r, human_r, len(Reversibility))
    cm_risk = confusion_matrix_for_dimension(auto_risk, human_risk, len(RiskLevel))

    # Cohen's kappa
    kappa_d = cohens_kappa(auto_d, human_d)
    kappa_t = cohens_kappa(auto_t, human_t)
    kappa_r = cohens_kappa(auto_r, human_r)
    kappa_risk = cohens_kappa(auto_risk, human_risk)

    # Overall: all dimensions match
    overall = sum(
        ad == hd and at == ht and ar == hr
        for ad, hd, at, ht, ar, hr in zip(auto_d, human_d, auto_t, human_t, auto_r, human_r)
    ) / total

    return {
        "per_dimension_accuracy": {
            "data_exposure": accuracy_d,
            "tool_escalation": accuracy_t,
            "reversibility": accuracy_r,
            "risk_level": accuracy_risk,
        },
        "confusion_matrices": {
            "data_exposure": cm_d.tolist(),
            "tool_escalation": cm_t.tolist(),
            "reversibility": cm_r.tolist(),
            "risk_level": cm_risk.tolist(),
        },
        "cohens_kappa": {
            "data_exposure": kappa_d,
            "tool_escalation": kappa_t,
            "reversibility": kappa_r,
            "risk_level": kappa_risk,
        },
        "overall_accuracy": overall,
        "total_compared": total,
        "error_analysis": error_analysis(traces, human_labels),
    }


def error_analysis(
    traces: list[Trace],
    human_labels: dict[tuple[str, str, int], StepClassification],
) -> list[dict[str, Any]]:
    """Identify patterns in misclassification.

    Groups errors by tool_name, scenario_category, dimension, and direction.
    """
    errors: list[dict[str, Any]] = []

    for trace in traces:
        for step in trace.steps:
            key = (trace.metadata.scenario_id, trace.metadata.run_id, step.step_index)
            if key not in human_labels:
                continue

            human = human_labels[key]
            auto = step.safety_state_after

            # Check each dimension
            mismatches = []
            if auto.data_exposure != human.data_exposure:
                mismatches.append({
                    "dimension": "data_exposure",
                    "auto": auto.data_exposure.name,
                    "human": human.data_exposure.name,
                    "direction": "over" if auto.data_exposure > human.data_exposure else "under",
                })
            if auto.tool_escalation != human.tool_escalation:
                mismatches.append({
                    "dimension": "tool_escalation",
                    "auto": auto.tool_escalation.name,
                    "human": human.tool_escalation.name,
                    "direction": "over" if auto.tool_escalation > human.tool_escalation else "under",
                })
            if auto.reversibility != human.reversibility:
                mismatches.append({
                    "dimension": "reversibility",
                    "auto": auto.reversibility.name,
                    "human": human.reversibility.name,
                    "direction": "over" if auto.reversibility > human.reversibility else "under",
                })

            if mismatches:
                tool_name = step.action.tool_call.tool_name if step.action.tool_call else "message"
                errors.append({
                    "scenario_id": trace.metadata.scenario_id,
                    "category": trace.metadata.scenario_category,
                    "step_index": step.step_index,
                    "tool_name": tool_name,
                    "label_source": step.label_source,
                    "mismatches": mismatches,
                })

    return errors
