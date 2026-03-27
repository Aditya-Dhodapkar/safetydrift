"""Compute classifier accuracy metrics comparing automated vs. human labels.

Usage:
    python scripts/classifier_accuracy.py --traces-dir results/traces/labeled --review-file results/human_review/review_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from safetydrift.classifier.human import import_human_labels
from safetydrift.classifier.metrics import compute_accuracy_report
from safetydrift.traces.io import load_trace

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute classifier accuracy")
    parser.add_argument("--traces-dir", default="results/traces/labeled")
    parser.add_argument("--review-file", default="results/human_review/review_data.json")
    parser.add_argument("--output", default="results/classifier_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load traces
    traces_dir = Path(args.traces_dir)
    traces = []
    for f in sorted(traces_dir.rglob("*.json")):
        traces.append(load_trace(f))
    logger.info(f"Loaded {len(traces)} traces")

    # Load human labels
    human_labels = import_human_labels(args.review_file)
    logger.info(f"Loaded {len(human_labels)} human annotations")

    # Compute metrics
    report = compute_accuracy_report(traces, human_labels)

    # Print summary
    print("\n=== Classifier Accuracy Report ===\n")
    print(f"Total steps compared: {report['total_compared']}")
    print(f"Overall accuracy (all dimensions): {report['overall_accuracy']:.3f}\n")

    print("Per-dimension accuracy:")
    for dim, acc in report["per_dimension_accuracy"].items():
        kappa = report["cohens_kappa"][dim]
        print(f"  {dim}: accuracy={acc:.3f}, kappa={kappa:.3f}")

    print(f"\nError analysis: {len(report['error_analysis'])} misclassified steps")

    # Save full report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
