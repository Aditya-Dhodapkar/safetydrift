"""Import human annotations and apply to labeled traces.

Usage:
    python scripts/import_human_review.py --review-file results/human_review/review_data.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from safetydrift.classifier.human import apply_human_labels, import_human_labels
from safetydrift.traces.io import load_trace, save_trace

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import human annotations")
    parser.add_argument("--review-file", required=True, help="Path to completed review JSON")
    parser.add_argument("--traces-dir", default="results/traces/labeled", help="Dir with labeled traces")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Import human labels
    human_labels = import_human_labels(args.review_file)
    logger.info(f"Imported {len(human_labels)} human annotations")

    # Load traces
    traces_dir = Path(args.traces_dir)
    traces = []
    trace_paths = []
    for trace_file in sorted(traces_dir.rglob("*.json")):
        traces.append(load_trace(trace_file))
        trace_paths.append(trace_file)

    # Apply human labels
    apply_human_labels(traces, human_labels)

    # Save updated traces
    for trace, path in zip(traces, trace_paths):
        save_trace(trace, path)

    human_count = sum(1 for t in traces for s in t.steps if s.label_source == "human")
    logger.info(f"Applied human labels. {human_count} steps now have label_source='human'")


if __name__ == "__main__":
    main()
