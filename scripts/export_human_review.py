"""Export a sample of labeled steps for human annotation.

Usage:
    python scripts/export_human_review.py --traces-dir results/traces/labeled --sample-rate 0.12
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from safetydrift.classifier.human import export_for_review
from safetydrift.scenarios.loader import load_all_scenarios
from safetydrift.traces.io import load_trace

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export steps for human review")
    parser.add_argument("--traces-dir", default="results/traces/labeled", help="Dir with labeled traces")
    parser.add_argument("--scenarios-dir", default="data/scenarios", help="Dir with scenario YAMLs")
    parser.add_argument("--output-dir", default="results/human_review", help="Output directory")
    parser.add_argument("--sample-rate", type=float, default=0.12, help="Sampling rate (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load scenarios
    scenarios_list = load_all_scenarios(args.scenarios_dir)
    scenarios = {s.id: s for s in scenarios_list}

    # Load traces from all subdirs
    traces_dir = Path(args.traces_dir)
    traces = []
    for trace_file in sorted(traces_dir.rglob("*.json")):
        traces.append(load_trace(trace_file))

    logger.info(f"Loaded {len(traces)} traces with {sum(len(t.steps) for t in traces)} total steps")

    json_path, csv_path = export_for_review(
        traces=traces,
        scenarios=scenarios,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    logger.info(f"Exported to:\n  JSON: {json_path}\n  CSV: {csv_path}")


if __name__ == "__main__":
    main()
