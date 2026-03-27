"""Run the 3-layer classifier pipeline on collected traces.

Usage:
    python scripts/label_traces.py
    python scripts/label_traces.py classifier.use_llm_judge=false  # rules only
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig

from safetydrift.classifier.llm_judge import LLMJudge
from safetydrift.classifier.pipeline import ClassifierPipeline
from safetydrift.scenarios.loader import load_all_scenarios
from safetydrift.traces.io import load_traces, save_trace

logger = logging.getLogger(__name__)


async def label_all(cfg: DictConfig) -> None:
    """Load raw traces, classify them, save labeled traces."""
    input_dir = Path(cfg.classifier.input_dir)
    output_dir = Path(cfg.classifier.output_dir)

    # Load scenarios
    scenarios_list = load_all_scenarios(cfg.experiment.scenarios_dir)
    scenarios = {s.id: s for s in scenarios_list}
    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Load raw traces
    traces = load_traces(input_dir)
    # Also load from category subdirs
    for cat_dir in input_dir.iterdir():
        if cat_dir.is_dir() and cat_dir.name != ".":
            for trace_file in sorted(cat_dir.glob("*.json")):
                from safetydrift.traces.io import load_trace
                traces.append(load_trace(trace_file))

    logger.info(f"Loaded {len(traces)} raw traces")

    # Build judge if enabled
    judge = None
    if cfg.classifier.use_llm_judge:
        judge = LLMJudge(
            model=cfg.classifier.llm_judge.model,
            temperature=cfg.classifier.llm_judge.temperature,
        )

    # Build and run pipeline
    pipeline = ClassifierPipeline(
        scenarios=scenarios,
        llm_judge=judge,
        rule_confidence_threshold=cfg.classifier.rule_confidence_threshold,
    )

    def progress(done: int, total: int) -> None:
        logger.info(f"Progress: {done}/{total} traces classified")

    await pipeline.classify_all(traces, progress_callback=progress)

    # Save labeled traces
    output_dir.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        cat_dir = output_dir / trace.metadata.scenario_category
        cat_dir.mkdir(parents=True, exist_ok=True)
        path = cat_dir / f"{trace.metadata.scenario_id}_{trace.metadata.run_id}_labeled.json"
        save_trace(trace, path)

    # Summary
    total_steps = sum(len(t.steps) for t in traces)
    rule_steps = sum(1 for t in traces for s in t.steps if s.label_source == "rule")
    judge_steps = sum(1 for t in traces for s in t.steps if s.label_source == "llm_judge")
    unlabeled = sum(1 for t in traces for s in t.steps if s.label_source is None)
    violations = sum(1 for t in traces for s in t.steps if s.is_violation_step)

    logger.info(f"\n=== Classification Summary ===")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"  Rule-classified: {rule_steps} ({100*rule_steps/total_steps:.1f}%)")
    logger.info(f"  LLM-classified: {judge_steps} ({100*judge_steps/total_steps:.1f}%)")
    logger.info(f"  Unlabeled: {unlabeled}")
    logger.info(f"  Violation steps: {violations}")
    logger.info(f"Saved to {output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(label_all(cfg))


if __name__ == "__main__":
    main()
