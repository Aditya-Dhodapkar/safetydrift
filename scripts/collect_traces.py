"""Collect execution traces by running agents on all scenarios.

Uses ExtendedSandboxExecutor for full tool coverage.
Saves raw traces to results/traces/raw/{category}/{scenario_id}_run{idx}.json.

Usage:
    python scripts/collect_traces.py llm=sonnet experiment.num_runs_per_scenario=5
    python scripts/collect_traces.py llm=sonnet experiment.num_runs_per_scenario=1  # quick test
"""

from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from safetydrift.harness.extended_sandbox import ExtendedSandboxExecutor
from safetydrift.harness.langgraph_runner import LangGraphRunner, SimpleAgentRunner
from safetydrift.harness.llm_backends import create_backend
from safetydrift.harness.sandbox import SimulatedSandboxExecutor
from safetydrift.scenarios.loader import load_all_scenarios
from safetydrift.traces.io import save_trace

logger = logging.getLogger(__name__)


async def collect_all(cfg: DictConfig) -> None:
    """Run all scenarios and collect traces."""
    scenarios = load_all_scenarios(cfg.experiment.scenarios_dir)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    backend = create_backend(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )

    executor = ExtendedSandboxExecutor(SimulatedSandboxExecutor())

    if cfg.agent.framework == "langgraph":
        runner = LangGraphRunner()
    else:
        runner = SimpleAgentRunner()

    output_dir = Path(cfg.experiment.trace_output_dir) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_runs = cfg.experiment.num_runs_per_scenario
    total = len(scenarios) * num_runs
    completed = 0
    failed = 0
    manifest_entries = []
    start_time = time.time()

    for scenario in scenarios:
        cat_dir = output_dir / scenario.category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(num_runs):
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total - completed) / rate if rate > 0 else 0

            logger.info(
                f"[{completed}/{total}] {scenario.id} run {run_idx + 1}/{num_runs} "
                f"(~{remaining:.0f}s remaining)"
            )

            trace = None
            for attempt in range(3):  # retry up to 3 times
                try:
                    trace = await runner.run(
                        scenario=scenario,
                        llm_backend=backend,
                        tool_executor=executor,
                        max_steps=cfg.agent.max_steps,
                    )
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"  Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)

            if trace is None:
                logger.error(f"  FAILED after 3 attempts: {scenario.id} run {run_idx}")
                failed += 1
                continue

            path = cat_dir / f"{scenario.id}_run{run_idx}.json"
            save_trace(trace, path)

            manifest_entries.append({
                "path": str(path),
                "scenario_id": scenario.id,
                "category": scenario.category,
                "run_index": run_idx,
                "num_steps": trace.metadata.num_steps,
                "reached_violation": trace.metadata.reached_violation,
            })

            logger.info(
                f"  → {trace.metadata.num_steps} steps, "
                f"violated={trace.metadata.reached_violation}"
            )

    # Save manifest
    manifest = {
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "llm_model": cfg.llm.model,
        "agent_framework": cfg.agent.framework,
        "num_scenarios": len(scenarios),
        "num_runs_per_scenario": num_runs,
        "total_traces": len(manifest_entries),
        "failed": failed,
        "traces": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        f"\nCollection complete: {len(manifest_entries)} traces saved, {failed} failed. "
        f"Manifest at {manifest_path}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(collect_all(cfg))


if __name__ == "__main__":
    main()
