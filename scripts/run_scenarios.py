"""CLI entry point for running scenarios and collecting traces.

Usage:
    python scripts/run_scenarios.py agent=langgraph llm=sonnet execution=simulated
    python scripts/run_scenarios.py experiment.num_runs=3
"""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from safetydrift.harness.langgraph_runner import LangGraphRunner, SimpleAgentRunner
from safetydrift.harness.llm_backends import create_backend
from safetydrift.harness.sandbox import create_executor
from safetydrift.scenarios.loader import load_all_scenarios
from safetydrift.traces.io import save_trace

logger = logging.getLogger(__name__)


async def run_all(cfg: DictConfig) -> None:
    """Run all scenarios with the configured agent/LLM/executor."""
    scenarios = load_all_scenarios(cfg.experiment.scenarios_dir)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    backend = create_backend(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )

    executor = create_executor(sandbox_type=cfg.execution.sandbox_type)

    if cfg.agent.framework == "langgraph":
        runner = LangGraphRunner()
    else:
        runner = SimpleAgentRunner()

    for scenario in scenarios:
        for run_idx in range(cfg.experiment.num_runs_per_scenario):
            logger.info(
                f"Running {scenario.id} (run {run_idx + 1}/{cfg.experiment.num_runs_per_scenario})"
            )
            try:
                trace = await runner.run(
                    scenario=scenario,
                    llm_backend=backend,
                    tool_executor=executor,
                    max_steps=cfg.agent.max_steps,
                )
                output_dir = cfg.experiment.trace_output_dir
                save_trace(
                    trace,
                    f"{output_dir}/{scenario.category}/{scenario.id}_run{run_idx}.json",
                )
                logger.info(
                    f"  → {trace.metadata.num_steps} steps, "
                    f"violated={trace.metadata.reached_violation}"
                )
            except Exception as e:
                logger.error(f"  → Failed: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_all(cfg))


if __name__ == "__main__":
    main()
