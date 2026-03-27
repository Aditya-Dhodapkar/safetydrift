"""Collect batch 2 traces with resume support.

Run from project root:
    python scripts/collect_traces_batch2.py

Safe to kill and restart — it skips already-collected traces.
Progress is printed every 10 traces.
"""

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from safetydrift.harness.llm_backends import AnthropicBackend
from safetydrift.harness.extended_sandbox import ExtendedSandboxExecutor
from safetydrift.harness.sandbox import SimulatedSandboxExecutor
from safetydrift.harness.langgraph_runner import SimpleAgentRunner
from safetydrift.scenarios.loader import load_all_scenarios
from safetydrift.traces.io import save_trace

OUTPUT_DIR = Path("results/traces/raw_batch2")
NUM_RUNS = 5
RUN_OFFSET = 5  # batch 2 uses run indices 5-9 (batch 1 used 0-4)


async def main():
    scenarios = load_all_scenarios("data/scenarios")
    print(f"Loaded {len(scenarios)} scenarios")

    backend = AnthropicBackend(model="claude-sonnet-4-20250514", temperature=0.1)
    executor = ExtendedSandboxExecutor(SimulatedSandboxExecutor())
    runner = SimpleAgentRunner()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build full task list
    tasks = []
    for scenario in scenarios:
        for run_idx in range(NUM_RUNS):
            cat_dir = OUTPUT_DIR / scenario.category
            filename = f"{scenario.id}_run{run_idx + RUN_OFFSET}.json"
            path = cat_dir / filename
            tasks.append((scenario, run_idx, path))

    # Filter out already-completed traces (resume support)
    remaining = [(s, r, p) for s, r, p in tasks if not p.exists()]
    done = len(tasks) - len(remaining)
    total = len(tasks)

    if done > 0:
        print(f"Resuming: {done}/{total} already done, {len(remaining)} remaining")
    else:
        print(f"Starting fresh: {total} traces to collect")

    if not remaining:
        print("All traces already collected!")
        return

    failed = 0
    start = time.time()

    for i, (scenario, run_idx, path) in enumerate(remaining):
        path.parent.mkdir(parents=True, exist_ok=True)

        trace = None
        for attempt in range(3):
            try:
                trace = await runner.run(scenario, backend, executor, max_steps=15)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  Retry {attempt+1} for {scenario.id}: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"  FAILED: {scenario.id} run {run_idx+RUN_OFFSET}: {e}")
                    failed += 1

        if trace:
            save_trace(trace, path)

        # Progress every 10 traces
        completed = done + i + 1
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            v = "V" if (trace and trace.metadata.reached_violation) else "."
            print(
                f"[{completed}/{total}] {i+1}/{len(remaining)} this session | "
                f"{failed} failed | ~{eta/60:.0f}min remaining | last: {scenario.id} {v}"
            )

    print(f"\nBatch 2 complete: {len(remaining) - failed} new traces, {failed} failed")
    print(f"Total traces in {OUTPUT_DIR}: {sum(1 for _ in OUTPUT_DIR.rglob('*.json'))}")


if __name__ == "__main__":
    asyncio.run(main())
