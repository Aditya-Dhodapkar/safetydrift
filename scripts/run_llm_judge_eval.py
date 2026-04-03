"""Evaluate LLM judge baselines (no-context and context-aware) on test traces.

Run from project root:
    python scripts/run_llm_judge_eval.py

Requires ANTHROPIC_API_KEY in .env. Uses Claude Haiku for cheap evaluation.
Safe to kill and restart — prints progress every 5 traces.
"""

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from safetydrift.evaluation.metrics import compute_metrics
from safetydrift.evaluation.replay import replay_trace_async
from safetydrift.harness.llm_backends import AnthropicBackend
from safetydrift.markov.estimation import train_test_split_traces
from safetydrift.monitor.baselines import ContextLLMJudgeMonitor, LLMJudgeMonitor
from safetydrift.traces.io import load_trace

TRACES_DIR = Path("results/traces/labeled_v2")
OUTPUT_PATH = Path("results/evaluation/llm_judge_comparison.json")


async def evaluate_monitor(monitor, test_traces, label):
    """Run a single monitor on all test traces with progress reporting."""
    results = []
    failed = 0
    start = time.time()

    for i, trace in enumerate(test_traces):
        try:
            result = await replay_trace_async(monitor, trace)
            results.append(result)
        except Exception as e:
            print(f"  FAILED trace {trace.metadata.scenario_id}: {e}")
            failed += 1

        if (i + 1) % 5 == 0 or (i + 1) == len(test_traces):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(test_traces) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{label}] {i+1}/{len(test_traces)} | "
                f"{failed} failed | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining"
            )

    return results


async def main():
    # 1. Load and split traces (same split as main evaluation)
    traces = [load_trace(f) for f in sorted(TRACES_DIR.rglob("*.json"))]
    print(f"Loaded {len(traces)} traces")

    _, test_traces = train_test_split_traces(traces, test_fraction=0.2, seed=42)
    print(f"Test set: {len(test_traces)} traces")

    violating = sum(1 for t in test_traces if t.metadata.reached_violation)
    safe = len(test_traces) - violating
    print(f"  {violating} violating, {safe} safe")

    # 2. Create backends and monitors
    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", temperature=0)

    monitors = [
        LLMJudgeMonitor(backend),
        ContextLLMJudgeMonitor(backend, context_window=3),
    ]

    # 3. Evaluate each monitor
    all_results = {}
    for monitor in monitors:
        print(f"\nEvaluating: {monitor.name}")
        results = await evaluate_monitor(monitor, test_traces, monitor.name)
        metrics = compute_metrics(results, monitor.name)

        ew = f"{metrics.mean_early_warning:.1f}" if metrics.mean_early_warning else "---"
        print(f"\n  Results for {monitor.name}:")
        print(f"    Detection: {100*metrics.detection_rate:.1f}% ({metrics.num_detected}/{metrics.num_violating_traces})")
        print(f"    FPR:       {100*metrics.false_positive_rate:.1f}% ({metrics.num_false_positives}/{metrics.num_safe_traces})")
        print(f"    Early warning: {ew} steps")
        print(f"    Overhead:  {metrics.mean_overhead_ms:.1f} ms/step")

        all_results[monitor.name] = {
            "detection_rate": metrics.detection_rate,
            "false_positive_rate": metrics.false_positive_rate,
            "mean_early_warning": metrics.mean_early_warning,
            "median_early_warning": metrics.median_early_warning,
            "mean_overhead_ms": metrics.mean_overhead_ms,
            "num_detected": metrics.num_detected,
            "num_violating": metrics.num_violating_traces,
            "num_false_positives": metrics.num_false_positives,
            "num_safe": metrics.num_safe_traces,
        }

    # 4. Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {OUTPUT_PATH}")

    # 5. Print comparison table
    print("\n" + "=" * 60)
    print("LLM JUDGE COMPARISON")
    print("=" * 60)
    print(f"{'Monitor':<25} {'Det%':>6} {'FPR%':>6} {'EW':>6} {'ms/step':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        ew = f"{r['mean_early_warning']:.1f}" if r['mean_early_warning'] else "---"
        print(f"{name:<25} {100*r['detection_rate']:>5.1f}% {100*r['false_positive_rate']:>5.1f}% {ew:>6} {r['mean_overhead_ms']:>7.1f}")


if __name__ == "__main__":
    asyncio.run(main())
