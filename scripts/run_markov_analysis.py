"""Run all Phase 3 Markov chain experiments.

Usage:
    python scripts/run_markov_analysis.py
    python scripts/run_markov_analysis.py markov.threshold=0.80
    python scripts/run_markov_analysis.py markov.granularity=fine markov.smoothing=1.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from safetydrift.markov.absorption import analyze_absorption, build_absorption_lookup
from safetydrift.markov.ablation import compute_learning_curve, run_dimension_ablation
from safetydrift.markov.analysis import analyze_per_category, compare_category_absorption, find_points_of_no_return
from safetydrift.markov.estimation import confidence_intervals, estimate_transition_matrix, train_test_split_traces
from safetydrift.markov.validation import validate_markov_property
from safetydrift.traces.io import load_trace

logger = logging.getLogger(__name__)


def _to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(data), indent=2, default=str))
    logger.info(f"  Saved: {path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    traces_dir = Path(cfg.markov.traces_dir)
    output_dir = Path(cfg.markov.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    granularity = cfg.markov.granularity
    smoothing = cfg.markov.smoothing
    horizons = list(cfg.markov.horizons)
    threshold = cfg.markov.threshold

    # 1. Load traces
    traces = []
    for f in sorted(traces_dir.rglob("*.json")):
        traces.append(load_trace(f))
    logger.info(f"Loaded {len(traces)} labeled traces")

    # 2. Train/test split
    train, test = train_test_split_traces(
        traces, test_fraction=cfg.markov.test_fraction, seed=cfg.markov.seed,
    )
    logger.info(f"Split: {len(train)} train, {len(test)} test")

    # 3. Estimate transition matrix
    logger.info(f"\n=== Transition Matrix ({granularity}) ===")
    tm = estimate_transition_matrix(train, granularity, smoothing)
    lower, upper = confidence_intervals(tm.counts)
    _save_json({
        "counts": tm.counts, "matrix": tm.matrix,
        "confidence_lower": lower, "confidence_upper": upper,
        "state_size": tm.state_size, "granularity": tm.granularity,
        "num_traces": tm.num_traces, "num_transitions": tm.num_transitions,
    }, output_dir / f"transition_matrix_{granularity}.json")

    # 4. Absorption analysis
    logger.info(f"\n=== Absorption Analysis ===")
    ar = analyze_absorption(tm, horizons=horizons)

    names = {0: "SAFE", 1: "MILD", 2: "ELEVATED", 3: "CRITICAL", 4: "VIOLATED"} if granularity == "coarse" else {}
    logger.info("Infinite-horizon absorption probabilities:")
    for idx, prob in sorted(ar.state_absorption_map.items()):
        name = names.get(idx, str(idx))
        logger.info(f"  {name}: P={prob:.4f}, E[steps]={ar.state_passage_map.get(idx, 0):.1f}")

    logger.info("\nFinite-horizon P(VIOLATED within N steps):")
    for h in horizons:
        probs = ar.finite_horizon_probs[h]
        logger.info(f"  Horizon {h}:")
        for i in range(min(5, len(probs))):
            name = names.get(i, str(i))
            logger.info(f"    {name}: {probs[i]:.3f}")

    _save_json({
        "absorption_probs": dict(ar.state_absorption_map),
        "mean_passage_times": dict(ar.state_passage_map),
        "finite_horizon": {str(h): ar.finite_horizon_probs[h] for h in horizons},
        "transient_indices": ar.transient_indices,
        "absorbing_indices": ar.absorbing_indices,
    }, output_dir / "absorption_results.json")

    # 5. Markov validation (Experiment 1)
    logger.info(f"\n=== Markov Property Validation (Experiment 1) ===")
    vr = validate_markov_property(train, test, granularity)
    logger.info(f"  Order 1: accuracy={vr.order_1_accuracy:.3f}, LL={vr.order_1_log_likelihood:.3f}")
    logger.info(f"  Order 2: accuracy={vr.order_2_accuracy:.3f}, LL={vr.order_2_log_likelihood:.3f}")
    logger.info(f"  Order 3: accuracy={vr.order_3_accuracy:.3f}, LL={vr.order_3_log_likelihood:.3f}")
    logger.info(f"  Chi-squared: stat={vr.chi_squared_statistic:.2f}, p={vr.chi_squared_p_value:.4f}, df={vr.degrees_of_freedom}")

    _save_json({
        "order_1": {"accuracy": vr.order_1_accuracy, "log_likelihood": vr.order_1_log_likelihood},
        "order_2": {"accuracy": vr.order_2_accuracy, "log_likelihood": vr.order_2_log_likelihood},
        "order_3": {"accuracy": vr.order_3_accuracy, "log_likelihood": vr.order_3_log_likelihood},
        "per_state_accuracy": vr.per_state_accuracy,
        "chi_squared": {"statistic": vr.chi_squared_statistic, "p_value": vr.chi_squared_p_value, "df": vr.degrees_of_freedom},
        "num_train": vr.num_train_transitions, "num_test": vr.num_test_transitions,
    }, output_dir / "validation_results.json")

    # 6. Points of no return (Experiment 2)
    logger.info(f"\n=== Points of No Return (Experiment 2) ===")
    ponr = find_points_of_no_return(ar, threshold=threshold, horizon=5)
    logger.info(f"  Threshold: {threshold}, Horizon: 5 steps")
    logger.info(f"  Cliff: {ponr.cliff_description}")
    for p in ponr.points_of_no_return:
        logger.info(f"  PONR: {p.risk_level_name} — P(VIOLATED in 5)={p.finite_horizon_prob:.3f}")

    _save_json({
        "threshold": ponr.threshold, "horizon": ponr.horizon,
        "points_of_no_return": [{"state": p.state_index, "name": p.risk_level_name,
                                  "prob": p.finite_horizon_prob, "mean_steps": p.mean_steps_to_violation}
                                 for p in ponr.points_of_no_return],
        "cliff_description": ponr.cliff_description,
        "zones": ponr.zones,
        "all_states_sorted": ponr.all_states_sorted,
    }, output_dir / "points_of_no_return.json")

    # 7. Per-category analysis
    logger.info(f"\n=== Per-Category Analysis ===")
    per_cat = analyze_per_category(traces, granularity, smoothing, threshold, horizon=5, horizons=horizons)
    comparison = compare_category_absorption(per_cat)

    for pcr in per_cat:
        logger.info(f"  {pcr.category}: {pcr.num_traces} traces, {pcr.num_violated} violations ({pcr.violation_rate:.0%})")
        for p in pcr.points_of_no_return.points_of_no_return:
            logger.info(f"    PONR: {p.risk_level_name} — P={p.finite_horizon_prob:.3f}")

    _save_json({
        "categories": {pcr.category: {
            "num_traces": pcr.num_traces, "num_violated": pcr.num_violated,
            "violation_rate": pcr.violation_rate,
            "transition_matrix": pcr.transition_matrix.matrix,
            "points_of_no_return": [{"name": p.risk_level_name, "prob": p.finite_horizon_prob}
                                     for p in pcr.points_of_no_return.points_of_no_return],
        } for pcr in per_cat},
        "comparison": comparison,
    }, output_dir / "per_category_analysis.json")

    # 8. Ablation (Experiment 4)
    logger.info(f"\n=== Dimension Ablation (Experiment 4) ===")
    ablation = run_dimension_ablation(train, test)
    for a in ablation:
        label = f"remove {a.removed_dimension}" if a.removed_dimension else "BASELINE (full)"
        logger.info(f"  {label}: accuracy={a.accuracy:.3f}, LL={a.log_likelihood:.3f}, states={a.state_size}")

    _save_json({
        "ablations": [{"removed": a.removed_dimension, "remaining": a.remaining_dimensions,
                        "accuracy": a.accuracy, "log_likelihood": a.log_likelihood,
                        "state_size": a.state_size} for a in ablation],
    }, output_dir / "ablation_results.json")

    # 9. Learning curve
    logger.info(f"\n=== Learning Curve ===")
    curve = compute_learning_curve(traces, n_folds=5, seed=cfg.markov.seed)
    for point in curve:
        logger.info(f"  {point.num_traces} traces: accuracy={point.accuracy:.3f} ± {point.accuracy_std:.3f}")

    _save_json({
        "points": [{"num_traces": p.num_traces, "accuracy": p.accuracy,
                     "accuracy_std": p.accuracy_std, "log_likelihood": p.log_likelihood,
                     "log_likelihood_std": p.log_likelihood_std} for p in curve],
    }, output_dir / "learning_curve.json")

    # 10. Absorption lookup for Phase 4
    lookup = build_absorption_lookup(ar, horizon=5)
    _save_json({"horizon": 5, "lookup": lookup}, output_dir / "absorption_lookup.json")
    logger.info(f"\n=== Done. Results saved to {output_dir} ===")


if __name__ == "__main__":
    main()
