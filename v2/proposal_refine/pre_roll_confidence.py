"""Train-only confidence gating for frozen exact-action B=1 trace predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
TRACES = V2 / "speculation_traces"
if __package__:
    from speculation_traces.common import canonical_json_bytes, load_json, sha256_file
    from speculation_traces.speculation_frontier import (
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
        HISTORY_ORDER,
        TargetEpisode,
        _load_episodes,
        construct_targets,
        episode_bootstrap_interval,
        fit_intent_thresholds,
        rank_counts,
    )
else:
    sys.path.insert(0, str(TRACES))
    from common import canonical_json_bytes, load_json, sha256_file
    from speculation_frontier import (
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
        HISTORY_ORDER,
        TargetEpisode,
        _load_episodes,
        construct_targets,
        episode_bootstrap_interval,
        fit_intent_thresholds,
        rank_counts,
    )


Top = tuple[str, float, int]
Counts = tuple[
    Top,
    dict[tuple[str, ...], Top],
]


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def fit_counts(episodes: Sequence[TargetEpisode], target: str) -> Counts:
    if not episodes:
        raise ValueError("fit requires episodes")
    global_counts: Counter[str] = Counter()
    transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        sequence = getattr(episode, target)
        global_counts.update(sequence)
        for index, value in enumerate(sequence):
            for order in range(1, min(HISTORY_ORDER, index) + 1):
                transitions[tuple(sequence[index - order : index])][value] += 1
    def top(counts: Mapping[str, int]) -> Top:
        prediction = rank_counts(counts)[0]
        support = int(sum(counts.values()))
        return prediction, int(counts[prediction]) / support, support

    return top(global_counts), {
        context: top(counts) for context, counts in transitions.items()
    }


def predict_with_confidence(model: Counts, history: Sequence[str]) -> dict[str, Any]:
    global_top, transitions = model
    for order in range(min(HISTORY_ORDER, len(history)), 0, -1):
        selected = transitions.get(tuple(history[-order:]))
        if selected:
            prediction, confidence, support = selected
            return {
                "prediction": prediction,
                "confidence": confidence,
                "context_order": order,
                "support": support,
            }
    prediction, confidence, support = global_top
    return {
        "prediction": prediction,
        "confidence": confidence,
        "context_order": 0,
        "support": support,
    }


def score_episodes(
    episodes: Sequence[TargetEpisode],
    model: Counts,
    target: str,
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        sequence = getattr(episode, target)
        for index in range(1, len(sequence)):
            prediction = predict_with_confidence(model, sequence[:index])
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "block_index": index,
                    "pre_roll_eligible": index < 8,
                    "actual": sequence[index],
                    "hit": prediction["prediction"] == sequence[index],
                    **prediction,
                }
            )
    return rows


def crossfit_scores(
    episodes: Sequence[TargetEpisode],
    target: str,
    folds: int = 5,
) -> list[dict[str, Any]]:
    rows = []
    for fold in range(folds):
        fit = [episode for episode in episodes if episode.episode_id % folds != fold]
        held = [episode for episode in episodes if episode.episode_id % folds == fold]
        model = fit_counts(fit, target)
        rows.extend(score_episodes(held, model, target))
    return sorted(rows, key=lambda row: (row["episode_id"], row["block_index"]))


def threshold_values(scores: Sequence[Mapping[str, Any]], quantiles: Iterable[float]) -> list[float]:
    values = np.asarray([row["confidence"] for row in scores], dtype=np.float64)
    return sorted(
        {
            float(value)
            for value in np.quantile(values, list(quantiles), method="higher")
        }
    )


def _rates(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    pre_roll_only: bool,
) -> dict[str, Any]:
    eligible = [
        row for row in rows if not pre_roll_only or row["pre_roll_eligible"]
    ]
    selected = [row for row in eligible if float(row["confidence"]) >= threshold]
    hits = sum(bool(row["hit"]) for row in selected)
    return {
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "hit_count": hits,
        "miss_count": len(selected) - hits,
        "coverage": len(selected) / len(eligible),
        "exact_hit_precision": hits / len(selected) if selected else None,
    }


def _episode_intervals(
    rows: Sequence[Mapping[str, Any]], threshold: float, *, pre_roll_only: bool
) -> dict[str, list[float] | None]:
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    selected = np.zeros(len(episode_ids), dtype=np.int64)
    eligible = np.zeros(len(episode_ids), dtype=np.int64)
    hits = np.zeros(len(episode_ids), dtype=np.int64)
    index = {episode_id: position for position, episode_id in enumerate(episode_ids)}
    for row in rows:
        if pre_roll_only and not row["pre_roll_eligible"]:
            continue
        position = index[int(row["episode_id"])]
        eligible[position] += 1
        if float(row["confidence"]) >= threshold:
            selected[position] += 1
            hits[position] += int(bool(row["hit"]))
    coverage = episode_bootstrap_interval(
        selected, eligible, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    precision = (
        episode_bootstrap_interval(
            hits, selected, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 1
        )
        if selected.sum()
        else None
    )
    return {
        "coverage_ci95": list(coverage),
        "exact_hit_precision_ci95": None if precision is None else list(precision),
    }


def _costs(
    rates: Mapping[str, Any],
    timing: Mapping[str, float],
) -> dict[str, Any]:
    eligible = int(rates["eligible_count"])
    selected = int(rates["selected_count"])
    hits = int(rates["hit_count"])
    misses = int(rates["miss_count"])
    unselected = eligible - selected
    proposal = float(timing["proposal_pre_action_mean_ms"])
    hit_host = float(timing["hit_action_to_host_mean_ms"])
    miss_host = float(timing["miss_action_to_host_mean_ms"])
    baseline = float(timing["baseline_action_to_host_mean_ms"])
    charged = selected * proposal + hits * hit_host + misses * miss_host + unselected * baseline
    baseline_total = eligible * baseline
    return {
        "proposal_count": selected,
        "abandoned_proposal_count": misses,
        "full_fallback_count": misses,
        "on_demand_unselected_count": unselected,
        "charged_work_ms": charged,
        "all_on_demand_work_ms": baseline_total,
        "charged_work_delta_ms": charged - baseline_total,
        "mean_charged_work_ms_per_eligible": charged / eligible,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def run(
    *,
    config_path: Path,
    benchmark_path: Path,
    output_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    benchmark = load_json(benchmark_path)
    if not benchmark["gates"]["pre_roll_b1_gate"]["passed"]:
        raise RuntimeError("measured pre-roll B=1 gate must pass before projection")
    blocks_path = TRACES / "manifests" / "blocks.json"
    block_manifest = load_json(blocks_path)
    episodes = _load_episodes(block_manifest)
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    thresholds = fit_intent_thresholds(train)
    train_targets = construct_targets(train, thresholds)
    test_targets = construct_targets(test, thresholds)
    quantiles = config["confidence"]["threshold_quantiles"]
    timing = benchmark["cost_model"]

    targets: dict[str, Any] = {}
    flat_rows = []
    for target in ("strict_hash", "intent"):
        calibration = crossfit_scores(train_targets, target)
        frozen_thresholds = threshold_values(calibration, quantiles)
        final_model = fit_counts(train_targets, target)
        test_scores = score_episodes(test_targets, final_model, target)
        frontier = []
        for threshold in frozen_thresholds:
            calibration_rates = _rates(
                calibration, threshold, pre_roll_only=True
            )
            test_all = _rates(test_scores, threshold, pre_roll_only=False)
            test_pre = _rates(test_scores, threshold, pre_roll_only=True)
            costs = _costs(test_pre, timing)
            row = {
                "threshold": threshold,
                "calibration_pre_roll": calibration_rates,
                "test_all_semantic_diagnostic": test_all,
                "test_pre_roll": {
                    **test_pre,
                    **_episode_intervals(
                        test_scores, threshold, pre_roll_only=True
                    ),
                },
                "expected_pre_roll_charged_work": costs,
            }
            frontier.append(row)
            flat_rows.append(
                {
                    "target": target,
                    "threshold": threshold,
                    "test_pre_roll_eligible_count": test_pre["eligible_count"],
                    "test_pre_roll_selected_count": test_pre["selected_count"],
                    "test_pre_roll_coverage": test_pre["coverage"],
                    "test_pre_roll_exact_hit_precision": test_pre[
                        "exact_hit_precision"
                    ],
                    "charged_work_delta_ms": costs["charged_work_delta_ms"],
                }
            )
        # Freeze the operating threshold using cross-fit train outcomes and
        # measured path costs only. Test outcomes never select the threshold.
        calibration_costs = [
            (
                _costs(
                    _rates(calibration, threshold, pre_roll_only=True),
                    timing,
                )["mean_charged_work_ms_per_eligible"],
                -threshold,
                threshold,
            )
            for threshold in frozen_thresholds
        ]
        operating_threshold = min(calibration_costs)[2]
        operating = next(
            row for row in frontier if row["threshold"] == operating_threshold
        )
        targets[target] = {
            "scope": (
                "exact acceptance eligible"
                if target == "strict_hash"
                else "diagnostic only; never exact acceptance or CAOL"
            ),
            "calibration": {
                "split": "train only",
                "method": "deterministic five-fold episode cross-fit",
                "episode_count": len(train_targets),
                "thresholds_frozen_before_test": True,
            },
            "operating_threshold": operating_threshold,
            "operating_point": operating,
            "frontier": frontier,
        }

    all_test_rows = score_episodes(
        test_targets, fit_counts(train_targets, "strict_hash"), "strict_hash"
    )
    pre_count = sum(row["pre_roll_eligible"] for row in all_test_rows)
    report = {
        "schema_version": "rtwm-v2-pre-roll-confidence-1",
        "status": "complete",
        "identity": {
            "config_sha256": sha256_file(config_path),
            "benchmark_sha256": sha256_file(benchmark_path),
            "blocks_manifest_sha256": sha256_file(blocks_path),
            "trace_summary_sha256": sha256_file(
                TRACES / "results" / "trace_summary.json"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "eligibility": {
            "all_test_decisions": len(all_test_rows),
            "pre_roll_test_decisions": pre_count,
            "pre_roll_eligible_fraction": pre_count / len(all_test_rows),
            "definition": (
                "episode-local action block indices 1..7; index 8 is the "
                "first rolling-cache boundary"
            ),
            "rolling_projection_performed": False,
        },
        "targets": targets,
        "claims": {
            "exact_hash_and_intent_distinct": True,
            "exact_acceptance_target": "strict_hash only",
            "intent_exact_acceptance": False,
            "intent_caol": False,
            "pre_roll_readiness_only": True,
        },
    }
    report["identity"]["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(report["identity"])
    ).hexdigest()
    atomic_json(output_path, report)
    atomic_bytes(csv_path, _csv_bytes(flat_rows))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "pre_roll_b1_config.json"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=HERE / "results" / "pre_roll_b1_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "pre_roll_confidence.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=HERE / "results" / "pre_roll_confidence_frontier.csv",
    )
    args = parser.parse_args()
    report = run(
        config_path=args.config,
        benchmark_path=args.benchmark,
        output_path=args.output,
        csv_path=args.csv,
    )
    exact = report["targets"]["strict_hash"]["operating_point"]["test_pre_roll"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "pre_roll_eligible_fraction": report["eligibility"][
                    "pre_roll_eligible_fraction"
                ],
                "exact_coverage": exact["coverage"],
                "exact_precision": exact["exact_hit_precision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
