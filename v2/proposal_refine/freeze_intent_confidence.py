"""Freeze the train-only intent-confidence operating threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "speculation_traces"
if __package__:
    from .pre_roll_confidence import atomic_json, crossfit_scores
    from speculation_traces.common import canonical_json_bytes, load_json, sha256_file
    from speculation_traces.speculation_frontier import (
        _load_episodes,
        construct_targets,
        fit_intent_thresholds,
    )
else:
    sys.path.insert(0, str(TRACES))
    from common import canonical_json_bytes, load_json, sha256_file
    from pre_roll_confidence import atomic_json, crossfit_scores
    from speculation_frontier import (
        _load_episodes,
        construct_targets,
        fit_intent_thresholds,
    )


Z_ONE_SIDED_95 = 1.6448536269514722


def wilson_lower(hits: int, selected: int, z: float = Z_ONE_SIDED_95) -> float:
    if selected == 0:
        return 0.0
    proportion = hits / selected
    denominator = 1.0 + z * z / selected
    center = proportion + z * z / (2.0 * selected)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / selected
        + z * z / (4.0 * selected * selected)
    )
    return (center - radius) / denominator


def measured_costs(benchmark: Mapping[str, Any]) -> dict[str, float]:
    timing = benchmark["cost_model"]
    proposal = float(timing["proposal_pre_action_mean_ms"])
    baseline = float(timing["baseline_action_to_host_mean_ms"])
    hit_action = float(timing["hit_action_to_host_mean_ms"])
    miss_action = float(timing["miss_action_to_host_mean_ms"])
    gross_hit_action_saving = baseline - hit_action
    net_hit_saved = gross_hit_action_saving - proposal
    miss_penalty = proposal + miss_action - baseline
    break_even = miss_penalty / (net_hit_saved + miss_penalty)
    return {
        "proposal_pre_action_ms": proposal,
        "baseline_action_to_host_ms": baseline,
        "hit_action_to_host_ms": hit_action,
        "miss_action_to_host_ms": miss_action,
        "gross_hit_action_saving_ms": gross_hit_action_saving,
        "net_hit_saved_ms_after_charging_proposal": net_hit_saved,
        "miss_penalty_ms_after_charging_abandoned_proposal": miss_penalty,
        "break_even_intent_precision": break_even,
    }


def _row(
    scores: Sequence[Mapping[str, Any]],
    threshold: float,
    costs: Mapping[str, float],
) -> dict[str, Any]:
    eligible = [row for row in scores if row["pre_roll_eligible"]]
    selected = [
        row for row in eligible if float(row["confidence"]) >= threshold
    ]
    hits = sum(bool(row["hit"]) for row in selected)
    misses = len(selected) - hits
    lower = wilson_lower(hits, len(selected))
    net_saved = (
        hits * float(costs["net_hit_saved_ms_after_charging_proposal"])
        - misses
        * float(costs["miss_penalty_ms_after_charging_abandoned_proposal"])
    )
    return {
        "threshold": threshold,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "hit_count": hits,
        "miss_count": misses,
        "coverage": len(selected) / len(eligible),
        "intent_precision": hits / len(selected) if selected else None,
        "intent_precision_wilson_one_sided_95_lower": lower,
        "exceeds_break_even_lower_bound": lower
        > float(costs["break_even_intent_precision"]),
        "expected_net_saved_ms": net_saved,
        "expected_net_saved_ms_per_eligible": net_saved / len(eligible),
    }


def run(config_path: Path, benchmark_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    benchmark = load_json(benchmark_path)
    blocks_path = TRACES / "manifests" / "blocks.json"
    block_manifest = load_json(blocks_path)
    train_manifest = {
        **block_manifest,
        "files": [
            entry
            for entry in block_manifest["files"]
            if entry["split"] == "train"
        ],
    }
    train = _load_episodes(train_manifest)
    thresholds = fit_intent_thresholds(train)
    targets = construct_targets(train, thresholds)
    scores = crossfit_scores(targets, "intent")
    costs = measured_costs(benchmark)
    values = sorted(
        {
            float(row["confidence"])
            for row in scores
            if row["pre_roll_eligible"]
        }
    )
    frontier = [_row(scores, threshold, costs) for threshold in values]
    admissible = [
        row
        for row in frontier
        if row["exceeds_break_even_lower_bound"]
        and row["expected_net_saved_ms"] > 0.0
    ]
    selected = (
        min(
            admissible,
            key=lambda row: (
                -float(row["expected_net_saved_ms"]),
                -float(row["threshold"]),
                float(row["coverage"]),
            ),
        )
        if admissible
        else None
    )
    report = {
        "schema_version": "rtwm-v2-intent-confidence-freeze-1",
        "status": "frozen_before_heldout_scoring",
        "identity": {
            "config_sha256": sha256_file(config_path),
            "pre_roll_benchmark_sha256": sha256_file(benchmark_path),
            "blocks_manifest_sha256": sha256_file(blocks_path),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "protocol": {
            "split_accessed": "train only",
            "test_labels_accessed": False,
            "calibration": "deterministic five-fold episode cross-fit",
            "precision_lower_bound": (
                "one-sided 95% Wilson score lower bound; "
                f"z={Z_ONE_SIDED_95}"
            ),
            "candidate_threshold_count": len(values),
            "reject_all_if_no_admissible_threshold": True,
        },
        "training": {
            "episode_count": len(train),
            "pre_roll_decision_count": sum(
                row["pre_roll_eligible"] for row in scores
            ),
            "intent_thresholds": thresholds,
        },
        "measured_cost_model": costs,
        "frontier": frontier,
        "frozen_operating_point": (
            {
                "mode": "propose",
                **selected,
            }
            if selected is not None
            else {
                "mode": "reject_all",
                "threshold": None,
                "reason": (
                    "no train-only threshold has a conservative intent-precision "
                    "lower bound above measured break-even with positive net work"
                ),
            }
        ),
    }
    report["identity"]["freeze_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "intent_pre_roll_config.json"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=HERE / "results" / "pre_roll_b1_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "manifests" / "intent_confidence_freeze.json",
    )
    args = parser.parse_args()
    report = run(args.config, args.benchmark, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "operating_point": report["frozen_operating_point"],
                "break_even": report["measured_cost_model"][
                    "break_even_intent_precision"
                ],
                "freeze_sha256": report["identity"]["freeze_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
