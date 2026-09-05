"""Evaluate the frozen intent-confidence gate on held-out pre-roll traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "speculation_traces"
if __package__:
    from .pre_roll_confidence import (
        atomic_bytes,
        atomic_json,
        fit_counts,
        score_episodes,
    )
    from speculation_traces.common import canonical_json_bytes, load_json, sha256_file
    from speculation_traces.speculation_frontier import _load_episodes, construct_targets
else:
    sys.path.insert(0, str(TRACES))
    from common import canonical_json_bytes, load_json, sha256_file
    from pre_roll_confidence import (
        atomic_bytes,
        atomic_json,
        fit_counts,
        score_episodes,
    )
    from speculation_frontier import _load_episodes, construct_targets


def _bootstrap_saved(
    rows: list[dict[str, Any]],
    *,
    net_hit_saved: float,
    miss_penalty: float,
    draws: int,
    seed: int,
) -> list[float]:
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    saved = np.zeros(len(episode_ids), dtype=np.float64)
    index = {episode_id: position for position, episode_id in enumerate(episode_ids)}
    for row in rows:
        if row["selected"]:
            saved[index[int(row["episode_id"])]] += (
                net_hit_saved if row["hit"] else -miss_penalty
            )
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0, len(episode_ids), size=(draws, len(episode_ids))
    )
    totals = saved[sampled].sum(axis=1)
    return [
        float(np.quantile(totals, 0.025, method="linear")),
        float(np.quantile(totals, 0.975, method="linear")),
    ]


def _csv(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def run(
    *,
    config_path: Path,
    freeze_path: Path,
    benchmark_path: Path,
    semantic_path: Path,
    output_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    freeze = load_json(freeze_path)
    benchmark = load_json(benchmark_path)
    semantic = load_json(semantic_path)
    if freeze["identity"]["config_sha256"] != sha256_file(config_path):
        raise ValueError("confidence freeze config hash mismatch")
    if freeze["identity"]["pre_roll_benchmark_sha256"] != sha256_file(
        benchmark_path
    ):
        raise ValueError("confidence freeze timing hash mismatch")
    blocks_path = TRACES / "manifests" / "blocks.json"
    manifest = load_json(blocks_path)
    episodes = _load_episodes(manifest)
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    thresholds = freeze["training"]["intent_thresholds"]
    train_targets = construct_targets(train, thresholds)
    test_targets = construct_targets(test, thresholds)
    model = fit_counts(train_targets, "intent")
    scores = [
        row
        for row in score_episodes(test_targets, model, "intent")
        if row["pre_roll_eligible"]
    ]
    operating = freeze["frozen_operating_point"]
    threshold = operating["threshold"]
    flat_rows = []
    for row in scores:
        selected = bool(
            operating["mode"] == "propose"
            and float(row["confidence"]) >= float(threshold)
        )
        flat_rows.append(
            {
                "episode_id": row["episode_id"],
                "block_index": row["block_index"],
                "prediction": row["prediction"],
                "actual": row["actual"],
                "confidence": row["confidence"],
                "context_order": row["context_order"],
                "selected": selected,
                "intent_hit": row["hit"],
            }
        )
    selected = [row for row in flat_rows if row["selected"]]
    hits = sum(bool(row["intent_hit"]) for row in selected)
    misses = len(selected) - hits
    costs = freeze["measured_cost_model"]
    baseline = float(costs["baseline_action_to_host_ms"])
    net_hit_saved = float(costs["net_hit_saved_ms_after_charging_proposal"])
    miss_penalty = float(
        costs["miss_penalty_ms_after_charging_abandoned_proposal"]
    )
    saved = hits * net_hit_saved - misses * miss_penalty
    all_on_demand = len(flat_rows) * baseline
    charged = all_on_demand - saved
    bootstrap = config["confidence_gate"]["bootstrap"]
    saved_ci = _bootstrap_saved(
        flat_rows,
        net_hit_saved=net_hit_saved,
        miss_penalty=miss_penalty,
        draws=int(bootstrap["draws"]),
        seed=int(bootstrap["seed"]),
    )
    canonical_readiness = bool(
        semantic["gates"]["pre_roll_proposal_readiness"]["passed"]
    )
    systems_p95 = float(
        semantic["aggregate"]["proposal_pre_action_ms"]["p95_ms"]
    )
    readiness_pass = bool(canonical_readiness and systems_p95 <= 750.0)
    semantic_pass = bool(semantic["gates"]["semantic_gate"]["passed"])
    less_than = charged < all_on_demand
    ci_excludes_zero = saved_ci[0] > 0.0
    positive = bool(
        readiness_pass and semantic_pass and less_than and ci_excludes_zero
    )
    report = {
        "schema_version": "rtwm-v2-intent-confidence-heldout-1",
        "status": "complete",
        "identity": {
            "config_sha256": sha256_file(config_path),
            "confidence_freeze_sha256": sha256_file(freeze_path),
            "freeze_identity_sha256": freeze["identity"]["freeze_sha256"],
            "pre_roll_benchmark_sha256": sha256_file(benchmark_path),
            "semantic_benchmark_sha256": sha256_file(semantic_path),
            "blocks_manifest_sha256": sha256_file(blocks_path),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "frozen_operating_point": operating,
        "heldout": {
            "split": "test",
            "episode_count": len(test),
            "pre_roll_eligible_count": len(flat_rows),
            "selected_count": len(selected),
            "coverage": len(selected) / len(flat_rows),
            "intent_hit_count": hits,
            "intent_miss_count": misses,
            "intent_hit_precision": hits / len(selected) if selected else None,
            "total_charged_work_ms": charged,
            "all_on_demand_work_ms": all_on_demand,
            "saved_work_ms": saved,
            "saved_work_ci95_ms": saved_ci,
            "bootstrap_unit": "whole test episode",
            "bootstrap_draws": int(bootstrap["draws"]),
        },
        "cost_model": costs,
        "gates": {
            "pre_roll_p95_at_most_750ms": readiness_pass,
            "canonical_block_0_readiness": canonical_readiness,
            "heldout_pre_roll_proposal_p95_ms": systems_p95,
            "semantic_gate": semantic_pass,
            "heldout_charged_work_less_than_on_demand": less_than,
            "heldout_saved_work_ci95_excludes_zero": ci_excludes_zero,
            "positive_scoped_serving_claim": positive,
        },
        "claims": {
            "serving_claim": positive,
            "exact_acceptance": False,
            "directional_obedience": False,
            "perceptual_equivalence": False,
            "rolling_gate_modified": False,
            "paper_modified": False,
        },
    }
    report["identity"]["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(report["identity"])
    ).hexdigest()
    atomic_json(output_path, report)
    atomic_bytes(csv_path, _csv(flat_rows))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "intent_pre_roll_config.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=HERE / "manifests" / "intent_confidence_freeze.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=HERE / "results" / "pre_roll_b1_benchmark.json",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=HERE / "results" / "intent_semantic_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "intent_confidence_heldout.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=HERE / "tables" / "intent_confidence_heldout.csv",
    )
    args = parser.parse_args()
    report = run(
        config_path=args.config,
        freeze_path=args.freeze,
        benchmark_path=args.benchmark,
        semantic_path=args.semantic,
        output_path=args.output,
        csv_path=args.csv,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "heldout": report["heldout"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
