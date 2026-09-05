"""Replay frozen VPT trace counts with measured proposal/refine distributions.

This replay is deliberately analytical: semantic hits and episode-bootstrap
intervals come from the hash-frozen VPT test artifact, while systems costs come
from measured single-H200 samples.  It never substitutes intent hits for exact
action hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
TRACE_RESULTS = V2 / "speculation_traces" / "results"
INCREMENTAL_SUMMARY = (
    V2 / "results" / "incremental_decode" / "stop_wallclock_chain_v1" / "summary.json"
)
PERCENTILES = (50, 90, 95, 99)
LEAD_WINDOW_MS = 750.0
MONTE_CARLO_DRAWS = 20_000
MONTE_CARLO_SEED = 20260821


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    return {
        "count": len(samples),
        **{f"p{q}_ms": percentile(samples, q) for q in PERCENTILES},
        "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _trace_rows() -> list[dict[str, Any]]:
    path = TRACE_RESULTS / "trace_metrics.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if (
            row["predictor"] == "markov_history_3"
            and row["subset"] == "all"
            and row["target"] in {"strict_hash", "intent"}
        ):
            selected.append(
                {
                    "target": row["target"],
                    "budget": int(row["budget"]),
                    "eligible_block_count": int(row["eligible_block_count"]),
                    "hit_count": int(row["hit_count"]),
                    "miss_count": int(row["miss_count"]),
                    "hit_rate": float(row["hit_rate"]),
                    "hit_rate_ci95": [
                        float(row["hit_rate_ci95_low"]),
                        float(row["hit_rate_ci95_high"]),
                    ],
                }
            )
    if len(selected) != 6:
        raise RuntimeError(f"expected six frozen trace rows, found {len(selected)}")
    return selected


def _rank_bands(rows: list[dict[str, Any]], target: str) -> dict[str, int]:
    hits = {
        row["budget"]: row["hit_count"] for row in rows if row["target"] == target
    }
    return {
        "rank_1": hits[1],
        "rank_2": hits[2] - hits[1],
        "rank_3_or_4": hits[4] - hits[2],
        "miss_at_4": next(
            row["miss_count"]
            for row in rows
            if row["target"] == target and row["budget"] == 4
        ),
    }


def _decode_samples() -> tuple[list[float], list[float]]:
    report = json.loads(INCREMENTAL_SUMMARY.read_text())
    device = [
        float(row["action_receipt_to_device_ready_ms"])
        - float(row["action_receipt_to_context_commit_ms"])
        for row in report["runs"]
    ]
    host_copy = [
        float(row["device_ready_to_host_ready_ms"]) for row in report["runs"]
    ]
    return device, host_copy


def _latency_model(
    proposal_samples: list[float],
    *,
    refinement_ms: float,
    commit_ms: float,
    full_denoise_ms: float,
    decode_device_samples: list[float],
    host_copy_samples: list[float],
    budget: int,
) -> dict[str, Any]:
    rng = random.Random(MONTE_CARLO_SEED + budget)
    proposal_by_rank = [[] for _ in range(budget)]
    final_by_rank = [[] for _ in range(budget)]
    host_by_rank = [[] for _ in range(budget)]
    miss_final = []
    miss_host = []
    for _ in range(MONTE_CARLO_DRAWS):
        cumulative = 0.0
        decode_index = rng.randrange(len(decode_device_samples))
        decode_ms = decode_device_samples[decode_index]
        copy_ms = host_copy_samples[decode_index]
        for rank in range(budget):
            cumulative += proposal_samples[rng.randrange(len(proposal_samples))]
            proposal_by_rank[rank].append(cumulative)
            final = cumulative + refinement_ms + commit_ms
            final_by_rank[rank].append(final)
            host_by_rank[rank].append(final + decode_ms + copy_ms)
        miss = cumulative + full_denoise_ms + commit_ms
        miss_final.append(miss)
        miss_host.append(miss + decode_ms + copy_ms)
    return {
        "proposal_ready_by_rank": [
            {
                "rank": index + 1,
                "latency": summary(values),
                "p95_within_750ms": percentile(values, 95) <= LEAD_WINDOW_MS,
            }
            for index, values in enumerate(proposal_by_rank)
        ],
        "final_commit_ready_by_hit_rank": [
            {"rank": index + 1, "latency": summary(values)}
            for index, values in enumerate(final_by_rank)
        ],
        "host_frame_ready_by_hit_rank": [
            {"rank": index + 1, "latency": summary(values)}
            for index, values in enumerate(host_by_rank)
        ],
        "miss_final_commit_ready": summary(miss_final),
        "miss_host_frame_ready": summary(miss_host),
    }


def replay(
    *,
    timing_path: Path,
    exact_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    timing = json.loads(timing_path.read_text())
    exact = json.loads(exact_report_path.read_text())
    if timing["status"] != "complete" or exact["status"] != "complete":
        raise RuntimeError("input reports must be complete")
    phase = "steady_roll"
    arm = "shallow_conditioning"
    proposal_samples = [
        float(row["host_sync_ms"])
        for row in timing["boundaries"][phase]["samples"][arm]
    ]
    if not timing["boundaries"][phase]["one_block_exactness"]["exact"]:
        raise RuntimeError("selected timing arm failed one-block exactness")
    components = exact["profiles"][phase]["components"]
    refinement_ms = sum(
        float(row["host_sync_ms"])
        for row in components["four_denoise_forwards"][1:]
    )
    full_denoise_ms = float(components["four_denoise_host_sync_ms"])
    commit_ms = float(components["context_commit"]["host_sync_ms"])
    decode_device, host_copy = _decode_samples()
    trace_rows = _trace_rows()

    system_models = {
        budget: _latency_model(
            proposal_samples,
            refinement_ms=refinement_ms,
            commit_ms=commit_ms,
            full_denoise_ms=full_denoise_ms,
            decode_device_samples=decode_device,
            host_copy_samples=host_copy,
            budget=budget,
        )
        for budget in (1, 2, 4)
    }
    proposal_mean = sum(proposal_samples) / len(proposal_samples)
    decode_mean = sum(decode_device) / len(decode_device)
    copy_mean = sum(host_copy) / len(host_copy)
    projections = []
    for row in trace_rows:
        budget = row["budget"]
        blocks = row["eligible_block_count"]
        hits = row["hit_count"]
        misses = row["miss_count"]
        candidate_count = blocks * budget
        abandoned = candidate_count - hits
        charged = {
            "proposal_candidates": candidate_count,
            "proposal_ms": candidate_count * proposal_mean,
            "abandoned_candidates": abandoned,
            "abandoned_work_ms": abandoned * proposal_mean,
            "refinement_invocations": hits,
            "refinement_ms": hits * refinement_ms,
            "context_commits": blocks,
            "context_commit_ms": blocks * commit_ms,
            "miss_fallbacks": misses,
            "miss_fallback_denoise_ms": misses * full_denoise_ms,
            "decode_invocations": blocks,
            "decode_device_ms": blocks * decode_mean,
            "host_copy_ms": blocks * copy_mean,
        }
        charged["total_charged_ms"] = (
            charged["proposal_ms"]
            + charged["refinement_ms"]
            + charged["context_commit_ms"]
            + charged["miss_fallback_denoise_ms"]
            + charged["decode_device_ms"]
            + charged["host_copy_ms"]
        )
        peak = max(
            int(sample["peak_allocated_bytes"])
            for sample in timing["boundaries"][phase]["samples"][arm]
        )
        projections.append(
            {
                **row,
                "semantic_scope": (
                    "exact action tensor sequence"
                    if row["target"] == "strict_hash"
                    else "approximate train-fitted intent token; not exact"
                ),
                "rank_bands_at_budget_4": _rank_bands(trace_rows, row["target"]),
                "systems_latency": system_models[budget],
                "proposal_ready_hit_count_at_p95": 0,
                "proposal_ready_hit_rate_at_p95": 0.0,
                "confidence_and_rank_gate_passed": False,
                "charged": charged,
                "peak_allocated_bytes_single_candidate": peak,
                "multi_candidate_peak": (
                    "not measured; B>1 rows are charged serial projections"
                    if budget > 1
                    else peak
                ),
            }
        )

    canonical_stop_path = V2 / "results" / "canonical_stop_scoring" / "statistics.json"
    canonical_stop = json.loads(canonical_stop_path.read_text())
    identity = {
        "schema": "rtwm-v2-proposal-trace-replay-1",
        "sources": {
            "trace_replay.py": sha256_file(Path(__file__)),
            "timing_samples": sha256_file(timing_path),
            "h200_exact": sha256_file(exact_report_path),
            "trace_metrics": sha256_file(TRACE_RESULTS / "trace_metrics.csv"),
            "trace_summary": sha256_file(TRACE_RESULTS / "trace_summary.json"),
            "incremental_decode_summary": sha256_file(INCREMENTAL_SUMMARY),
            "canonical_stop_statistics": sha256_file(canonical_stop_path),
        },
        "protocol": {
            "timing_arm": arm,
            "timing_phase": phase,
            "lead_window_ms": LEAD_WINDOW_MS,
            "monte_carlo_draws": MONTE_CARLO_DRAWS,
            "monte_carlo_seed": MONTE_CARLO_SEED,
            "rank_policy": (
                "rank 1, rank 2, and combined rank 3-or-4 counts are inferred "
                "from cumulative frozen B=1/2/4 hit counts"
            ),
        },
    }
    identity["identity_sha256"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    report = {
        "schema_version": "rtwm-v2-proposal-trace-replay-1",
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "frozen_vpt_test": {
            "episode_count": 85,
            "eligible_block_count": 28310,
            "bootstrap": {
                "unit": "whole test episode",
                "draws": 10000,
                "interval": "percentile 95%",
                "source": "speculation_traces/results/trace_summary.json",
            },
        },
        "measured_inputs": {
            "proposal": summary(proposal_samples),
            "refinement_host_ms": refinement_ms,
            "context_commit_host_ms": commit_ms,
            "full_denoise_host_ms": full_denoise_ms,
            "decode_device": summary(decode_device),
            "device_to_host_copy": summary(host_copy),
        },
        "projections": projections,
        "canonical_stop_caol_only": {
            "trace_wide_caol_reported": False,
            "reason": "arbitrary VPT traces are not canonical STOP interventions",
            "external_exact_stop_context": {
                "d0": canonical_stop["by_delay"]["d0"]["metrics"],
                "d1": canonical_stop["by_delay"]["d1"]["metrics"],
                "source": "results/canonical_stop_scoring/statistics.json",
            },
        },
        "intent_semantic_gate": {
            "passed": False,
            "final_latent_error_measured": False,
            "raw_uint8_error_measured": False,
            "revision_magnitude_measured": False,
            "long_horizon_drift_measured": False,
            "perceptual_equivalence_claim": False,
            "reason": (
                "intent representatives are approximate and the exact B=1 "
                "proposal readiness prerequisite failed before a semantic arm "
                "could be retained"
            ),
        },
        "claims": {
            "single_h200": True,
            "proposal_p95_gate_passed": False,
            "trace_readiness_passed": False,
            "exact_hash_and_intent_separated": True,
            "all_generated_and_abandoned_work_charged": True,
            "b2_or_multi_gpu_claim": False,
        },
    }
    atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing",
        type=Path,
        default=HERE / "results" / "timing_samples_unfused.json",
    )
    parser.add_argument(
        "--exact",
        type=Path,
        default=HERE / "results" / "h200_exact.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "trace_replay.json",
    )
    args = parser.parse_args()
    report = replay(
        timing_path=args.timing,
        exact_report_path=args.exact,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "projection_count": len(report["projections"]),
                "trace_readiness_passed": report["claims"]["trace_readiness_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
