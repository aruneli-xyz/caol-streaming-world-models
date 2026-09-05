"""Run the frozen held-out approximate-intent semantic gate on one H200."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from einops import rearrange
from skimage.metrics import structural_similarity

if __package__:
    from . import gpu_experiment as experiment
    from .conditioning import make_shallow_conditioning_x0
    from .core import ExactProposalTransaction
    from .intent_refine import accept_intent_refine
    from .pre_roll_b1_benchmark import decode_full
else:
    import gpu_experiment as experiment
    from conditioning import make_shallow_conditioning_x0
    from core import ExactProposalTransaction
    from intent_refine import accept_intent_refine
    from pre_roll_b1_benchmark import decode_full

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "intent_pre_roll_config.json"
SELECTION = HERE / "manifests" / "intent_heldout_selection.json"
OUTPUT = HERE / "results" / "intent_semantic_benchmark.json"


def _stream(
    decision: Mapping[str, Any], *, representative_at_target: bool
) -> tuple[np.ndarray, np.ndarray]:
    keyboard = np.asarray(
        decision["prefix_and_continuation_keyboard"], dtype=np.float32
    )
    camera = np.asarray(
        decision["prefix_and_continuation_camera_degrees"], dtype=np.float32
    )
    target = int(decision["block_index"])
    if representative_at_target:
        keyboard[target] = np.asarray(
            decision["representative_keyboard"], dtype=np.float32
        )
        camera[target] = np.asarray(
            decision["representative_camera_degrees"], dtype=np.float32
        )
    keyboard = keyboard.reshape(-1, 23)
    camera = camera.reshape(-1, 2)
    missing = experiment.N_FRAMES - keyboard.shape[0]
    if missing > 0:
        keyboard = np.concatenate(
            [keyboard, np.repeat(keyboard[-1:], missing, axis=0)]
        )
        camera = np.concatenate(
            [camera, np.repeat(camera[-1:], missing, axis=0)]
        )
    return keyboard[: experiment.N_FRAMES], camera[: experiment.N_FRAMES]


def _mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(
        torch.mean(
            (reference.float().cpu() - candidate.float().cpu()).square()
        ).item()
    )


def _raw_metrics(
    reference: np.ndarray, candidate: np.ndarray, support_frame: int
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError("raw output shape mismatch")
    before_reference = reference[:, :support_frame]
    before_candidate = candidate[:, :support_frame]
    before_exact = bool(np.array_equal(before_reference, before_candidate))
    by_view = []
    for view in range(reference.shape[0]):
        first = reference[view, support_frame:].astype(np.float64)
        second = candidate[view, support_frame:].astype(np.float64)
        mse = float(np.mean((first - second) ** 2))
        psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))
        ssim_frames = [
            float(
                structural_similarity(
                    first[frame].astype(np.uint8),
                    second[frame].astype(np.uint8),
                    data_range=255,
                    channel_axis=-1,
                )
            )
            for frame in range(first.shape[0])
        ]
        by_view.append(
            {
                "view": view,
                "frame_count": first.shape[0],
                "mse": mse,
                "psnr_db": psnr,
                "ssim_mean": float(np.mean(ssim_frames)),
                "ssim_min_frame": float(np.min(ssim_frames)),
            }
        )
    return {
        "shape": list(reference.shape),
        "decoder_support_frame": support_frame,
        "before_support_exact": before_exact,
        "before_support_changed_elements": int(
            np.count_nonzero(before_reference != before_candidate)
        ),
        "by_view": by_view,
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
    }


def _decode_output(session: Any, end_block: int) -> np.ndarray:
    output_end = end_block * session.nfpb
    output_views = rearrange(
        session.output,
        "b (v t) c h w -> b v t c h w",
        v=session.n_views,
    )[:, :, :output_end]
    prefix = rearrange(
        output_views, "b v t c h w -> b (v t) c h w"
    ).contiguous()
    return decode_full(session.model.tokenizer, prefix, n_views=session.n_views)


def _run_sequence(
    *,
    engine: Any,
    batch: Mapping[str, Any],
    actual_x0: Any,
    actual_tensors: Mapping[str, torch.Tensor],
    representative_x0: Any,
    representative_tensors: Mapping[str, torch.Tensor],
    target: int,
    source_hashes: Mapping[str, str],
    approximate: bool,
    sample_id: int,
) -> dict[str, Any]:
    session = experiment.new_session(engine, batch, 1701)
    prefix = f"intent-heldout:{sample_id}"
    experiment.advance_to(
        session,
        actual_x0,
        actual_tensors,
        target,
        source_hashes,
        prefix,
    )
    logical = f"{prefix}:block:{target}"
    proposal_ms = 0.0
    if approximate:
        transaction = ExactProposalTransaction(
            session, logical_session_id=logical, source_hashes=source_hashes
        )
        proposal, proposal_timing = experiment.timed_cuda(
            lambda: transaction.propose(representative_x0, representative_tensors)
        )
        proposal_ms = float(proposal_timing["host_sync_ms"])
        first, refine_timing = experiment.timed_cuda(
            lambda: accept_intent_refine(
                transaction, proposal, actual_x0, actual_tensors
            )
        )
        target_compute_ms = float(refine_timing["host_sync_ms"])
    else:
        transaction = ExactProposalTransaction(
            session, logical_session_id=logical, source_hashes=source_hashes
        )
        first, full_timing = experiment.timed_cuda(
            lambda: transaction.run_full(actual_x0, actual_tensors)
        )
        target_compute_ms = float(full_timing["host_sync_ms"])
    first_latent = first.latent.detach().cpu().clone()
    decode_started = time.perf_counter()
    raw = _decode_output(session, target + 1)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    continuation_ms = 0.0
    final = first
    for block in range(target + 1, target + 5):
        continuation = ExactProposalTransaction(
            session,
            logical_session_id=f"{prefix}:block:{block}",
            source_hashes=source_hashes,
        )
        final, timing = experiment.timed_cuda(
            lambda transaction=continuation: transaction.run_full(
                actual_x0, actual_tensors
            )
        )
        continuation_ms += float(timing["host_sync_ms"])
    result = {
        "first_latent": first_latent,
        "continuation_final_latent": final.latent.detach().cpu().clone(),
        "continuation_output": session.output.detach().cpu().clone(),
        "raw": raw,
        "timing": {
            "proposal_pre_action_ms": proposal_ms,
            "target_compute_ms": target_compute_ms,
            "semantic_full_prefix_decode_ms": decode_ms,
            "four_continuation_blocks_ms": continuation_ms,
            "total_scoring_work_ms": (
                proposal_ms + target_compute_ms + decode_ms + continuation_ms
            ),
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    session.model.kv_cache1 = None
    del session
    torch.cuda.empty_cache()
    return result


def benchmark(output: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    selection = json.loads(SELECTION.read_text())
    base_config = json.loads(experiment.CONFIG.read_text())
    if selection["identity"]["config_sha256"] != experiment.sha256_file(CONFIG):
        raise ValueError("selection was not frozen against current config")
    pre_roll_benchmark = HERE / "results" / "pre_roll_b1_benchmark.json"
    readiness = json.loads(pre_roll_benchmark.read_text())
    identity = experiment.source_identity()
    identity["intent_config_sha256"] = experiment.sha256_file(CONFIG)
    identity["selection_file_sha256"] = experiment.sha256_file(SELECTION)
    identity["selection_sha256"] = selection["identity"]["selection_sha256"]
    identity["pre_roll_benchmark_sha256"] = experiment.sha256_file(
        pre_roll_benchmark
    )
    identity["semantic_benchmark_source_sha256"] = experiment.sha256_file(
        Path(__file__)
    )
    identity["intent_refine_source_sha256"] = experiment.sha256_file(
        HERE / "intent_refine.py"
    )
    identity["identity_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(identity)
    ).hexdigest()

    engine = experiment.build_engine(base_config)
    images = experiment.load_images()
    source_hashes = identity["sources"] | {
        "intent_refine.py": identity["intent_refine_source_sha256"]
    }
    rows = []
    started = time.perf_counter()
    for decision in selection["decisions"]:
        sample_started = time.perf_counter()
        actual_keyboard, actual_camera = _stream(
            decision, representative_at_target=False
        )
        representative_keyboard, representative_camera = _stream(
            decision, representative_at_target=True
        )
        batch, actual_stock_x0, actual_tensors, actual_conditioning = (
            experiment.make_x0(
                engine, images, actual_keyboard, actual_camera
            )
        )
        actual_x0, _ = make_shallow_conditioning_x0(actual_stock_x0)
        _, representative_stock_x0, representative_tensors, representative_conditioning = (
            experiment.make_x0(
                engine,
                images,
                representative_keyboard,
                representative_camera,
            )
        )
        representative_x0, _ = make_shallow_conditioning_x0(
            representative_stock_x0
        )
        target = int(decision["block_index"])
        torch.cuda.reset_peak_memory_stats()
        reference = _run_sequence(
            engine=engine,
            batch=batch,
            actual_x0=actual_x0,
            actual_tensors=actual_tensors,
            representative_x0=representative_x0,
            representative_tensors=representative_tensors,
            target=target,
            source_hashes=source_hashes,
            approximate=False,
            sample_id=int(decision["sample_id"]),
        )
        candidate = _run_sequence(
            engine=engine,
            batch=batch,
            actual_x0=actual_x0,
            actual_tensors=actual_tensors,
            representative_x0=representative_x0,
            representative_tensors=representative_tensors,
            target=target,
            source_hashes=source_hashes,
            approximate=True,
            sample_id=int(decision["sample_id"]),
        )
        support = 0 if target == 0 else target * 12 - 3
        raw = _raw_metrics(reference["raw"], candidate["raw"], support)
        row = {
            "sample_id": decision["sample_id"],
            "episode_id": decision["episode_id"],
            "block_index": target,
            "intent": decision["intent"],
            "movement_camera_group": decision["movement_camera_group"],
            "confidence": decision["confidence"],
            "actual_strict_hash": decision["actual_strict_hash"],
            "representative_strict_hash": decision[
                "representative_strict_hash"
            ],
            "final_latent_mse": _mse(
                reference["first_latent"], candidate["first_latent"]
            ),
            "continuation_final_latent_mse": _mse(
                reference["continuation_final_latent"],
                candidate["continuation_final_latent"],
            ),
            "continuation_output_mse_diagnostic": _mse(
                reference["continuation_output"],
                candidate["continuation_output"],
            ),
            "raw_uint8": raw,
            "reference_timing": reference["timing"],
            "candidate_timing": candidate["timing"],
            "conditioning_host_sync_ms": {
                "actual": actual_conditioning,
                "representative": representative_conditioning,
            },
            "peak_allocated_bytes": max(
                reference["peak_allocated_bytes"],
                candidate["peak_allocated_bytes"],
            ),
            "sample_wall_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        experiment.atomic_json(
            output,
            {
                "schema_version": "rtwm-v2-intent-semantic-benchmark-1",
                "status": "running",
                "identity": identity,
                "completed_sample_count": len(rows),
                "samples": rows,
            },
        )
        del reference, candidate
        torch.cuda.empty_cache()

    thresholds = config["semantic_gate"]
    final_mse_max = max(row["final_latent_mse"] for row in rows)
    continuation_mse_max = max(
        row["continuation_final_latent_mse"] for row in rows
    )
    psnr_min = [
        min(row["raw_uint8"]["by_view"][view]["psnr_db"] for row in rows)
        for view in range(2)
    ]
    ssim_min = [
        min(row["raw_uint8"]["by_view"][view]["ssim_mean"] for row in rows)
        for view in range(2)
    ]
    before_support_exact = all(
        row["raw_uint8"]["before_support_exact"] for row in rows
    )
    components = {
        "final_latent_mse": final_mse_max
        <= float(thresholds["final_latent_mse_max"]),
        "raw_uint8_psnr_both_views": all(
            value >= float(thresholds["raw_uint8_psnr_min_db_per_view"])
            for value in psnr_min
        ),
        "raw_uint8_ssim_both_views": all(
            value >= float(thresholds["raw_uint8_ssim_min_per_view"])
            for value in ssim_min
        ),
        "no_raw_change_before_decoder_support": before_support_exact,
        "four_block_continuation_latent_mse": continuation_mse_max
        <= float(thresholds["continuation_final_latent_mse_max"]),
    }
    proposal_samples = [
        row["candidate_timing"]["proposal_pre_action_ms"] for row in rows
    ]
    report = {
        "schema_version": "rtwm-v2-intent-semantic-benchmark-1",
        "status": "complete",
        "identity": identity,
        "scope": {
            "gpu": torch.cuda.get_device_name(0),
            "device_count": 1,
            "batch_size": 1,
            "cache_phase": "pre_roll_only",
            "sample_count": len(rows),
            "selection_frozen_before_scoring": True,
        },
        "protocol": {
            "approximate_path": (
                "one train-representative intent denoise step, three exact "
                "actual-action refinement steps, exact actual context commit"
            ),
            "continuation": "four exact actual-action full blocks",
            "raw_metric_definition": thresholds["ssim_definition"],
            "all_scoring_work_charged": True,
        },
        "samples": rows,
        "aggregate": {
            "final_latent_mse_max": final_mse_max,
            "continuation_final_latent_mse_max": continuation_mse_max,
            "raw_uint8_psnr_min_db_by_view": psnr_min,
            "raw_uint8_ssim_min_mean_by_view": ssim_min,
            "all_before_support_exact": before_support_exact,
            "proposal_pre_action_ms": experiment.percentiles(proposal_samples),
            "candidate_total_scoring_work_ms": sum(
                row["candidate_timing"]["total_scoring_work_ms"] for row in rows
            ),
            "reference_total_scoring_work_ms": sum(
                row["reference_timing"]["total_scoring_work_ms"] for row in rows
            ),
            "max_peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in rows
            ),
        },
        "gates": {
            "pre_roll_proposal_readiness": {
                "source": "results/pre_roll_b1_benchmark.json",
                "p95_ms": readiness["gates"]["pre_roll_b1_gate"][
                    "proposal_p95_ms"
                ],
                "limit_ms": 750.0,
                "passed": readiness["gates"]["pre_roll_b1_gate"][
                    "proposal_readiness_passed"
                ],
            },
            "semantic_gate": {
                "components": components,
                "passed": all(components.values()),
            },
        },
        "gpu_wall_runtime_seconds": time.perf_counter() - started,
        "claims": {
            "approximate_semantic_gate_only": True,
            "exact_acceptance": False,
            "directional_obedience": False,
            "perceptual_equivalence": False,
            "rolling_readiness": False,
        },
    }
    experiment.atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = benchmark(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "semantic_gate_passed": report["gates"]["semantic_gate"][
                    "passed"
                ],
                "aggregate": report["aggregate"],
                "gpu_wall_runtime_seconds": report[
                    "gpu_wall_runtime_seconds"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
