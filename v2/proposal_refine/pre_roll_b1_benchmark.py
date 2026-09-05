"""Measured exact B=1 proposal hit/miss benchmark at the pre-roll boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from einops import rearrange

if __package__:
    from . import gpu_experiment as experiment
    from .conditioning import make_shallow_conditioning_x0
    from .core import ExactProposalTransaction, tensor_sha256
else:
    import gpu_experiment as experiment
    from conditioning import make_shallow_conditioning_x0
    from core import ExactProposalTransaction, tensor_sha256


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "pre_roll_b1_config.json"
OUTPUT = HERE / "results" / "pre_roll_b1_benchmark.json"


class IncrementalDecoder:
    """One exact Wan decoder-cache session with global normalization indices."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.started = False

    def start(self) -> None:
        if self.started:
            raise RuntimeError("incremental decoder already started")
        self.tokenizer.clear_cache()
        self.started = True

    def decode(self, normalized: torch.Tensor, *, global_start: int) -> torch.Tensor:
        if not self.started:
            raise RuntimeError("incremental decoder not started")
        model = self.tokenizer.model
        end = global_start + normalized.shape[2]
        mean = model.video_mean[:, :, global_start:end].to(
            device=normalized.device, dtype=normalized.dtype
        )
        std = model.video_std[:, :, global_start:end].to(
            device=normalized.device, dtype=normalized.dtype
        )
        return model.decode(
            (normalized * std + mean).contiguous(),
            clear_decoder_cache=False,
        )


def raw_uint8_views(decoded: torch.Tensor, n_views: int) -> torch.Tensor:
    raw = ((decoded.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return (
        raw.reshape(1, n_views, 3, *raw.shape[2:])
        .permute(0, 1, 3, 4, 5, 2)
        .contiguous()[0]
    )


def flatten_latent(latent: torch.Tensor, n_views: int) -> torch.Tensor:
    return rearrange(
        latent,
        "b (v t) c h w -> (b v) c t h w",
        v=n_views,
    ).contiguous()


def decode_incremental(
    decoder: IncrementalDecoder,
    latent: torch.Tensor,
    *,
    global_start: int,
    n_views: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    flat = flatten_latent(latent, n_views)
    started = time.perf_counter_ns()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    decoded = decoder.decode(flat, global_start=global_start)
    raw_device = raw_uint8_views(decoded, n_views)
    end_event.record()
    torch.cuda.synchronize()
    device_ready = time.perf_counter_ns()
    raw_host = raw_device.cpu().numpy()
    host_ready = time.perf_counter_ns()
    frames = int(raw_host.shape[1])
    timing = {
        "cuda_event_ms": float(start_event.elapsed_time(end_event)),
        "device_ready_ms": (device_ready - started) / 1e6,
        "host_ready_ms": (host_ready - started) / 1e6,
        "host_copy_ms": (host_ready - device_ready) / 1e6,
        "queued_frames": frames,
        "consumed_frames": frames,
        "dropped_frames": 0,
    }
    del decoded, raw_device
    return raw_host, timing


def decode_full(
    tokenizer: Any, latents: torch.Tensor, *, n_views: int
) -> np.ndarray:
    flat = flatten_latent(latents, n_views)
    model = tokenizer.model
    end = flat.shape[2]
    mean = model.video_mean[:, :, :end].to(device=flat.device, dtype=flat.dtype)
    std = model.video_std[:, :, :end].to(device=flat.device, dtype=flat.dtype)
    tokenizer.clear_cache()
    decoded = model.decode(
        (flat * std + mean).contiguous(), clear_decoder_cache=True
    )
    raw = raw_uint8_views(decoded, n_views).cpu().numpy()
    del decoded
    return raw


def compare_array(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    exact = bool(np.array_equal(first, second))
    return {
        "exact": exact,
        "max_error": (
            0
            if exact
            else int(
                np.abs(first.astype(np.int16) - second.astype(np.int16)).max()
            )
        ),
        "shape": list(first.shape),
        "reference_sha256": hashlib.sha256(
            np.ascontiguousarray(first).tobytes()
        ).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            np.ascontiguousarray(second).tobytes()
        ).hexdigest(),
    }


def timed_transaction(operation: Any) -> tuple[Any, dict[str, float]]:
    return experiment.timed_cuda(operation)


def summarize_path(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "proposal_pre_action_ms",
        "action_to_final_commit_ms",
        "action_to_host_ready_ms",
        "decode_host_ready_ms",
        "total_charged_ms",
    )
    return {
        name: experiment.percentiles(
            [float(row[name]) for row in rows if name in row]
        )
        for name in numeric
        if any(name in row for row in rows)
    } | {
        "count": len(rows),
        "max_peak_allocated_bytes": max(
            int(row["peak_allocated_bytes"]) for row in rows
        ),
        "total_dropped_frames": sum(int(row["dropped_frames"]) for row in rows),
    }


def _restore_parent(session: Any, parent: Any) -> None:
    session.restore_delta(parent)
    torch.cuda.empty_cache()


def run_path(
    *,
    kind: str,
    repetition: int,
    session: Any,
    parent: Any,
    stop_x0: Any,
    stop_tensors: Mapping[str, torch.Tensor],
    unchanged_x0: Any,
    unchanged_tensors: Mapping[str, torch.Tensor],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _restore_parent(session, parent)
    torch.cuda.reset_peak_memory_stats()
    logical = f"pre-roll-b1:timing:{repetition}"
    proposal_ms = None
    if kind == "on_demand_full":
        transaction = ExactProposalTransaction(
            session, logical_session_id=logical, source_hashes=source_hashes
        )
        result, action_timing = timed_transaction(
            lambda: transaction.run_full(stop_x0, stop_tensors)
        )
    elif kind == "known_hit":
        transaction = ExactProposalTransaction(
            session, logical_session_id=logical, source_hashes=source_hashes
        )
        proposal, proposal_timing = timed_transaction(
            lambda: transaction.propose(stop_x0, stop_tensors)
        )
        proposal_ms = proposal_timing["host_sync_ms"]
        result, action_timing = timed_transaction(
            lambda: transaction.accept_exact(proposal, stop_x0, stop_tensors)
        )
    elif kind == "miss":
        transaction = ExactProposalTransaction(
            session, logical_session_id=logical, source_hashes=source_hashes
        )
        proposal, proposal_timing = timed_transaction(
            lambda: transaction.propose(unchanged_x0, unchanged_tensors)
        )
        proposal_ms = proposal_timing["host_sync_ms"]
        result, action_timing = timed_transaction(
            lambda: transaction.miss(proposal, stop_x0, stop_tensors)
        )
    else:
        raise ValueError(kind)

    decoder = IncrementalDecoder(session.model.tokenizer)
    decoder.start()
    _, decode_timing = decode_incremental(
        decoder,
        result.latent,
        global_start=0,
        n_views=session.n_views,
    )
    action_to_commit = float(action_timing["host_sync_ms"])
    action_to_host = action_to_commit + float(decode_timing["host_ready_ms"])
    row = {
        "repetition": repetition,
        "path": kind,
        "action_to_final_commit_ms": action_to_commit,
        "action_to_host_ready_ms": action_to_host,
        "decode_host_ready_ms": decode_timing["host_ready_ms"],
        "dropped_frames": decode_timing["dropped_frames"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if proposal_ms is not None:
        row.update(
            {
                "proposal_pre_action_ms": proposal_ms,
                "proposal_ready_before_750ms": proposal_ms <= 750.0,
                "proposal_ready_margin_ms": 750.0 - proposal_ms,
                "abandoned_proposal": kind == "miss",
                "total_charged_ms": proposal_ms + action_to_host,
            }
        )
    else:
        row["total_charged_ms"] = action_to_host
    return row


def exact_sequence(
    *,
    kind: str,
    session: Any,
    stop_x0: Any,
    stop_tensors: Mapping[str, torch.Tensor],
    unchanged_x0: Any,
    unchanged_tensors: Mapping[str, torch.Tensor],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    first_logical = "pre-roll-b1:exact:block:0"
    if kind == "on_demand_full":
        first = ExactProposalTransaction(
            session, logical_session_id=first_logical, source_hashes=source_hashes
        ).run_full(stop_x0, stop_tensors)
    elif kind == "known_hit":
        transaction = ExactProposalTransaction(
            session, logical_session_id=first_logical, source_hashes=source_hashes
        )
        proposal = transaction.propose(stop_x0, stop_tensors)
        first = transaction.accept_exact(proposal, stop_x0, stop_tensors)
    elif kind == "miss":
        transaction = ExactProposalTransaction(
            session, logical_session_id=first_logical, source_hashes=source_hashes
        )
        proposal = transaction.propose(unchanged_x0, unchanged_tensors)
        first = transaction.miss(proposal, stop_x0, stop_tensors)
    else:
        raise ValueError(kind)

    first_latent = first.latent.clone()
    first_output = session.output.clone()
    first_live = experiment.digest_live_state(session)
    decoder = IncrementalDecoder(session.model.tokenizer)
    decoder.start()
    first_raw, first_decode = decode_incremental(
        decoder, first.latent, global_start=0, n_views=session.n_views
    )

    continuation = ExactProposalTransaction(
        session,
        logical_session_id="pre-roll-b1:exact:block:1",
        source_hashes=source_hashes,
    ).run_full(stop_x0, stop_tensors)
    second_raw, second_decode = decode_incremental(
        decoder,
        continuation.latent,
        global_start=session.nfpb,
        n_views=session.n_views,
    )
    incremental_raw = np.concatenate([first_raw, second_raw], axis=1)
    output_end = session.block_index * session.nfpb
    output_views = rearrange(
        session.output,
        "b (v t) c h w -> b v t c h w",
        v=session.n_views,
    )[:, :, :output_end]
    output_prefix = rearrange(
        output_views, "b v t c h w -> b (v t) c h w"
    ).contiguous()
    full_raw = decode_full(
        session.model.tokenizer, output_prefix, n_views=session.n_views
    )
    return {
        "first": {
            "latent": first_latent,
            "output": first_output,
            "live": first_live,
            "raw": first_raw,
        },
        "continuation": {
            "latent": continuation.latent.clone(),
            "output": session.output.clone(),
            "live": experiment.digest_live_state(session),
            "incremental_raw": incremental_raw,
            "full_raw": full_raw,
        },
        "decode": {
            "first": first_decode,
            "continuation": second_decode,
            "dropped_frames": first_decode["dropped_frames"]
            + second_decode["dropped_frames"],
        },
    }


def compare_sequence(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "first_latent": experiment.compare_tensor(
            reference["first"]["latent"], candidate["first"]["latent"]
        ),
        "first_output": experiment.compare_tensor(
            reference["first"]["output"], candidate["first"]["output"]
        ),
        "first_live_kv": experiment.compare_live_digest(
            reference["first"]["live"], candidate["first"]["live"]
        ),
        "first_raw_uint8": compare_array(
            reference["first"]["raw"], candidate["first"]["raw"]
        ),
        "continuation_latent": experiment.compare_tensor(
            reference["continuation"]["latent"],
            candidate["continuation"]["latent"],
        ),
        "continuation_output": experiment.compare_tensor(
            reference["continuation"]["output"],
            candidate["continuation"]["output"],
        ),
        "continuation_live_kv": experiment.compare_live_digest(
            reference["continuation"]["live"],
            candidate["continuation"]["live"],
        ),
        "continuation_raw_uint8": compare_array(
            reference["continuation"]["incremental_raw"],
            candidate["continuation"]["incremental_raw"],
        ),
        "candidate_incremental_vs_full_raw_uint8": compare_array(
            candidate["continuation"]["full_raw"],
            candidate["continuation"]["incremental_raw"],
        ),
    }
    checks = list(result.values())
    result["exact"] = all(row["exact"] for row in checks)
    result["max_error"] = max(row["max_error"] for row in checks)
    result["dropped_frames"] = candidate["decode"]["dropped_frames"]
    return result


def benchmark(output: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    base_config = json.loads(experiment.CONFIG.read_text())
    identity = experiment.source_identity()
    identity.update(
        {
            "pre_roll_config_sha256": experiment.sha256_file(CONFIG),
            "benchmark_source_sha256": experiment.sha256_file(Path(__file__)),
        }
    )
    identity["identity_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(identity)
    ).hexdigest()
    engine = experiment.build_engine(base_config)
    images = experiment.load_images()
    stop_keyboard, stop_camera = experiment.action_arrays("canonical_stop", 0)
    stop_batch, stop_stock_x0, stop_tensors, _ = experiment.make_x0(
        engine, images, stop_keyboard, stop_camera
    )
    stop_x0, _ = make_shallow_conditioning_x0(stop_stock_x0)
    unchanged_keyboard, unchanged_camera = experiment.action_arrays("unchanged", 0)
    _, unchanged_stock_x0, unchanged_tensors, _ = experiment.make_x0(
        engine, images, unchanged_keyboard, unchanged_camera
    )
    unchanged_x0, _ = make_shallow_conditioning_x0(unchanged_stock_x0)
    session = experiment.new_session(
        engine, stop_batch, int(base_config["model"]["seed"])
    )
    parent = session.fork_delta()
    source_hashes = identity["sources"]
    paths = ("on_demand_full", "known_hit", "miss")
    warmups = int(config["benchmark"]["warmups"])
    repetitions = int(config["benchmark"]["measured_repetitions"])
    started = time.perf_counter()

    for warmup in range(warmups):
        for kind in paths:
            run_path(
                kind=kind,
                repetition=-(warmup + 1),
                session=session,
                parent=parent,
                stop_x0=stop_x0,
                stop_tensors=stop_tensors,
                unchanged_x0=unchanged_x0,
                unchanged_tensors=unchanged_tensors,
                source_hashes=source_hashes,
            )
    rows = {kind: [] for kind in paths}
    rng = random.Random(config["benchmark"]["randomized_path_order_seed"])
    for repetition in range(repetitions):
        order = list(paths)
        rng.shuffle(order)
        for kind in order:
            rows[kind].append(
                run_path(
                    kind=kind,
                    repetition=repetition,
                    session=session,
                    parent=parent,
                    stop_x0=stop_x0,
                    stop_tensors=stop_tensors,
                    unchanged_x0=unchanged_x0,
                    unchanged_tensors=unchanged_tensors,
                    source_hashes=source_hashes,
                )
            )
        experiment.atomic_json(
            output,
            {
                "schema_version": "rtwm-v2-pre-roll-b1-benchmark-1",
                "status": "running",
                "identity": identity,
                "identity_sha256": identity["identity_sha256"],
                "samples": rows,
            },
        )

    exact_session = experiment.new_session(
        engine, stop_batch, int(base_config["model"]["seed"])
    )
    reference = exact_sequence(
        kind="on_demand_full",
        session=exact_session,
        stop_x0=stop_x0,
        stop_tensors=stop_tensors,
        unchanged_x0=unchanged_x0,
        unchanged_tensors=unchanged_tensors,
        source_hashes=source_hashes,
    )
    exact_session = experiment.new_session(
        engine, stop_batch, int(base_config["model"]["seed"])
    )
    hit = exact_sequence(
        kind="known_hit",
        session=exact_session,
        stop_x0=stop_x0,
        stop_tensors=stop_tensors,
        unchanged_x0=unchanged_x0,
        unchanged_tensors=unchanged_tensors,
        source_hashes=source_hashes,
    )
    exact_session = experiment.new_session(
        engine, stop_batch, int(base_config["model"]["seed"])
    )
    miss = exact_sequence(
        kind="miss",
        session=exact_session,
        stop_x0=stop_x0,
        stop_tensors=stop_tensors,
        unchanged_x0=unchanged_x0,
        unchanged_tensors=unchanged_tensors,
        source_hashes=source_hashes,
    )
    hit_exact = compare_sequence(reference, hit)
    miss_exact = compare_sequence(reference, miss)
    baseline_by_rep = {
        row["repetition"]: row for row in rows["on_demand_full"]
    }
    hit_by_rep = {row["repetition"]: row for row in rows["known_hit"]}
    commit_savings = [
        baseline_by_rep[index]["action_to_final_commit_ms"]
        - hit_by_rep[index]["action_to_final_commit_ms"]
        for index in sorted(hit_by_rep)
    ]
    host_savings = [
        baseline_by_rep[index]["action_to_host_ready_ms"]
        - hit_by_rep[index]["action_to_host_ready_ms"]
        for index in sorted(hit_by_rep)
    ]
    summaries = {name: summarize_path(values) for name, values in rows.items()}
    proposal_p95 = summaries["known_hit"]["proposal_pre_action_ms"]["p95_ms"]
    zero_drops = all(
        summary["total_dropped_frames"] == 0 for summary in summaries.values()
    ) and hit_exact["dropped_frames"] == miss_exact["dropped_frames"] == 0
    exact = bool(hit_exact["exact"] and miss_exact["exact"])
    gate_passed = bool(proposal_p95 <= 750.0 and exact and zero_drops)
    report = {
        "schema_version": "rtwm-v2-pre-roll-b1-benchmark-1",
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "scope": {
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": 1,
            "block_index": 0,
            "cache_phase": "pre_roll",
            "synthetic_action_event_ms": 750.0,
            "rolling_gate_modified": False,
            "b2_tested": False,
        },
        "protocol": {
            "warmups_per_path": warmups,
            "randomized_measured_repetitions_per_path": repetitions,
            "known_hit": "canonical_stop proposal and canonical_stop observation",
            "miss": "unchanged proposal, canonical_stop observation",
            "proposal_work_occurs_before_action_event": True,
            "action_timing_excludes_pre_action_proposal": True,
            "incremental_decode": (
                "cached Wan decoder with global normalization indices and "
                "synchronous uint8 host copy"
            ),
        },
        "samples": rows,
        "summaries": summaries,
        "known_hit_savings": {
            "action_to_final_commit_ms": experiment.percentiles(commit_savings),
            "action_to_host_ready_ms": experiment.percentiles(host_savings),
        },
        "miss_accounting": {
            "proposal_is_abandoned_and_charged": True,
            "fallback_is_full_observed_action_and_charged": True,
            "pre_action_proposal_not_subtracted_from_action_latency": True,
        },
        "exactness": {
            "known_hit": hit_exact,
            "miss": miss_exact,
            "required_max_error": 0,
            "all_exact": exact,
            "zero_drops": zero_drops,
            "continuation_blocks_checked": 1,
        },
        "gates": {
            "pre_roll_b1_gate": {
                "proposal_p95_ms": proposal_p95,
                "limit_ms": 750.0,
                "proposal_readiness_passed": proposal_p95 <= 750.0,
                "exactness_passed": exact,
                "zero_drops_passed": zero_drops,
                "passed": gate_passed,
            },
            "global_rolling_gate": {
                "passed": False,
                "unchanged": True,
                "source": "results/summary.json",
            },
        },
        "cost_model": {
            "proposal_pre_action_mean_ms": summaries["known_hit"][
                "proposal_pre_action_ms"
            ]["mean_ms"],
            "baseline_action_to_host_mean_ms": summaries["on_demand_full"][
                "action_to_host_ready_ms"
            ]["mean_ms"],
            "hit_action_to_host_mean_ms": summaries["known_hit"][
                "action_to_host_ready_ms"
            ]["mean_ms"],
            "miss_action_to_host_mean_ms": summaries["miss"][
                "action_to_host_ready_ms"
            ]["mean_ms"],
        },
        "gpu_wall_runtime_seconds": time.perf_counter() - started,
        "claims": {
            "exact_b1_pre_roll_proposal_readiness": gate_passed,
            "rolling_cache_readiness": False,
            "end_to_end_benefit_unconditional": False,
        },
    }
    experiment.atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = benchmark(args.output)
    gate = report["gates"]["pre_roll_b1_gate"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": gate["passed"],
                "proposal_p95_ms": gate["proposal_p95_ms"],
                "gpu_wall_runtime_seconds": report["gpu_wall_runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
