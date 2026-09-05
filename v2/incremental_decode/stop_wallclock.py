"""Fresh canonical STOP wall-clock chain with cached incremental decode."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
RTWM = V2.parent
SAFESWM = V2.parents[1] / "safeswm"
GAMMA = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
DEFAULT_CONFIG = HERE / "stop_wallclock_config.json"
DEFAULT_OUTPUT = V2 / "results" / "incremental_decode" / "stop_wallclock_chain_v1"
PROMPT = "Two Minecraft players exploring the world"
SCHEMA = "rtwm-v2-incremental-stop-wallclock-1"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RTWM))

from core import (  # noqa: E402
    ProbeError,
    array_sha256,
    artifact_metadata,
    atomic_write_bytes,
    atomic_write_json,
    canonical_sha256,
    compare_exact,
    decoded_to_uint8_views,
    file_sha256,
    percentile_summary,
    seal_manifest,
    strict_resume,
)
from gpu_probe import CachedWanDecoder, decode_full, make_sink  # noqa: E402


def relative_to_v2(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), V2.resolve())).as_posix()


def raw_array_sha256(array: Any) -> str:
    import hashlib
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_config_path(config_path: Path, relative: str) -> Path:
    return (config_path.parent / relative).resolve()


def find_record(
    manifest: dict[str, Any], rollout_id: str
) -> dict[str, Any]:
    record = next(
        (
            row
            for row in manifest.get("rollouts", [])
            if row.get("rollout_id") == rollout_id
        ),
        None,
    )
    if record is None or record.get("status") != "complete":
        raise ProbeError(f"missing complete canonical record {rollout_id}")
    return record


def verified_artifact(
    root: Path, record: dict[str, Any], name: str
) -> Path:
    metadata = record.get("artifacts", {}).get(name)
    if not isinstance(metadata, dict):
        raise ProbeError(f"{record.get('rollout_id')} lacks {name}")
    relative = Path(str(metadata.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProbeError(f"canonical {name} path is not safely relative")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != int(metadata["bytes"])
        or file_sha256(path) != metadata["sha256"]
    ):
        raise ProbeError(f"canonical {name} failed size/hash verification")
    return path


def load_canonical_inputs(
    config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    canonical = config["canonical_inputs"]
    stop_protocol_path = resolve_config_path(
        config_path, canonical["stop_protocol"]
    )
    stop_manifest_path = resolve_config_path(
        config_path, canonical["stop_manifest"]
    )
    control_manifest_path = resolve_config_path(
        config_path, canonical["control_manifest"]
    )
    expected_files = (
        (stop_protocol_path, canonical["stop_protocol_sha256"]),
        (stop_manifest_path, canonical["stop_manifest_sha256"]),
        (control_manifest_path, canonical["control_manifest_sha256"]),
    )
    for path, expected in expected_files:
        if not path.is_file() or file_sha256(path) != expected:
            raise ProbeError(f"canonical input hash mismatch: {path}")

    protocol = load_json(stop_protocol_path)
    stop_manifest = load_json(stop_manifest_path)
    control_manifest = load_json(control_manifest_path)
    if (
        protocol.get("schema_version") != "rtwm-v2-canonical-stop-1"
        or protocol.get("immutable") is not True
        or stop_manifest.get("schema_version")
        != canonical["stop_manifest_schema"]
        or control_manifest.get("schema_version")
        != canonical["control_manifest_schema"]
    ):
        raise ProbeError("canonical STOP/control schema or immutability mismatch")
    action_hash = canonical["action_protocol_sha256"]
    if (
        protocol["action_protocol"]["canonical_action_protocol_sha256"]
        != action_hash
        or stop_manifest["action_protocol_sha256"] != action_hash
        or control_manifest["action_protocol_sha256"] != action_hash
    ):
        raise ProbeError("canonical action protocol identity mismatch")
    if (
        protocol["action_protocol"]["solaris_source_sha256"]
        != canonical["solaris_source_sha256"]
    ):
        raise ProbeError("Solaris action source hash mismatch")
    for manifest in (stop_manifest, control_manifest):
        if (
            manifest["model_artifacts"]["checkpoint_sha256"]
            != canonical["model_checkpoint_sha256"]
            or manifest["model_artifacts"]["tokenizer_sha256"]
            != canonical["tokenizer_sha256"]
        ):
            raise ProbeError("canonical model artifact identity mismatch")
    if stop_manifest["source"] != control_manifest["source"]:
        raise ProbeError("STOP/control Gamma source identities differ")

    stop_record = find_record(
        stop_manifest, canonical["stop_action_rollout_id"]
    )
    control_record = find_record(
        control_manifest, canonical["control_action_rollout_id"]
    )
    stop_root = stop_manifest_path.parent
    control_root = control_manifest_path.parent
    stop_action_path = verified_artifact(
        stop_root, stop_record, "action_tensors"
    )
    control_action_path = verified_artifact(
        control_root, control_record, "action_tensors"
    )
    with np.load(stop_action_path, allow_pickle=False) as payload:
        generation_keyboard = np.asarray(payload["generation_keyboard"]).copy()
        generation_camera = np.asarray(payload["generation_camera"]).copy()
        post_keyboard = np.asarray(payload["post_keyboard"]).copy()
        post_camera = np.asarray(payload["post_camera"]).copy()
    with np.load(control_action_path, allow_pickle=False) as payload:
        control_keyboard = np.asarray(payload["generation_keyboard"])
        control_camera = np.asarray(payload["generation_camera"])

    if not (
        np.array_equal(generation_keyboard, control_keyboard)
        and np.array_equal(generation_camera, control_camera)
    ):
        raise ProbeError("canonical STOP/control forward tensors differ")
    expected_hashes = stop_record["action_tensor_hashes"]
    actual_hashes = {
        "generation_keyboard_sha256": raw_array_sha256(generation_keyboard),
        "generation_camera_sha256": raw_array_sha256(generation_camera),
        "post_keyboard_sha256": raw_array_sha256(post_keyboard),
        "post_camera_sha256": raw_array_sha256(post_camera),
    }
    if actual_hashes != expected_hashes:
        raise ProbeError("canonical STOP action tensor hashes differ")
    change_frame = int(protocol["model"]["change_frame"])
    if (
        generation_keyboard.shape != (1, int(protocol["model"]["n_frames"]), 23)
        or generation_camera.shape[-1] != 2
        or not np.all(generation_keyboard[:, :, 11] == 1)
        or np.count_nonzero(generation_keyboard[:, :, :11])
        or np.count_nonzero(generation_keyboard[:, :, 12:])
        or np.count_nonzero(post_keyboard[:, change_frame:])
        or np.count_nonzero(generation_camera)
        or np.count_nonzero(post_camera)
    ):
        raise ProbeError("action artifact is not canonical forward-to-zero STOP")

    return {
        "protocol": protocol,
        "stop_manifest": stop_manifest,
        "control_manifest": control_manifest,
        "generation_keyboard": generation_keyboard,
        "generation_camera": generation_camera,
        "post_keyboard": post_keyboard,
        "post_camera": post_camera,
        "action_tensor_hashes": actual_hashes,
        "provenance": {
            "stop_protocol": {
                "path": relative_to_v2(stop_protocol_path),
                "sha256": file_sha256(stop_protocol_path),
            },
            "stop_manifest": {
                "path": relative_to_v2(stop_manifest_path),
                "sha256": file_sha256(stop_manifest_path),
                "rollout_id": stop_record["rollout_id"],
            },
            "stop_action_tensors": {
                "path": relative_to_v2(stop_action_path),
                "sha256": file_sha256(stop_action_path),
            },
            "control_manifest": {
                "path": relative_to_v2(control_manifest_path),
                "sha256": file_sha256(control_manifest_path),
                "rollout_id": control_record["rollout_id"],
            },
            "control_action_tensors": {
                "path": relative_to_v2(control_action_path),
                "sha256": file_sha256(control_action_path),
            },
            "action_tensor_hashes": actual_hashes,
            "gamma_source": stop_manifest["source"],
            "model_artifacts": stop_manifest["model_artifacts"],
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "rtwm-v2-incremental-stop-wallclock-config-1"
        or config.get("immutable") is not True
    ):
        raise ProbeError("unsupported or mutable wall-clock config")
    if (
        config["scene"] != "buildTower_normal"
        or config["synthetic_action_receipt_latent"] != 24
        or config["latent_frames_per_block"] != 3
        or config["n_views"] != 2
        or config["seeds"] != [301, 302]
    ):
        raise ProbeError("frozen wall-clock design changed")
    expected = [
        (301, "d0", 27),
        (301, "d1", 30),
        (302, "d1", 30),
        (302, "d0", 27),
    ]
    actual = [
        (row["seed"], row["delay"], row["admission_latent"])
        for row in config["counterbalanced_run_order"]
    ]
    if actual != expected:
        raise ProbeError("run order is not the frozen counterbalance")


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "numpy", "einops", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def environment(torch: Any) -> dict[str, Any]:
    device = torch.cuda.current_device()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, allow_nan=False, sort_keys=True).encode() + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, payload)


def record_event(
    events: list[dict[str, Any]],
    event: str,
    timestamp_ns: int,
    origin_ns: int,
    **fields: Any,
) -> None:
    events.append(
        {
            "event": event,
            "monotonic_ns": timestamp_ns,
            "relative_ms": (timestamp_ns - origin_ns) / 1e6,
            **fields,
        }
    )


def run_identity(
    common_identity: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    return {
        **common_identity,
        "run": {
            "run_index": int(spec["run_index"]),
            "seed": int(spec["seed"]),
            "delay": str(spec["delay"]),
            "admission_latent": int(spec["admission_latent"]),
        },
    }


def run_one(
    *,
    spec: dict[str, Any],
    config: dict[str, Any],
    common_identity: dict[str, Any],
    output: Path,
    engine: Any,
    batch_forward: dict[str, Any],
    x0_forward: Callable[..., Any],
    x0_stop: Callable[..., Any],
    torch: Any,
) -> dict[str, Any]:
    from einops import rearrange
    from driver_v2 import BlockwiseSession

    seed = int(spec["seed"])
    delay = str(spec["delay"])
    admission = int(spec["admission_latent"])
    receipt_latent = int(config["synthetic_action_receipt_latent"])
    run_id = f"{config['scene']}__seed{seed}__{delay}"
    run_dir = output / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    identity = run_identity(common_identity, spec)
    resumed = strict_resume(manifest_path, expected_identity=identity)
    if resumed is not None:
        print(f"[stop-chain] resume verified {run_id}", flush=True)
        return resumed
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ProbeError(f"nonempty incomplete run directory: {run_dir}")

    engine.clear_cache()
    session = BlockwiseSession(engine, batch_forward, seed=seed)
    if (
        session.n_views != int(config["n_views"])
        or session.nfpb != int(config["latent_frames_per_block"])
    ):
        raise ProbeError("runtime Gamma view/block layout changed")
    decoder = CachedWanDecoder(engine.model.tokenizer, torch)
    decoder.start()
    sink = make_sink(config["sink"])

    origin_ns = time.perf_counter_ns()
    receipt_ns = admission_ns = None
    admission_block: dict[str, Any] | None = None
    chunks = []
    events: list[dict[str, Any]] = []
    queued_total = dropped_total = consumed_total = 0
    incremental_gpu_ms = 0.0

    while session.block_index * session.nfpb <= admission:
        latent_start = session.block_index * session.nfpb
        latent_end = latent_start + session.nfpb
        if latent_start == receipt_latent:
            receipt_ns = time.perf_counter_ns()
            record_event(
                events,
                "synthetic_action_received",
                receipt_ns,
                origin_ns,
                receipt_latent=receipt_latent,
                action_semantics="canonical forward-to-zero STOP request",
            )
        use_stop = latent_start >= admission
        if use_stop and admission_ns is None:
            if receipt_ns is None:
                raise ProbeError("STOP admitted before synthetic receipt")
            admission_ns = time.perf_counter_ns()
            record_event(
                events,
                "action_admitted",
                admission_ns,
                origin_ns,
                admission_latent=admission,
                delay=delay,
            )

        result = session.step_block(x0_stop if use_stop else x0_forward)
        block_event: dict[str, Any] = {
            "event": "block_chain",
            "block_index": int(result.block_index),
            "latent_start": latent_start,
            "latent_end_exclusive": latent_end,
            "condition": "stop" if use_stop else "forward",
            "block_started_ns": int(result.block_started_ns),
            "denoise_complete_ns": int(result.denoise_complete_ns),
            "context_cache_commit_ns": int(result.latent_committed_ns),
            "denoise_ms": float(result.denoise_ms),
            "context_cache_commit_ms": float(result.commit_ms),
            "steps_used": int(result.steps_used),
        }

        flat_block = rearrange(
            result.latent,
            "b (v t) c h w -> (b v) c t h w",
            v=session.n_views,
        ).contiguous()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        decode_started_ns = time.perf_counter_ns()
        start_event.record()
        decoded = decoder.decode(flat_block, global_start=latent_start)
        decode_complete_ns = time.perf_counter_ns()
        raw_device = decoded_to_uint8_views(
            decoded, batch=1, n_views=session.n_views
        )
        end_event.record()
        torch.cuda.synchronize()
        device_ready_ns = time.perf_counter_ns()
        gpu_ms = float(start_event.elapsed_time(end_event))
        raw_host = raw_device.cpu().numpy()
        host_ready_ns = time.perf_counter_ns()
        incremental_gpu_ms += gpu_ms
        chunks.append(raw_host)

        pixel_start = 0 if latent_start == 0 else latent_start * 4 - 3
        pixel_end = pixel_start + int(raw_host.shape[1])
        queued = dropped = consumed = 0
        sink_started_ns = sink_complete_ns = None
        sink_result = None
        if sink is not None:
            queued = int(raw_host.shape[1])
            sink_started_ns = time.perf_counter_ns()
            sink_result = sink(raw_host)
            sink_complete_ns = time.perf_counter_ns()
            consumed = int(sink_result["frames_consumed"])
        queued_total += queued
        dropped_total += dropped
        consumed_total += consumed
        block_event.update(
            {
                "decode_started_ns": decode_started_ns,
                "decode_complete_ns": decode_complete_ns,
                "device_ready_ns": device_ready_ns,
                "host_ready_ns": host_ready_ns,
                "decode_return_ms": (decode_complete_ns - decode_started_ns)
                / 1e6,
                "decode_device_ready_ms": (
                    device_ready_ns - decode_started_ns
                )
                / 1e6,
                "decode_host_ready_ms": (host_ready_ns - decode_started_ns)
                / 1e6,
                "gpu_decode_and_quantize_ms": gpu_ms,
                "global_pixel_start": pixel_start,
                "global_pixel_end_exclusive": pixel_end,
                "host_ready_views": [0, 1],
                "host_ready_frame_count_per_view": int(raw_host.shape[1]),
                "queued_frames": queued,
                "dropped_frames": dropped,
                "consumed_frames": consumed,
                "sink_started_ns": sink_started_ns,
                "sink_complete_ns": sink_complete_ns,
                "sink_result": sink_result,
                "sink_is_physical_presentation": False,
            }
        )
        events.append(block_event)
        if use_stop:
            support_frame = admission * 4 - 3
            if pixel_start != support_frame:
                raise ProbeError("admission block does not begin at RF support")
            admission_block = block_event
            record_event(
                events,
                "support_frames_host_ready",
                host_ready_ns,
                origin_ns,
                support_frame=support_frame,
                views=[0, 1],
                semantics="synchronous raw uint8 host readiness",
            )
            if sink_complete_ns is not None:
                record_event(
                    events,
                    "sink_consumed",
                    sink_complete_ns,
                    origin_ns,
                    support_frame=support_frame,
                    physical_presentation=False,
                )
            del decoded, raw_device
            break
        del decoded, raw_device

    if (
        receipt_ns is None
        or admission_ns is None
        or admission_block is None
    ):
        raise ProbeError("live STOP event chain is incomplete")

    import numpy as np

    incremental_raw = np.concatenate(chunks, axis=1)
    prefix_end = admission + session.nfpb
    output_views = rearrange(
        session.output,
        "b (v t) c h w -> (b v) c t h w",
        v=session.n_views,
    )[:, :, :prefix_end].contiguous()
    full_start_event = torch.cuda.Event(enable_timing=True)
    full_end_event = torch.cuda.Event(enable_timing=True)
    full_started_ns = time.perf_counter_ns()
    full_start_event.record()
    full_decoded = decode_full(engine.model.tokenizer, output_views)
    full_raw_device = decoded_to_uint8_views(
        full_decoded, batch=1, n_views=session.n_views
    )
    full_end_event.record()
    torch.cuda.synchronize()
    full_device_ready_ns = time.perf_counter_ns()
    full_gpu_ms = float(full_start_event.elapsed_time(full_end_event))
    full_raw = full_raw_device.cpu().numpy()
    full_host_ready_ns = time.perf_counter_ns()
    comparison = compare_exact(incremental_raw, full_raw)
    del full_decoded, full_raw_device

    support_host_ns = int(admission_block["host_ready_ns"])
    sink_complete = admission_block["sink_complete_ns"]
    metrics = {
        "action_receipt_to_admission_ms": (admission_ns - receipt_ns) / 1e6,
        "action_receipt_to_denoise_complete_ms": (
            int(admission_block["denoise_complete_ns"]) - receipt_ns
        )
        / 1e6,
        "action_receipt_to_context_commit_ms": (
            int(admission_block["context_cache_commit_ns"]) - receipt_ns
        )
        / 1e6,
        "action_receipt_to_device_ready_ms": (
            int(admission_block["device_ready_ns"]) - receipt_ns
        )
        / 1e6,
        "action_receipt_to_host_ready_ms": (
            support_host_ns - receipt_ns
        )
        / 1e6,
        "action_receipt_to_sink_consumed_ms": (
            (int(sink_complete) - receipt_ns) / 1e6
            if sink_complete is not None
            else None
        ),
        "admission_to_denoise_complete_ms": (
            int(admission_block["denoise_complete_ns"]) - admission_ns
        )
        / 1e6,
        "denoise_complete_to_context_commit_ms": (
            int(admission_block["context_cache_commit_ns"])
            - int(admission_block["denoise_complete_ns"])
        )
        / 1e6,
        "context_commit_to_device_ready_ms": (
            int(admission_block["device_ready_ns"])
            - int(admission_block["context_cache_commit_ns"])
        )
        / 1e6,
        "device_ready_to_host_ready_ms": (
            support_host_ns - int(admission_block["device_ready_ns"])
        )
        / 1e6,
    }
    timing_claimable = bool(comparison["exact"])
    events_path = run_dir / "events.jsonl"
    equivalence_path = run_dir / "equivalence.json"
    write_jsonl(events_path, events)
    atomic_write_json(
        equivalence_path,
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "incremental_vs_matching_full_prefix": comparison,
            "timing_claimable": timing_claimable,
        },
    )
    manifest_body = {
        "schema_version": SCHEMA,
        "status": "complete",
        "immutable": True,
        "identity": identity,
        "run_id": run_id,
        "scene": config["scene"],
        "seed": seed,
        "delay": delay,
        "run_index": int(spec["run_index"]),
        "admission_latent": admission,
        "synthetic_action_receipt_latent": receipt_latent,
        "support_frame": admission * 4 - 3,
        "equivalence": comparison,
        "timing_claimable": timing_claimable,
        "withheld_reason": (
            None
            if timing_claimable
            else "nonzero raw uint8 incremental/full-prefix error"
        ),
        "metrics": metrics,
        "gpu": {
            "incremental_decode_total_ms": incremental_gpu_ms,
            "matching_full_prefix_decode_ms": full_gpu_ms,
            "matching_full_prefix_wall_device_ready_ms": (
                full_device_ready_ns - full_started_ns
            )
            / 1e6,
            "matching_full_prefix_wall_host_ready_ms": (
                full_host_ready_ns - full_started_ns
            )
            / 1e6,
        },
        "queue": {
            "queued_frames": queued_total,
            "dropped_frames": dropped_total,
            "consumed_frames": consumed_total,
        },
        "sink": {
            "kind": config["sink"],
            "physical_presentation_measured": False,
            "semantics": "synchronous host callback consumption only",
        },
        "blocks_generated": int(admission_block["block_index"]) + 1,
        "artifacts": {
            "events": artifact_metadata(events_path, root=run_dir),
            "equivalence": artifact_metadata(
                equivalence_path, root=run_dir
            ),
        },
    }
    sealed = seal_manifest(manifest_body)
    atomic_write_json(manifest_path, sealed)
    print(
        f"[stop-chain] {run_id} exact={comparison['exact']} "
        f"host={metrics['action_receipt_to_host_ready_ms']:.1f}ms",
        flush=True,
    )
    if not timing_claimable:
        raise ProbeError(f"{run_id} failed exactness; timing withheld")
    return sealed


METRICS = (
    "action_receipt_to_admission_ms",
    "action_receipt_to_denoise_complete_ms",
    "action_receipt_to_context_commit_ms",
    "action_receipt_to_device_ready_ms",
    "action_receipt_to_host_ready_ms",
    "action_receipt_to_sink_consumed_ms",
)


def summarize(
    *,
    output: Path,
    config: dict[str, Any],
    common_identity: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    if not all(run["timing_claimable"] for run in runs):
        raise ProbeError("refusing timing aggregates: an exactness gate failed")
    aggregates: dict[str, Any] = {}
    for delay in ("d0", "d1"):
        selected = [run for run in runs if run["delay"] == delay]
        if len(selected) != 2:
            raise ProbeError(f"{delay} does not have exactly two seeds")
        aggregates[delay] = {}
        for metric in METRICS:
            values = np.asarray(
                [float(run["metrics"][metric]) for run in selected],
                dtype=np.float64,
            )
            aggregates[delay][metric] = {
                "n": 2,
                "mean_ms": float(values.mean()),
                "sample_sd_ms": float(values.std(ddof=1)),
                "values_ms": [float(value) for value in values],
                "uncertainty_label": config["uncertainty"]["label"],
            }

    run_rows = []
    for run in sorted(runs, key=lambda row: row["run_index"]):
        row = {
            "row_type": "run",
            "run_id": run["run_id"],
            "run_index": run["run_index"],
            "seed": run["seed"],
            "delay": run["delay"],
            "admission_latent": run["admission_latent"],
            "support_frame": run["support_frame"],
            "exact": run["equivalence"]["exact"],
            "max_error": run["equivalence"]["max_error"],
            "incremental_decode_gpu_total_ms": run["gpu"][
                "incremental_decode_total_ms"
            ],
            **run["metrics"],
        }
        run_rows.append(row)
    csv_path = output / "summary.csv"
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(run_rows[0]))
    writer.writeheader()
    writer.writerows(run_rows)
    atomic_write_bytes(csv_path, csv_buffer.getvalue().encode())

    summary_path = output / "summary.json"
    summary_payload = {
        "schema_version": SCHEMA,
        "timing_claimable": True,
        "run_count": 4,
        "exact_run_count": 4,
        "uncertainty": config["uncertainty"],
        "aggregates": aggregates,
        "runs": run_rows,
        "claim_scope": config["claim_scope"],
    }
    atomic_write_json(summary_path, summary_payload)

    import matplotlib.pyplot as plt

    component_keys = (
        "action_receipt_to_admission_ms",
        "admission_to_denoise_complete_ms",
        "denoise_complete_to_context_commit_ms",
        "context_commit_to_device_ready_ms",
        "device_ready_to_host_ready_ms",
    )
    component_labels = (
        "Wait for admission",
        "Admission → denoise",
        "Context-cache commit",
        "Cached decode → device",
        "Device → host",
    )
    colors = ("#9ecae1", "#3182bd", "#756bb1", "#31a354", "#fd8d3c")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(2)
    bottoms = np.zeros(2)
    for key, label, color in zip(component_keys, component_labels, colors):
        means = np.asarray(
            [
                np.mean(
                    [
                        run["metrics"][key]
                        for run in runs
                        if run["delay"] == delay
                    ]
                )
                for delay in ("d0", "d1")
            ]
        )
        axis.bar(x, means, bottom=bottoms, label=label, color=color, width=0.62)
        bottoms += means
    for delay_index, delay in enumerate(("d0", "d1")):
        totals = [
            run["metrics"]["action_receipt_to_host_ready_ms"]
            for run in runs
            if run["delay"] == delay
        ]
        axis.scatter(
            [delay_index - 0.08, delay_index + 0.08],
            totals,
            color="black",
            marker="o",
            s=24,
            zorder=4,
            label="Seed totals" if delay_index == 0 else None,
        )
    axis.set_xticks(x, ["d0 · latent 27", "d1 · latent 30"])
    axis.set_ylabel("Wall-clock milliseconds from synthetic receipt")
    axis.set_title("Canonical STOP: host-ready raw support frame")
    axis.text(
        0.99,
        0.02,
        "n=2 seeds/delay; dots are runs; sink ≠ physical presentation",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=180)
    plt.close(figure)
    plot_path = output / "decomposition.png"
    atomic_write_bytes(plot_path, image.getvalue())

    artifacts = {
        "summary_json": artifact_metadata(summary_path, root=output),
        "summary_csv": artifact_metadata(csv_path, root=output),
        "decomposition_plot": artifact_metadata(plot_path, root=output),
    }
    for run in runs:
        run_manifest = (
            output / "runs" / run["run_id"] / "manifest.json"
        )
        artifacts[f"run_{run['run_index']}_manifest"] = artifact_metadata(
            run_manifest, root=output
        )
    master_body = {
        "schema_version": SCHEMA,
        "status": "complete",
        "immutable": True,
        "identity": common_identity,
        "run_order": config["counterbalanced_run_order"],
        "all_exact": True,
        "timing_claimable": True,
        "uncertainty": config["uncertainty"],
        "artifacts": artifacts,
    }
    sealed = seal_manifest(master_body)
    atomic_write_json(output / "manifest.json", sealed)
    return sealed


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    config = load_json(args.config)
    validate_config(config)
    canonical = load_canonical_inputs(args.config, config)
    checkpoint = MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"
    tokenizer = MODELS / "gamma-world" / "tokenizer.pth"
    scene_path = GAMMA / "data" / config["scene"] / "first_frame.png"
    dependencies = {
        "stop_wallclock.py": file_sha256(Path(__file__)),
        "core.py": file_sha256(HERE / "core.py"),
        "gpu_probe.py": file_sha256(HERE / "gpu_probe.py"),
        "driver_v2.py": file_sha256(RTWM / "driver_v2.py"),
        "wan2pt1.py": file_sha256(
            GAMMA
            / "gamma_world"
            / "_src"
            / "predict2"
            / "tokenizers"
            / "wan2pt1.py"
        ),
        "inference_i2v.py": file_sha256(
            GAMMA
            / "gamma_world"
            / "_src"
            / "gamma_world"
            / "inference"
            / "inference_i2v.py"
        ),
    }
    if (
        file_sha256(checkpoint)
        != config["canonical_inputs"]["model_checkpoint_sha256"]
        or file_sha256(tokenizer)
        != config["canonical_inputs"]["tokenizer_sha256"]
    ):
        raise ProbeError("local model/tokenizer hashes are noncanonical")
    common_identity = {
        "schema_version": SCHEMA,
        "config": {
            "path": relative_to_v2(args.config),
            "sha256": file_sha256(args.config),
        },
        "canonical_provenance": canonical["provenance"],
        "scene": {
            "name": config["scene"],
            "path": relative_to_v2(scene_path),
            "sha256": file_sha256(scene_path),
        },
        "model": {
            "checkpoint_sha256": file_sha256(checkpoint),
            "tokenizer_sha256": file_sha256(tokenizer),
        },
        "dependencies": dependencies,
    }
    import torch

    if not torch.cuda.is_available():
        raise ProbeError("CUDA is unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    common_identity["environment"] = environment(torch)

    master_path = args.output / "manifest.json"
    resumed = strict_resume(
        master_path, expected_identity=common_identity
    )
    if resumed is not None:
        print("[stop-chain] complete immutable study resume verified", flush=True)
        return resumed

    sys.path[:0] = [str(GAMMA), str(GAMMA / "scripts"), str(SAFESWM)]
    import numpy as np
    from PIL import Image
    from gamma_world._src.gamma_world.inference.inference_i2v import (
        I2VInference,
        IS_PREPROCESSED_KEY,
        to_with_skip_tensor,
    )
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from inference import format_hydra_value

    protocol = canonical["protocol"]
    model_config = protocol["model"]
    spec = MODEL_SPECS["causal_few_step"]
    overrides = [
        f"{key}={format_hydra_value(value)}"
        for key, value in spec.config_overrides.items()
    ]
    engine = I2VInference(
        experiment_name=spec.experiment,
        ckpt_path=str(checkpoint),
        config_file=spec.config_file,
        guidance=float(model_config["guidance"]),
        shift=None,
        num_sampling_steps=spec.default_num_steps or 35,
        seed=1,
        context_parallel_size=1,
        experiment_opts=overrides,
        vae_pth=str(tokenizer),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = int(model_config["fps"])
    model = engine.model
    image = Image.open(scene_path)
    half = image.width // 2
    images = [
        np.asarray(image.crop((0, 0, half, image.height))).copy(),
        np.asarray(image.crop((half, 0, image.width, image.height))).copy(),
    ]
    generation_keyboard = torch.from_numpy(canonical["generation_keyboard"])
    generation_camera = torch.from_numpy(canonical["generation_camera"])
    post_keyboard = torch.from_numpy(canonical["post_keyboard"])
    post_camera = torch.from_numpy(canonical["post_camera"])

    def make_batch(keys: Any, camera: Any) -> dict[str, Any]:
        return engine.build_inference_batch(
            images,
            PROMPT,
            [(keys, camera), (keys, camera)],
            num_frames=int(model_config["n_frames"]),
            num_conditional_frames=1,
        )

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        if "video" in batch:
            batch["video"] = batch["video"].float()
            if not batch.get(IS_PREPROCESSED_KEY, False):
                batch["video"] = batch["video"] / 127.5 - 1.0
            batch["video"] = torch.clamp(batch["video"], -1, 1)
        batch[IS_PREPROCESSED_KEY] = True
        batch = to_with_skip_tensor(batch, **model.tensor_kwargs)
        engine.inplace_compute_text_embeddings_online(
            batch, use_negative_prompt=True
        )
        batch = model.get_data_batch_with_latent_view_indices(batch)
        model._normalize_video_databatch_inplace(batch)
        return batch

    batch_forward = preprocess(
        make_batch(generation_keyboard, generation_camera)
    )
    batch_stop = preprocess(make_batch(post_keyboard, post_camera))
    x0_kwargs = {
        "n_views": int(config["n_views"]),
        "guidance": float(model_config["guidance"]),
        "is_negative_prompt": True,
    }
    x0_forward = model.get_x0_fn_from_batch(batch_forward, **x0_kwargs)
    x0_stop = model.get_x0_fn_from_batch(batch_stop, **x0_kwargs)

    runs = []
    for run_spec in config["counterbalanced_run_order"]:
        runs.append(
            run_one(
                spec=run_spec,
                config=config,
                common_identity=common_identity,
                output=args.output,
                engine=engine,
                batch_forward=batch_forward,
                x0_forward=x0_forward,
                x0_stop=x0_stop,
                torch=torch,
            )
        )
    return summarize(
        output=args.output,
        config=config,
        common_identity=common_identity,
        runs=runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical STOP live cached-decode wall-clock chain"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except (ProbeError, ValueError) as error:
        raise SystemExit(f"STOP wall-clock chain refused: {error}") from error


if __name__ == "__main__":
    main()
