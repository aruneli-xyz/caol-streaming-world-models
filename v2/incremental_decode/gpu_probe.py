"""Exactness-gated Wan2.1 cached incremental decode benchmark.

This script reads, but never modifies, a canonical directional_d0 rollout.
Run it with Gamma-World's Python environment on one CUDA GPU.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
SAFESWM = V2.parents[1] / "safeswm"
GAMMA = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
DEFAULT_SOURCE = V2 / "results" / "directional_d0"
DEFAULT_OUTPUT = V2 / "results" / "incremental_decode" / "directional_control_2blocks"
DEFAULT_ROLLOUT = "buildTower_normal__seed201__control"
WAN_SOURCE = (
    GAMMA / "gamma_world" / "_src" / "predict2" / "tokenizers" / "wan2pt1.py"
)
N_VIEWS = 2
BLOCK_LATENTS = 3

sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    ProbeError,
    SCHEMA_VERSION,
    array_sha256,
    artifact_metadata,
    atomic_save_npy,
    atomic_write_bytes,
    atomic_write_json,
    compare_exact,
    decoded_to_uint8_views,
    file_sha256,
    global_normalization_slice,
    percentile_summary,
    seal_manifest,
    split_gamma_views,
    strict_resume,
    validate_block_plan,
)


def relative_to_v2(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), V2.resolve())).as_posix()


def load_source(
    source_root: Path, rollout_id: str
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise ProbeError(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "rtwm-v2-directional-rollouts-2":
        raise ProbeError("source is not the Solaris-canonical directional result schema")
    record = next(
        (
            row
            for row in manifest.get("rollouts", [])
            if row.get("rollout_id") == rollout_id
        ),
        None,
    )
    if record is None or record.get("status") != "complete":
        raise ProbeError(f"source rollout is absent or incomplete: {rollout_id}")
    artifacts = record.get("artifacts", {})
    paths = {}
    for name in ("latent", "decoded_u8"):
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict):
            raise ProbeError(f"source rollout lacks {name} metadata")
        relative = Path(str(metadata.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProbeError(f"source {name} path is not safely relative")
        path = source_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["bytes"])
            or file_sha256(path) != metadata["sha256"]
        ):
            raise ProbeError(f"source {name} failed size/hash verification")
        paths[name] = path
    return manifest, record, paths["latent"], paths["decoded_u8"]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "torch", "einops", "safetensors"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def build_identity(
    *,
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    source_record: dict[str, Any],
    latent_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    tokenizer_path = MODELS / "gamma-world" / "tokenizer.pth"
    return {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": args.rollout_id,
        "n_views": N_VIEWS,
        "block_latents": BLOCK_LATENTS,
        "max_blocks": args.max_blocks,
        "sink": args.sink,
        "source": {
            "root": relative_to_v2(args.source_root),
            "manifest": relative_to_v2(args.source_root / "manifest.json"),
            "manifest_sha256": file_sha256(args.source_root / "manifest.json"),
            "schema_version": source_manifest["schema_version"],
            "config_sha256": source_manifest["config_sha256"],
            "protocol_sha256": source_manifest["protocol_sha256"],
            "action_protocol_sha256": source_manifest["action_protocol_sha256"],
            "gamma_source": source_manifest["source"],
            "model_artifacts": source_manifest["model_artifacts"],
            "record_completed_at": source_record["completed_at"],
            "latent": {
                "path": source_record["artifacts"]["latent"]["path"],
                "bytes": latent_path.stat().st_size,
                "sha256": file_sha256(latent_path),
            },
            "decoded_u8": {
                "path": source_record["artifacts"]["decoded_u8"]["path"],
                "bytes": raw_path.stat().st_size,
                "sha256": file_sha256(raw_path),
            },
        },
        "dependencies": {
            "gpu_probe.py": file_sha256(Path(__file__)),
            "core.py": file_sha256(HERE / "core.py"),
            "wan2pt1.py": file_sha256(WAN_SOURCE),
            "tokenizer.pth": file_sha256(tokenizer_path),
        },
    }


def source_raw_views(
    raw_path: Path, *, frames: int, expected_height: int, expected_width: int
) -> Any:
    import numpy as np

    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if (
        raw.ndim != 4
        or raw.dtype != np.uint8
        or raw.shape[0] < frames
        or raw.shape[1] != expected_height
        or raw.shape[2] != expected_width * N_VIEWS
        or raw.shape[3] != 3
    ):
        raise ProbeError(
            "canonical raw artifact is not [T,H,2*W,3] with the expected probe size; "
            f"got {raw.shape} {raw.dtype}"
        )
    return np.stack(
        [
            np.asarray(
                raw[:frames, :, view * expected_width : (view + 1) * expected_width]
            )
            for view in range(N_VIEWS)
        ],
        axis=0,
    )


class CachedWanDecoder:
    """Own one Wan decoder cache for an entire incremental session."""

    def __init__(self, tokenizer: Any, torch_module: Any):
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.started = False

    def start(self) -> None:
        if self.started:
            raise ProbeError("incremental decoder session already started")
        self.tokenizer.clear_cache()
        self.started = True

    def decode(self, normalized_chunk: Any, *, global_start: int) -> Any:
        if not self.started:
            raise ProbeError("incremental decoder session was not started")
        model = self.tokenizer.model
        denormalized = global_normalization_slice(
            normalized_chunk,
            model.video_mean,
            model.video_std,
            global_start=global_start,
        )
        # Calling WanVAE.decode directly is intentional: the public interface
        # restarts normalization at zero and does not expose this cache flag.
        return model.decode(denormalized, clear_decoder_cache=False)


def decode_full(tokenizer: Any, flat_latent: Any) -> Any:
    model = tokenizer.model
    denormalized = global_normalization_slice(
        flat_latent, model.video_mean, model.video_std, global_start=0
    )
    tokenizer.clear_cache()
    return model.decode(denormalized, clear_decoder_cache=True)


def make_sink(kind: str) -> Callable[[Any], dict[str, Any]] | None:
    if kind == "none":
        return None
    if kind != "hash":
        raise ValueError(kind)

    def consume(frames: Any) -> dict[str, Any]:
        return {
            "sha256": array_sha256(frames),
            "frames_consumed": int(frames.shape[1]),
        }

    return consume


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, allow_nan=False, sort_keys=True).encode() + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    source_manifest, source_record, latent_path, raw_path = load_source(
        args.source_root, args.rollout_id
    )
    identity = build_identity(
        args=args,
        source_manifest=source_manifest,
        source_record=source_record,
        latent_path=latent_path,
        raw_path=raw_path,
    )
    manifest_path = args.output / "manifest.json"
    resumed = strict_resume(manifest_path, expected_identity=identity)
    if resumed is not None:
        print(json.dumps({"resumed": True, **resumed["summary"]}, sort_keys=True))
        return resumed
    if args.output.exists() and any(args.output.iterdir()):
        raise ProbeError(
            "output directory is nonempty without a valid immutable manifest; "
            "choose a fresh output directory"
        )

    sys.path.insert(0, str(GAMMA))
    import numpy as np
    import torch
    from gamma_world._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

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

    latent = torch.load(latent_path, map_location="cpu")
    if not isinstance(latent, torch.Tensor):
        raise ProbeError("source latent artifact is not a tensor")
    flat = split_gamma_views(latent, N_VIEWS)
    if flat.shape[0] != N_VIEWS or flat.shape[1] != 16:
        raise ProbeError(f"unexpected Gamma latent shape {tuple(latent.shape)}")
    plan = validate_block_plan(
        int(flat.shape[2]), block_latents=BLOCK_LATENTS, max_blocks=args.max_blocks
    )
    probe_latents = plan[-1][1]
    flat = flat[:, :, :probe_latents].contiguous().to("cuda")

    tokenizer = Wan2pt1VAEInterface(
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        keep_decoder_cache=True,
    )
    torch.cuda.synchronize()

    with torch.inference_mode():
        full_start_ns = time.perf_counter_ns()
        full_start_event = torch.cuda.Event(enable_timing=True)
        full_end_event = torch.cuda.Event(enable_timing=True)
        full_start_event.record()
        full_decoded = decode_full(tokenizer, flat)
        full_decode_complete_ns = time.perf_counter_ns()
        full_raw_device = decoded_to_uint8_views(
            full_decoded, batch=1, n_views=N_VIEWS
        )
        full_end_event.record()
        torch.cuda.synchronize()
        full_device_ready_ns = time.perf_counter_ns()
        full_gpu_ms = float(full_start_event.elapsed_time(full_end_event))
        full_raw = full_raw_device.cpu().numpy()
        full_host_ready_ns = time.perf_counter_ns()
        del full_decoded, full_raw_device

        session = CachedWanDecoder(tokenizer, torch)
        session.start()
        origin_ns = time.perf_counter_ns()
        sink = make_sink(args.sink)
        chunks = []
        events: list[dict[str, Any]] = []
        total_queued = total_dropped = total_consumed = 0
        for block_index, (start, end) in enumerate(plan):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            decode_started_ns = time.perf_counter_ns()
            start_event.record()
            decoded = session.decode(flat[:, :, start:end], global_start=start)
            decode_complete_ns = time.perf_counter_ns()
            raw_device = decoded_to_uint8_views(decoded, batch=1, n_views=N_VIEWS)
            end_event.record()
            torch.cuda.synchronize()
            device_ready_ns = time.perf_counter_ns()
            gpu_ms = float(start_event.elapsed_time(end_event))
            raw_host = raw_device.cpu().numpy()
            host_ready_ns = time.perf_counter_ns()

            frames = int(raw_host.shape[1])
            sink_started_ns = sink_complete_ns = None
            sink_result = None
            queued = dropped = consumed = 0
            if sink is not None:
                queued = frames
                sink_started_ns = time.perf_counter_ns()
                sink_result = sink(raw_host)
                sink_complete_ns = time.perf_counter_ns()
                consumed = int(sink_result["frames_consumed"])
            total_queued += queued
            total_dropped += dropped
            total_consumed += consumed
            chunks.append(raw_host)
            events.append(
                {
                    "block_index": block_index,
                    "global_latent_start": start,
                    "global_latent_end_exclusive": end,
                    "decoded_frames": frames,
                    "decode_started_ns": decode_started_ns,
                    "decode_complete_ns": decode_complete_ns,
                    "device_ready_ns": device_ready_ns,
                    "host_ready_ns": host_ready_ns,
                    "decode_start_offset_ms": (decode_started_ns - origin_ns) / 1e6,
                    "decode_return_ms": (decode_complete_ns - decode_started_ns) / 1e6,
                    "device_ready_ms": (device_ready_ns - decode_started_ns) / 1e6,
                    "host_ready_ms": (host_ready_ns - decode_started_ns) / 1e6,
                    "gpu_decode_and_quantize_ms": gpu_ms,
                    "sink_started_ns": sink_started_ns,
                    "sink_complete_ns": sink_complete_ns,
                    "sink_result": sink_result,
                    "queued_frames": queued,
                    "dropped_frames": dropped,
                    "consumed_frames": consumed,
                }
            )
            del decoded, raw_device

    incremental_raw = np.concatenate(chunks, axis=1)
    comparison = compare_exact(incremental_raw, full_raw)
    canonical_raw = source_raw_views(
        raw_path,
        frames=int(full_raw.shape[1]),
        expected_height=int(full_raw.shape[2]),
        expected_width=int(full_raw.shape[3]),
    )
    canonical_comparison = compare_exact(full_raw, canonical_raw)
    # The required decode-equivalence gate is incremental versus a one-shot
    # decode of the identical latent prefix in this process.  The retained
    # directional raw was produced earlier from a full-length invocation; keep
    # that comparison as a provenance diagnostic, but do not conflate it with
    # the incremental cache equivalence question.
    timing_claimable = bool(comparison["exact"])

    args.output.mkdir(parents=True, exist_ok=True)
    incremental_path = args.output / "incremental_raw_u8.npy"
    full_path = args.output / "full_raw_u8.npy"
    events_path = args.output / "block_events.jsonl"
    atomic_save_npy(incremental_path, incremental_raw)
    atomic_save_npy(full_path, full_raw)
    write_jsonl(events_path, events)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    timings = {
        "claimable": timing_claimable,
        "withheld_reason": (
            None
            if timing_claimable
            else "raw uint8 exactness gate failed; timings are diagnostic only"
        ),
        "incremental": {
            "decode_return": percentile_summary(
                row["decode_return_ms"] for row in events
            ),
            "device_ready": percentile_summary(
                row["device_ready_ms"] for row in events
            ),
            "host_ready": percentile_summary(row["host_ready_ms"] for row in events),
            "gpu_decode_and_quantize": percentile_summary(
                row["gpu_decode_and_quantize_ms"] for row in events
            ),
            "gpu_total_ms": float(
                sum(row["gpu_decode_and_quantize_ms"] for row in events)
            ),
        },
        "full": {
            "decode_return_ms": (full_decode_complete_ns - full_start_ns) / 1e6,
            "device_ready_ms": (full_device_ready_ns - full_start_ns) / 1e6,
            "host_ready_ms": (full_host_ready_ns - full_start_ns) / 1e6,
            "gpu_decode_and_quantize_ms": full_gpu_ms,
        },
    }
    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "immutable": True,
        "path_base": "manifest_directory",
        "identity": identity,
        "probe": {
            "source_latent_shape_bcvthw": [int(x) for x in latent.shape],
            "flat_latent_shape_bvcthw": [int(x) for x in flat.shape],
            "blocks": len(plan),
            "block_latents": BLOCK_LATENTS,
            "global_normalization_slices": [
                {"start": start, "end_exclusive": end} for start, end in plan
            ],
            "cache": {
                "cleared_once_at_session_start": True,
                "per_chunk_clear_decoder_cache": False,
            },
        },
        "equivalence": {
            "incremental_vs_full": comparison,
            "full_vs_canonical_source_raw": canonical_comparison,
            "canonical_reference_note": (
                "diagnostic comparison to the independently produced full-length "
                "directional decode; not part of the identical-input incremental "
                "versus one-shot prefix gate"
            ),
            "passed": timing_claimable,
        },
        "timings": timings,
        "sink": {
            "kind": args.sink,
            "semantics": (
                "optional synchronous host callback consumption; this is not "
                "physical display presentation"
            ),
            "queued_frames": total_queued,
            "dropped_frames": total_dropped,
            "consumed_frames": total_consumed,
            "physical_presentation_measured": False,
        },
        "environment": environment,
        "artifacts": {
            "incremental_raw_u8": artifact_metadata(
                incremental_path, root=args.output
            ),
            "full_raw_u8": artifact_metadata(full_path, root=args.output),
            "block_events": artifact_metadata(events_path, root=args.output),
        },
        "summary": {
            "equivalence_passed": timing_claimable,
            "max_error": comparison["max_error"],
            "canonical_max_error": canonical_comparison["max_error"],
            "views": comparison["views"],
            "frames": int(incremental_raw.shape[1]),
            "timing_claimable": timing_claimable,
        },
    }
    sealed = seal_manifest(manifest_body)
    atomic_write_json(manifest_path, sealed)
    print(json.dumps(manifest_body["summary"], sort_keys=True))
    return sealed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exactness-gated Wan2.1 cached incremental decode probe"
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rollout-id", default=DEFAULT_ROLLOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-blocks", type=int, default=2)
    parser.add_argument(
        "--sink",
        choices=("none", "hash"),
        default="hash",
        help="optional synchronous host callback (never physical presentation)",
    )
    return parser.parse_args()


def main() -> None:
    try:
        result = run(parse_args())
    except (ProbeError, ValueError) as error:
        raise SystemExit(f"incremental decode probe refused: {error}") from error
    if not result["equivalence"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
