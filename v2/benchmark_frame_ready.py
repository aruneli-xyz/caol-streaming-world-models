"""Timestamp action admission, latent commit, decode, and frame readiness.

This benchmark injects a synthetic action-arrival event at the latent block
boundary aligned with the configured control change. It reports host-ready
frames, not physical display presentation.

Run from the Gamma-World checkout:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/torchrun --nproc_per_node=1 \
      /path/to/rtwm/v2/benchmark_frame_ready.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


V2 = Path(__file__).resolve().parent
RTWM = V2.parent
SAFESWM = RTWM.parent / "safeswm"
GAMMA_REPO = SAFESWM / "external" / "Gamma-World"
DEFAULT_CONFIG = V2 / "config" / "confirmatory.json"
DEFAULT_OUTPUT = V2 / "results" / "frame_ready"
PROMPT = "Two Minecraft players exploring the world"
LOCKED_SCENE = "buildTower_normal"
LOCKED_SEEDS = [301, 302]

sys.path.insert(0, str(V2))
sys.path.insert(0, str(RTWM))

from artifacts import artifact_metadata, atomic_write_json  # noqa: E402
from gates import require_confirmatory_gate  # noqa: E402
from matched_counterfactual import configure_determinism  # noqa: E402
from preflight import git_source_identity, require_source  # noqa: E402
from protocol import canonical_sha256, file_sha256, load_protocol  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def relative_ms(timestamp_ns: int, origin_ns: int) -> float:
    return (timestamp_ns - origin_ns) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--gate",
        type=Path,
        default=V2 / "results" / "confirmatory_gate" / "gate_report.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-patch",
        type=Path,
        default=V2 / "results" / "provenance" / "gamma_world_source.patch",
    )
    parser.add_argument("--scene", default=LOCKED_SCENE)
    parser.add_argument("--seeds", default=",".join(map(str, LOCKED_SEEDS)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    require_confirmatory_gate(args.gate, protocol)
    source = git_source_identity(GAMMA_REPO)
    require_source(protocol, source)
    source_patch_sha256 = file_sha256(args.source_patch)
    if source_patch_sha256 != source["diff_sha256"]:
        raise RuntimeError(
            "archived Gamma source patch does not match the active source diff"
        )
    config = protocol.data
    model_config = config["model"]
    confirmatory = config["confirmatory"]
    if args.scene not in confirmatory["scenes"]:
        raise ValueError(f"scene is not in the locked protocol: {args.scene}")
    delays = list(confirmatory["test"]["delays"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.scene != LOCKED_SCENE or seeds != LOCKED_SEEDS:
        raise ValueError(
            "frame-ready protocol is locked to "
            f"scene={LOCKED_SCENE}, seeds={LOCKED_SEEDS}"
        )

    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(GAMMA_REPO))

    import numpy as np
    import torch
    from PIL import Image

    from driver_v2 import BlockwiseSession
    from tracks import N_FRAMES, build_engine, make_x0

    if N_FRAMES != int(model_config["n_frames"]):
        raise RuntimeError("tracks.N_FRAMES does not match the confirmatory protocol")
    determinism = configure_determinism(torch)
    determinism["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    torch.set_grad_enabled(False)
    device = torch.cuda.current_device()
    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": int(
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        ),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }

    engine = build_engine()
    image_path = GAMMA_REPO / "data" / args.scene / "first_frame.png"
    image = Image.open(image_path)
    half = image.width // 2
    images = [
        np.asarray(image.crop((0, 0, half, image.height))).copy(),
        np.asarray(image.crop((half, 0, image.width, image.height))).copy(),
    ]
    n_frames = int(model_config["n_frames"])
    change_frame = int(confirmatory["change_frame"])
    stride = int(model_config["latent_stride_pixels"])
    change_latent = change_frame // stride
    forward = ["forward"] * n_frames
    stop = ["forward"] * change_frame + ["stay"] * (n_frames - change_frame)
    batch_forward, x0_forward = make_x0(engine, forward, images, PROMPT)
    _, x0_stop = make_x0(engine, stop, images, PROMPT)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    identity = {
        "schema_version": "rtwm-v2-frame-ready-1",
        **protocol.identity,
        "gamma_source": source,
        "source_patch": {
            "path": str(args.source_patch.resolve().relative_to(V2)),
            "sha256": source_patch_sha256,
        },
        "benchmark_script_sha256": file_sha256(Path(__file__)),
        "dependencies": {
            "driver_v2.py": file_sha256(RTWM / "driver_v2.py"),
            "tracks.py": file_sha256(RTWM / "tracks.py"),
        },
        "environment": environment,
        "scene": args.scene,
        "seeds": seeds,
        "event_semantics": {
            "action_received": "synthetic event injected at the block boundary aligned to change_frame",
            "frame_ready": "measured raw-support frame copied synchronously to host memory",
            "presentation": "not measured",
        },
        "determinism": determinism,
    }
    if existing is not None and not args.force:
        for key, expected in identity.items():
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"existing frame-ready manifest {key} mismatch; use a new output"
                )

    runs: list[dict[str, Any]] = [] if existing is None or args.force else list(
        existing.get("runs", [])
    )
    completed = {
        (int(row["seed"]), str(row["delay"]))
        for row in runs
        if row.get("status") == "complete"
    }

    for seed_index, seed in enumerate(seeds):
        ordered_delays = delays if seed_index % 2 == 0 else list(reversed(delays))
        for order_position, delay in enumerate(ordered_delays):
            delay_name = str(delay["name"])
            key = (seed, delay_name)
            if key in completed and not args.force:
                print(f"[frame-ready] skip complete seed{seed}/{delay_name}", flush=True)
                continue
            admission_latent = int(delay["admission_latent"])
            admission_frame = admission_latent * stride
            effect_start_frame = admission_latent * stride - (stride - 1)
            if effect_start_frame >= n_frames:
                raise RuntimeError("effect support begins outside the rendered horizon")

            engine.clear_cache()
            session = BlockwiseSession(engine, batch_forward, seed=seed)
            if session.nfpb != int(model_config["latent_frames_per_block"]):
                raise RuntimeError("runtime block size differs from locked protocol")
            origin_ns = time.perf_counter_ns()
            action_received_ns: int | None = None
            action_admitted_ns: int | None = None
            events: list[dict[str, Any]] = []
            decoded_result = None

            while session.block_index < session.num_blocks:
                latent_start = session.block_index * session.nfpb
                if latent_start == change_latent:
                    action_received_ns = time.perf_counter_ns()
                    events.append(
                        {
                            "event": "action_received",
                            "monotonic_ns": action_received_ns,
                            "relative_ms": relative_ms(action_received_ns, origin_ns),
                            "change_frame": change_frame,
                            "change_latent": change_latent,
                        }
                    )
                use_stop = latent_start >= admission_latent
                if use_stop and action_admitted_ns is None:
                    if action_received_ns is None:
                        raise RuntimeError("action admitted before receipt")
                    action_admitted_ns = time.perf_counter_ns()
                    events.append(
                        {
                            "event": "action_admitted",
                            "monotonic_ns": action_admitted_ns,
                            "relative_ms": relative_ms(action_admitted_ns, origin_ns),
                            "admission_frame": admission_frame,
                            "admission_latent": admission_latent,
                        }
                    )

                result = session.step_block(x0_stop if use_stop else x0_forward)
                events.append(
                    {
                        "event": "latent_block_committed",
                        "block_index": result.block_index,
                        "latent_start": latent_start,
                        "latent_end": latent_start + session.nfpb,
                        "condition": "stop" if use_stop else "forward",
                        "block_started_ns": result.block_started_ns,
                        "denoise_complete_ns": result.denoise_complete_ns,
                        "latent_committed_ns": result.latent_committed_ns,
                        "relative_ms": relative_ms(
                            result.latent_committed_ns, origin_ns
                        ),
                        "denoise_ms": result.denoise_ms,
                        "commit_ms": result.commit_ms,
                        "steps_used": result.steps_used,
                    }
                )
                if latent_start == admission_latent:
                    decoded_result = session.decode_current_timed()
                    events.append(
                        {
                            "event": "decode_complete",
                            "decode_started_ns": decoded_result.decode_started_ns,
                            "decode_complete_ns": decoded_result.decode_complete_ns,
                            "relative_ms": relative_ms(
                                decoded_result.decode_complete_ns, origin_ns
                            ),
                            "decode_ms": decoded_result.decode_ms,
                        }
                    )
                    video = decoded_result.video
                    if video.ndim != 5 or video.shape[0] != 1:
                        raise RuntimeError(
                            f"unexpected decoded tensor shape: {list(video.shape)}"
                        )
                    if video.shape[2] != 2 * session.num_pixel_frames_per_view:
                        raise RuntimeError(
                            "decoded temporal axis does not contain two full views"
                        )
                    ready_indices = [
                        effect_start_frame,
                        session.num_pixel_frames_per_view + effect_start_frame,
                    ]
                    ready_frames = video[0, :, ready_indices].detach().float().cpu()
                    frame_ready_ns = time.perf_counter_ns()
                    events.append(
                        {
                            "event": "frame_ready",
                            "monotonic_ns": frame_ready_ns,
                            "relative_ms": relative_ms(frame_ready_ns, origin_ns),
                            "effect_start_frame": effect_start_frame,
                            "view_indices": ready_indices,
                            "host_tensor_shape": list(ready_frames.shape),
                            "host_tensor_bytes": (
                                ready_frames.numel() * ready_frames.element_size()
                            ),
                        }
                    )
                    break

            if (
                action_received_ns is None
                or action_admitted_ns is None
                or decoded_result is None
            ):
                raise RuntimeError("incomplete frame-ready event chain")
            latent_event = next(
                row
                for row in reversed(events)
                if row["event"] == "latent_block_committed"
            )
            frame_event = events[-1]
            run_id = f"{args.scene}__seed{seed}__{delay_name}"
            event_path = args.output / "events" / f"{run_id}.jsonl"
            write_jsonl(event_path, events)
            run = {
                "run_id": run_id,
                "status": "complete",
                "seed": seed,
                "scene": args.scene,
                "delay": delay_name,
                "run_order_within_seed": order_position,
                "change_frame": change_frame,
                "admission_frame": admission_frame,
                "admission_lag_frames": admission_frame - change_frame,
                "effect_start_frame": effect_start_frame,
                "action_to_admission_ms": (
                    action_admitted_ns - action_received_ns
                )
                / 1_000_000,
                "action_to_latent_commit_ms": (
                    int(latent_event["latent_committed_ns"]) - action_received_ns
                )
                / 1_000_000,
                "decode_ms": decoded_result.decode_ms,
                "action_to_frame_ready_ms": (
                    int(frame_event["monotonic_ns"]) - action_received_ns
                )
                / 1_000_000,
                "events": artifact_metadata(event_path, relative_to=args.output),
            }
            runs = [row for row in runs if row.get("run_id") != run_id] + [run]
            payload = {**identity, "runs": sorted(runs, key=lambda row: row["run_id"])}
            payload["manifest_sha256"] = canonical_sha256(payload)
            atomic_write_json(manifest_path, payload)
            print(
                f"[frame-ready] {run_id} ready="
                f"{run['action_to_frame_ready_ms']:.1f}ms",
                flush=True,
            )
            del decoded_result, ready_frames
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[frame-ready] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
