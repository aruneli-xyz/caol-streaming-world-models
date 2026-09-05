"""Resumably generate the independent 52-rollout Solaris-canonical grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SAFESWM = HERE.parents[1] / "safeswm"
GAMMA = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
PROMPT = "Two Minecraft players exploring the world"
sys.path.insert(0, str(HERE))

from artifacts import (  # noqa: E402
    ArtifactError,
    artifact_metadata,
    atomic_numpy_save,
    atomic_numpy_savez,
    atomic_torch_save,
    atomic_write_json,
    commit_temporary,
    temporary_path,
    verify_rollout_artifacts,
)
from directional_study import (  # noqa: E402
    action_protocol_sha256,
    action_sequence,
    load_directional_protocol,
    rollout_specs,
)
from matched_counterfactual import configure_determinism  # noqa: E402
from preflight_directional import gamma_source_identity  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_preflight(path: Path, protocol: Any, output: Path) -> dict[str, Any]:
    report = load_json(path)
    actual = canonical_sha256({key: value for key, value in report.items() if key != "report_sha256"})
    valid = (
        report.get("schema_version") == "rtwm-v2-directional-preflight-1"
        and report.get("config_sha256") == protocol.file_sha256
        and report.get("protocol_sha256") == protocol.canonical_sha256
        and report.get("report_sha256") == actual
        and report.get("allowed") is True
        and report.get("fresh_root", {}).get("path") == str(output.resolve())
        and report.get("fresh_root", {}).get("legacy_inputs_allowed") is False
    )
    if not valid:
        raise RuntimeError("fresh directional preflight is missing, stale, blocked, or bound elsewhere")
    return report


def raw_support_starts(protocol: Any) -> dict[int, int]:
    manifest_path = HERE / protocol.data["evidence"]["decoder_rf_manifest"]
    manifest = load_json(manifest_path)
    metadata = manifest["artifacts"]["hybrid_results"]
    rows_path = manifest_path.parent / metadata["path"]
    if file_sha256(rows_path) != metadata["sha256"]:
        raise RuntimeError("RF hybrid evidence hash mismatch")
    target = int(protocol.data["model"]["admission_latent"])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    row = next((item for item in rows if int(item["latent_index"]) == target), None)
    if row is None:
        raise RuntimeError(f"RF evidence lacks latent {target}")
    starts = {}
    for view in row["views"]:
        if int(view["early_changed_frame_count"]) != 0:
            raise RuntimeError("RF evidence has pre-support effects")
        starts[int(view["view_index"])] = int(view["earliest_changed_frame"])
    if set(starts) != {0, 1}:
        raise RuntimeError("RF evidence does not cover both views")
    return starts


def tensor_sha256(array: Any) -> str:
    return hashlib.sha256(array.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def upsert(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    for index, existing in enumerate(manifest["rollouts"]):
        if existing["rollout_id"] == record["rollout_id"]:
            manifest["rollouts"][index] = record
            return
    manifest["rollouts"].append(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--preflight", type=Path, default=HERE / "results" / "directional_d0_preflight.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "directional_d0")
    parser.add_argument("--stage", choices=["calibration", "validation", "test", "all"], default="all")
    parser.add_argument("--arms", help="comma-separated arm filter for safe targeted regeneration")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = load_directional_protocol(args.config)
    require_preflight(args.preflight, protocol, args.output)
    source = gamma_source_identity()
    model_config = protocol.data["model"]
    if (
        source["commit"] != model_config["source_commit"]
        or source["diff_sha256"] != model_config["source_diff_sha256"]
    ):
        raise RuntimeError("Gamma source identity changed after preflight")
    support_starts = raw_support_starts(protocol)
    all_specs = rollout_specs(protocol)
    specs = all_specs if args.stage == "all" else [
        row for row in all_specs if row["stage"] == args.stage
    ]
    if args.arms:
        selected_arms = set(args.arms.split(","))
        specs = [row for row in specs if row["arm"] in selected_arms]
    if args.limit is not None:
        specs = specs[:args.limit]

    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("SAFESWM_DUMP_LATENTS", "1")
    sys.path[:0] = [str(GAMMA), str(GAMMA / "scripts"), str(SAFESWM)]
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from gamma_world._src.gamma_world.inference.inference_i2v import (
        I2VInference,
        IS_PREPROCESSED_KEY,
        to_with_skip_tensor,
    )
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from gamma_world._src.imaginaire.visualize.video import save_img_or_video
    from inference import format_hydra_value

    torch.set_grad_enabled(False)
    determinism = configure_determinism(torch)
    determinism["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    spec = MODEL_SPECS["causal_few_step"]
    overrides = [f"{key}={format_hydra_value(value)}" for key, value in spec.config_overrides.items()]
    engine = I2VInference(
        experiment_name=spec.experiment,
        ckpt_path=str(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
        config_file=spec.config_file,
        guidance=float(model_config["guidance"]),
        shift=None,
        num_sampling_steps=spec.default_num_steps or 35,
        seed=1,
        context_parallel_size=1,
        experiment_opts=overrides,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = int(model_config["fps"])
    model = engine.model
    n_frames = int(model_config["n_frames"])
    admission_latent = int(model_config["admission_latent"])
    scene_hashes = {
        scene: file_sha256(GAMMA / "data" / scene / "first_frame.png")
        for scene in protocol.data["design"]["scenes"]
    }
    manifest_base = {
        "schema_version": "rtwm-v2-directional-rollouts-2",
        **protocol.identity,
        "action_protocol_sha256": action_protocol_sha256(protocol),
        "source": source,
        "model_artifacts": {
            "checkpoint_sha256": file_sha256(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
            "tokenizer_sha256": file_sha256(MODELS / "gamma-world" / "tokenizer.pth"),
        },
        "scene_hashes": scene_hashes,
        "raw_support_starts": {str(key): value for key, value in support_starts.items()},
        "planned_rollouts": 52,
        "planned_rollout_ids": [row["rollout_id"] for row in all_specs],
        "determinism": determinism,
        "rollouts": [],
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        for key, expected in manifest_base.items():
            if key != "rollouts" and manifest.get(key) != expected:
                raise RuntimeError(f"unsafe resume: manifest field {key!r} changed")
    else:
        manifest = manifest_base
        atomic_write_json(manifest_path, manifest)

    def preprocess(batch):
        if "video" in batch:
            batch["video"] = batch["video"].float()
            if not batch.get(IS_PREPROCESSED_KEY, False):
                batch["video"] = batch["video"] / 127.5 - 1.0
            batch["video"] = torch.clamp(batch["video"], -1, 1)
        batch[IS_PREPROCESSED_KEY] = True
        batch = to_with_skip_tensor(batch, **model.tensor_kwargs)
        engine.inplace_compute_text_embeddings_online(batch, use_negative_prompt=True)
        batch = model.get_data_batch_with_latent_view_indices(batch)
        model._normalize_video_databatch_inplace(batch)
        return batch

    original_get = model.get_x0_fn_from_batch
    forward_action = action_sequence(protocol, "canonical_forward")
    for run_index, run in enumerate(specs, 1):
        rollout_id = run["rollout_id"]
        existing = next((row for row in manifest["rollouts"] if row["rollout_id"] == rollout_id), None)
        if existing and existing.get("status") == "complete" and not args.force:
            verify_rollout_artifacts(
                args.output, existing,
                required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
            )
            print(f"[directional] skip verified {rollout_id}", flush=True)
            continue
        scene, seed = run["scene"], int(run["seed"])
        first_frame = GAMMA / "data" / scene / "first_frame.png"
        if file_sha256(first_frame) != scene_hashes[scene]:
            raise RuntimeError(f"scene changed before {rollout_id}")
        image = Image.open(first_frame)
        half = image.width // 2
        images = [
            np.asarray(image.crop((0, 0, half, image.height))).copy(),
            np.asarray(image.crop((half, 0, image.width, image.height))).copy(),
        ]
        post_action = action_sequence(protocol, run["transition"])
        generation_keyboard = torch.tensor(forward_action["keyboard"], dtype=torch.float32).unsqueeze(0)
        generation_camera = torch.tensor(forward_action["camera"], dtype=torch.float32).unsqueeze(0)
        post_keyboard = torch.tensor(post_action["keyboard"], dtype=torch.float32).unsqueeze(0)
        post_camera = torch.tensor(post_action["camera"], dtype=torch.float32).unsqueeze(0)
        action_hashes = {
            "generation_keyboard_sha256": tensor_sha256(generation_keyboard),
            "generation_camera_sha256": tensor_sha256(generation_camera),
            "post_keyboard_sha256": tensor_sha256(post_keyboard),
            "post_camera_sha256": tensor_sha256(post_camera),
        }

        def make_batch(keys, cameras):
            return engine.build_inference_batch(
                images, PROMPT, [(keys, cameras), (keys, cameras)],
                num_frames=n_frames, num_conditional_frames=1,
            )

        is_intervention = run["arm"] in {"back", "yaw_positive"}
        # Test controls use the identical d0 dispatcher plumbing, with forward
        # on both sides, so independently generated paired prefixes are exact.
        is_hybrid_dispatch = is_intervention or run["arm"] == "control"
        if is_hybrid_dispatch:
            post_batch = preprocess(make_batch(post_keyboard, post_camera))

            def patched_get(data_batch, **kwargs):
                pre_fn = original_get(data_batch, **kwargs)
                post_fn = original_get(post_batch, **kwargs)

                def dispatch(noisy_image_or_video, timestep, kv_cache=None, crossattn_cache=None,
                             current_start=None, current_end=None, start_frame_for_rope=None):
                    selected = post_fn if (
                        start_frame_for_rope is not None
                        and start_frame_for_rope >= admission_latent
                    ) else pre_fn
                    return selected(
                        noisy_image_or_video, timestep, kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache, current_start=current_start,
                        current_end=current_end, start_frame_for_rope=start_frame_for_rope,
                    )
                return dispatch
            model.get_x0_fn_from_batch = patched_get
        else:
            model.get_x0_fn_from_batch = original_get

        output_dir = args.output / run["stage"] / scene / f"seed{seed}" / run["arm"]
        paths = {
            "video": output_dir / "generated.mp4",
            "latent": output_dir / "latent.pt",
            "decoded_u8": output_dir / "decoded_u8.npy",
            "detector_frames": output_dir / "detector_frames.npz",
            "action_tensors": output_dir / "action_tensors.npz",
        }
        record = {
            **run,
            "admission_latent": admission_latent if is_intervention else None,
            "action_protocol_sha256": action_protocol_sha256(protocol),
            "action_tensor_hashes": action_hashes,
            "status": "running",
            "run_index": run_index,
        }
        upsert(manifest, record)
        atomic_write_json(manifest_path, manifest)
        engine.clear_cache()
        started = time.perf_counter()
        try:
            video = engine.generate_from_batch(
                make_batch(generation_keyboard, generation_camera),
                guidance=float(model_config["guidance"]),
                seed=seed,
            )
        except Exception as error:
            record.update(status="failed", error=repr(error))
            upsert(manifest, record)
            atomic_write_json(manifest_path, manifest)
            raise
        finally:
            model.get_x0_fn_from_batch = original_get
        elapsed = time.perf_counter() - started

        normalized = ((video + 1.0) / 2.0).clamp(0, 1)
        latent = getattr(engine, "last_sample_latent", None)
        if latent is None:
            raise RuntimeError("sampler latent capture missing")
        decoded = (
            normalized[0].mul(255).floor().to(torch.uint8)
            .permute(1, 2, 3, 0).cpu().numpy()
        )
        half_width = decoded.shape[2] // 2
        detector = np.empty((2, n_frames, 160, 240), dtype=np.uint8)
        for frame_index, frame in enumerate(decoded):
            for view, pixels in enumerate((frame[:, :half_width], frame[:, half_width:])):
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                detector[view, frame_index] = cv2.resize(
                    gray, (240, 160), interpolation=cv2.INTER_AREA
                )

        latent_prefix_error = None
        raw_prefix_errors = None
        if is_intervention:
            control = next(
                (
                    row for row in manifest["rollouts"]
                    if row.get("scene") == scene and int(row.get("seed", -1)) == seed
                    and row.get("arm") == "control" and row.get("status") == "complete"
                ),
                None,
            )
            if control is None:
                raise ArtifactError(f"fresh canonical control missing before {rollout_id}")
            control_paths = verify_rollout_artifacts(args.output, control)
            control_latent = torch.load(control_paths["latent"], map_location="cpu")
            if latent.shape != control_latent.shape or latent.ndim != 5 or latent.shape[2] % 2:
                raise ArtifactError("control/intervention latent shape mismatch")
            shape = (latent.shape[0], latent.shape[1], 2, latent.shape[2] // 2, latent.shape[3], latent.shape[4])
            latent_views = latent.reshape(shape)
            control_views = control_latent.reshape(shape)
            latent_prefix_error = float(
                (latent_views[:, :, :, :admission_latent]
                 - control_views[:, :, :, :admission_latent]).abs().max()
            )
            control_decoded = np.load(control_paths["decoded_u8"], mmap_mode="r")
            raw_prefix_errors = []
            for view in (0, 1):
                x0, x1, end = view * half_width, (view + 1) * half_width, support_starts[view]
                raw_prefix_errors.append(float(np.max(np.abs(
                    decoded[:end, :, x0:x1].astype(np.int16)
                    - control_decoded[:end, :, x0:x1].astype(np.int16)
                ))))
            if latent_prefix_error != 0 or any(value != 0 for value in raw_prefix_errors):
                record.update(
                    status="failed", error="prefix mismatch",
                    latent_prefix_max_error=latent_prefix_error,
                    raw_prefix_max_errors=raw_prefix_errors,
                )
                upsert(manifest, record)
                atomic_write_json(manifest_path, manifest)
                raise ArtifactError(f"{rollout_id} failed exact fresh-control prefix")

        output_dir.mkdir(parents=True, exist_ok=True)
        with temporary_path(paths["video"], suffix=".mp4") as temporary_video:
            temporary_video.unlink()
            save_img_or_video(
                normalized[0], str(temporary_video.with_suffix("")),
                fps=int(model_config["fps"]),
            )
            commit_temporary(temporary_video, paths["video"])
        atomic_torch_save(paths["latent"], latent, torch)
        atomic_numpy_save(paths["decoded_u8"], decoded, np)
        atomic_numpy_savez(paths["detector_frames"], np, views=detector)
        atomic_numpy_savez(
            paths["action_tensors"], np,
            generation_keyboard=generation_keyboard.cpu().numpy(),
            generation_camera=generation_camera.cpu().numpy(),
            post_keyboard=post_keyboard.cpu().numpy(),
            post_camera=post_camera.cpu().numpy(),
        )
        artifacts = {
            name: artifact_metadata(path, relative_to=args.output)
            for name, path in paths.items()
        }
        record.update(
            status="complete", total_s=elapsed, artifacts=artifacts,
            latent_prefix_max_error=latent_prefix_error,
            raw_prefix_max_errors=raw_prefix_errors,
            raw_support_starts=support_starts if is_intervention else None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        upsert(manifest, record)
        atomic_write_json(manifest_path, manifest)
        print(
            f"[directional] complete {run_index}/{len(specs)} {rollout_id} {elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
