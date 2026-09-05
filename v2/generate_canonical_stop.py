"""Generate 24 resumable canonical forward-to-stay STOP interventions."""

from __future__ import annotations

import argparse
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
    ArtifactError, artifact_metadata, atomic_numpy_save, atomic_numpy_savez,
    atomic_torch_save, atomic_write_json, commit_temporary, temporary_path,
    verify_rollout_artifacts,
)
from canonical_stop import (  # noqa: E402
    action_sequence, array_sha256, load_stop_protocol, stop_specs,
    validate_canonical_evidence,
)
from matched_counterfactual import configure_determinism  # noqa: E402
from preflight_directional import gamma_source_identity  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rf_support(protocol: Any) -> dict[tuple[int, int], int]:
    path = HERE / protocol.data["canonical_evidence"]["decoder_rf_manifest"]
    manifest = load_json(path)
    metadata = manifest["artifacts"]["hybrid_results"]
    rows_path = path.parent / metadata["path"]
    if file_sha256(rows_path) != metadata["sha256"]:
        raise RuntimeError("RF hybrid evidence hash mismatch")
    required = {int(row["admission_latent"]) for row in protocol.data["design"]["delays"]}
    starts = {}
    for row in (json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()):
        index = int(row["latent_index"])
        if index not in required:
            continue
        for view in row["views"]:
            if int(view["early_changed_frame_count"]) != 0:
                raise RuntimeError("RF evidence contains pre-support changes")
            starts[(index, int(view["view_index"]))] = int(view["earliest_changed_frame"])
    if set(starts) != {(index, view) for index in required for view in (0, 1)}:
        raise RuntimeError("RF support does not cover d0/d1 and both views")
    return starts


def upsert(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    for index, existing in enumerate(manifest["rollouts"]):
        if existing["rollout_id"] == record["rollout_id"]:
            manifest["rollouts"][index] = record
            return
    manifest["rollouts"].append(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "canonical_stop.json")
    parser.add_argument("--preflight", type=Path, default=HERE / "results" / "canonical_stop_preflight.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "canonical_stop")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = load_stop_protocol(args.config)
    gate = load_json(args.preflight)
    gate_hash = canonical_sha256({key: value for key, value in gate.items() if key != "gate_sha256"})
    if (
        gate.get("schema_version") != "rtwm-v2-canonical-stop-preflight-1"
        or gate.get("config_sha256") != protocol.file_sha256
        or gate.get("protocol_sha256") != protocol.canonical_sha256
        or gate.get("gate_sha256") != gate_hash
        or gate.get("allowed") is not True
        or gate.get("output_root") != str(args.output.resolve())
    ):
        raise RuntimeError("canonical STOP preflight is missing, stale, or bound elsewhere")
    evidence = validate_canonical_evidence(protocol, verify_artifacts=True)
    if not evidence["allowed"]:
        raise RuntimeError("canonical control/null evidence changed after preflight")
    source = gamma_source_identity()
    model_config = protocol.data["model"]
    if (
        source["commit"] != model_config["source_commit"]
        or source["diff_sha256"] != model_config["source_diff_sha256"]
    ):
        raise RuntimeError("Gamma source changed after preflight")
    supports = rf_support(protocol)
    specs = stop_specs(protocol)
    if args.limit is not None:
        specs = specs[:args.limit]
    controls = evidence["controls"]
    control_root = HERE / protocol.data["canonical_evidence"]["source_root"]

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
        I2VInference, IS_PREPROCESSED_KEY, to_with_skip_tensor,
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
        config_file=spec.config_file, guidance=float(model_config["guidance"]),
        shift=None, num_sampling_steps=spec.default_num_steps or 35, seed=1,
        context_parallel_size=1, experiment_opts=overrides,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = int(model_config["fps"])
    model = engine.model
    n_frames = int(model_config["n_frames"])
    action_hash = protocol.data["action_protocol"]["canonical_action_protocol_sha256"]
    forward = action_sequence(protocol, stop=False)
    stop = action_sequence(protocol, stop=True)
    generation_keyboard = torch.from_numpy(forward["keyboard"])
    generation_camera = torch.from_numpy(forward["camera"])
    post_keyboard = torch.from_numpy(stop["keyboard"])
    post_camera = torch.from_numpy(stop["camera"])
    tensor_hashes = {
        "generation_keyboard_sha256": array_sha256(forward["keyboard"]),
        "generation_camera_sha256": array_sha256(forward["camera"]),
        "post_keyboard_sha256": array_sha256(stop["keyboard"]),
        "post_camera_sha256": array_sha256(stop["camera"]),
    }
    scene_hashes = {
        scene: file_sha256(GAMMA / "data" / scene / "first_frame.png")
        for scene in protocol.data["design"]["scenes"]
    }
    manifest_base = {
        "schema_version": "rtwm-v2-canonical-stop-rollouts-1",
        **protocol.identity,
        "action_protocol_sha256": action_hash,
        "source": source,
        "model_artifacts": {
            "checkpoint_sha256": file_sha256(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
            "tokenizer_sha256": file_sha256(MODELS / "gamma-world" / "tokenizer.pth"),
        },
        "scene_hashes": scene_hashes,
        "control_manifest_sha256": protocol.data["canonical_evidence"]["source_manifest_sha256"],
        "null_gate_sha256": protocol.data["canonical_evidence"]["null_gate_sha256"],
        "planned_interventions": 24,
        "planned_rollout_ids": [row["rollout_id"] for row in stop_specs(protocol)],
        "rf_support": {f"{index}:v{view}": frame for (index, view), frame in supports.items()},
        "determinism": determinism,
        "rollouts": [],
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        for key, expected in manifest_base.items():
            if key != "rollouts" and manifest.get(key) != expected:
                raise RuntimeError(f"unsafe STOP resume: manifest {key!r} changed")
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
    for run_index, run in enumerate(specs, 1):
        existing = next((row for row in manifest["rollouts"] if row["rollout_id"] == run["rollout_id"]), None)
        if existing and existing.get("status") == "complete" and not args.force:
            verify_rollout_artifacts(
                args.output, existing,
                required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
            )
            print(f"[canonical-stop] skip verified {run['rollout_id']}", flush=True)
            continue
        scene, seed, admission = run["scene"], int(run["seed"]), int(run["admission_latent"])
        first_frame = GAMMA / "data" / scene / "first_frame.png"
        if file_sha256(first_frame) != scene_hashes[scene]:
            raise RuntimeError("scene changed")
        image = Image.open(first_frame)
        half = image.width // 2
        images = [
            np.asarray(image.crop((0, 0, half, image.height))).copy(),
            np.asarray(image.crop((half, 0, image.width, image.height))).copy(),
        ]

        def make_batch(keys, camera):
            return engine.build_inference_batch(
                images, PROMPT, [(keys, camera), (keys, camera)],
                num_frames=n_frames, num_conditional_frames=1,
            )

        post_batch = preprocess(make_batch(post_keyboard, post_camera))

        def patched_get(data_batch, **kwargs):
            pre_fn = original_get(data_batch, **kwargs)
            post_fn = original_get(post_batch, **kwargs)

            def dispatch(noisy_image_or_video, timestep, kv_cache=None, crossattn_cache=None,
                         current_start=None, current_end=None, start_frame_for_rope=None):
                selected = post_fn if (
                    start_frame_for_rope is not None and start_frame_for_rope >= admission
                ) else pre_fn
                return selected(
                    noisy_image_or_video, timestep, kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache, current_start=current_start,
                    current_end=current_end, start_frame_for_rope=start_frame_for_rope,
                )
            return dispatch

        output_dir = args.output / scene / f"seed{seed}" / run["arm"]
        paths = {
            "video": output_dir / "generated.mp4",
            "latent": output_dir / "latent.pt",
            "decoded_u8": output_dir / "decoded_u8.npy",
            "detector_frames": output_dir / "detector_frames.npz",
            "action_tensors": output_dir / "action_tensors.npz",
        }
        record = {
            **run, "action_protocol_sha256": action_hash,
            "action_tensor_hashes": tensor_hashes,
            "control_rollout_id": controls[(scene, seed)]["rollout_id"],
            "status": "running", "run_index": run_index,
        }
        upsert(manifest, record)
        atomic_write_json(manifest_path, manifest)
        engine.clear_cache()
        model.get_x0_fn_from_batch = patched_get
        started = time.perf_counter()
        try:
            video = engine.generate_from_batch(
                make_batch(generation_keyboard, generation_camera),
                guidance=float(model_config["guidance"]), seed=seed,
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
                detector[view, frame_index] = cv2.resize(
                    cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY),
                    (240, 160), interpolation=cv2.INTER_AREA,
                )
        control = controls[(scene, seed)]
        control_paths = verify_rollout_artifacts(control_root, control)
        control_latent = torch.load(control_paths["latent"], map_location="cpu")
        shape = (latent.shape[0], latent.shape[1], 2, latent.shape[2] // 2, latent.shape[3], latent.shape[4])
        latent_error = float(
            (latent.reshape(shape)[:, :, :, :admission]
             - control_latent.reshape(shape)[:, :, :, :admission]).abs().max()
        )
        control_decoded = np.load(control_paths["decoded_u8"], mmap_mode="r")
        raw_errors = []
        effect_starts = []
        for view in (0, 1):
            start = supports[(admission, view)]
            effect_starts.append(start)
            x0, x1 = view * half_width, (view + 1) * half_width
            raw_errors.append(float(np.max(np.abs(
                decoded[:start, :, x0:x1].astype(np.int16)
                - control_decoded[:start, :, x0:x1].astype(np.int16)
            ))))
        if latent_error != 0 or raw_errors != [0.0, 0.0]:
            record.update(
                status="failed", error="prefix mismatch",
                latent_prefix_max_error=latent_error, raw_prefix_max_errors=raw_errors,
            )
            upsert(manifest, record)
            atomic_write_json(manifest_path, manifest)
            raise ArtifactError(f"{run['rollout_id']} prefix mismatch")
        output_dir.mkdir(parents=True, exist_ok=True)
        with temporary_path(paths["video"], suffix=".mp4") as temporary:
            temporary.unlink()
            save_img_or_video(normalized[0], str(temporary.with_suffix("")), fps=int(model_config["fps"]))
            commit_temporary(temporary, paths["video"])
        atomic_torch_save(paths["latent"], latent, torch)
        atomic_numpy_save(paths["decoded_u8"], decoded, np)
        atomic_numpy_savez(paths["detector_frames"], np, views=detector)
        atomic_numpy_savez(
            paths["action_tensors"], np,
            generation_keyboard=forward["keyboard"], generation_camera=forward["camera"],
            post_keyboard=stop["keyboard"], post_camera=stop["camera"],
        )
        artifacts = {
            name: artifact_metadata(path, relative_to=args.output)
            for name, path in paths.items()
        }
        record.update(
            status="complete", total_s=elapsed, artifacts=artifacts,
            latent_prefix_max_error=latent_error, raw_prefix_max_errors=raw_errors,
            effect_start_frames=effect_starts,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        upsert(manifest, record)
        atomic_write_json(manifest_path, manifest)
        print(
            f"[canonical-stop] complete {run_index}/{len(specs)} {run['rollout_id']} {elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
