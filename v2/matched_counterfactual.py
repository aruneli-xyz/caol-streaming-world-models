"""Resumable matched STOP-versus-forward Gamma-World generation.

Run from the Gamma-World checkout with its virtual environment:

    RTWM_V2_STAGE=stage1 .venv/bin/torchrun --nproc_per_node=1 \
        /path/to/rtwm/v2/matched_counterfactual.py
"""

from __future__ import annotations

import argparse
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
MODELS = SAFESWM / "models"
DEFAULT_CONFIG = V2 / "config" / "pilot.json"
DEFAULT_OUTPUT = V2 / "results" / "matched"
PROMPT = "Two Minecraft players exploring the world"

sys.path.insert(0, str(V2))

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
from gates import require_confirmatory_gate, require_stage2_gate  # noqa: E402
from preflight import git_source_identity, require_source  # noqa: E402
from protocol import file_sha256, load_protocol  # noqa: E402


def configure_determinism(torch_module) -> dict[str, Any]:
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms_warn_only": True,
    }


def load_manifest(path: Path, base: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text())
        for key, expected in base.items():
            if key == "rollouts":
                continue
            if key not in manifest:
                raise ValueError(f"existing manifest lacks resume identity field {key!r}")
            if manifest[key] != expected:
                raise ValueError(
                    f"existing manifest {key!r} does not match this run "
                    f"({manifest[key]!r} != {expected!r})"
                )
        if not isinstance(manifest.get("rollouts"), list):
            raise ValueError("existing manifest rollouts must be a list")
        return manifest
    atomic_write_json(path, base)
    return base


def validate_resume_record(
    record: dict[str, Any],
    expected: dict[str, Any],
    output_root: Path,
) -> None:
    """Verify a completed record before trusting it as resumable."""
    for key, value in expected.items():
        if record.get(key) != value:
            raise ArtifactError(
                f"rollout {record.get('rollout_id')} {key} mismatch: "
                f"{record.get(key)!r} != {value!r}"
            )
    if record.get("status") != "complete":
        raise ArtifactError(f"rollout {record.get('rollout_id')} is not complete")
    verify_rollout_artifacts(output_root, record)


def upsert_rollout(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    rollouts = manifest.setdefault("rollouts", [])
    for index, existing in enumerate(rollouts):
        if existing["rollout_id"] == record["rollout_id"]:
            rollouts[index] = record
            return
    rollouts.append(record)


def build_specs(config: dict[str, Any], stage: str, seeds: list[int] | None) -> list[dict[str, Any]]:
    if "confirmatory" in config:
        confirmatory = config["confirmatory"]
        scenes = confirmatory["scenes"]
        specs: list[dict[str, Any]] = []
        selected_splits = (
            ["calibration", "validation", "test"]
            if stage == "all"
            else [stage]
        )
        for split in selected_splits:
            split_config = confirmatory[split]
            selected_seeds = seeds or [int(seed) for seed in split_config["seeds"]]
            for scene in scenes:
                for seed in selected_seeds:
                    if split in ("calibration", "validation"):
                        for arm in split_config["arms"]:
                            commands = confirmatory["calibration"]["commands"][arm]
                            specs.append(
                                {
                                    "split": split,
                                    "scene": scene,
                                    "seed": seed,
                                    "arm": arm,
                                    "delay_blocks": None,
                                    "admission_latent": None,
                                    "pre_command": commands["pre"],
                                    "post_command": commands["post"],
                                }
                            )
                    else:
                        control = split_config["arms"]["control"]
                        specs.append(
                            {
                                "split": "test",
                                "scene": scene,
                                "seed": seed,
                                "arm": "control",
                                "delay_blocks": None,
                                "admission_latent": None,
                                "pre_command": control["pre"],
                                "post_command": control["post"],
                            }
                        )
                        stop = split_config["arms"]["stop"]
                        for delay in split_config["delays"]:
                            specs.append(
                                {
                                    "split": "test",
                                    "scene": scene,
                                    "seed": seed,
                                    "arm": f"stop_{delay['name']}",
                                    "delay_blocks": int(delay["delay_blocks"]),
                                    "admission_latent": int(delay["admission_latent"]),
                                    "pre_command": stop["pre"],
                                    "post_command": stop["post"],
                                }
                            )
        return specs

    pilot = config["pilot"]
    selected_seeds = seeds or [int(seed) for seed in pilot["seeds"]]
    if stage == "stage1":
        scenes = pilot["stage1_scenes"]
    elif stage == "stage2":
        scenes = pilot["stage2_scenes"]
    else:
        scenes = pilot["stage1_scenes"] + pilot["stage2_scenes"]

    specs: list[dict[str, Any]] = []
    for scene in scenes:
        for seed in selected_seeds:
            specs.append(
                {
                    "split": "exploratory",
                    "scene": scene,
                    "seed": seed,
                    "arm": "control",
                    "delay_blocks": None,
                    "pre_command": "forward",
                    "post_command": "forward",
                }
            )
            for delay in pilot["delays"]:
                specs.append(
                    {
                        "split": "exploratory",
                        "scene": scene,
                        "seed": seed,
                        "arm": f"stop_d{delay}",
                        "delay_blocks": int(delay),
                        "pre_command": "forward",
                        "post_command": "stay",
                    }
                )

    canary = pilot["duplicate_control_canary"]
    if canary["enabled"] and canary["scene"] in scenes and int(canary["seed"]) in selected_seeds:
        specs.append(
            {
                "split": "exploratory",
                "scene": canary["scene"],
                "seed": int(canary["seed"]),
                "arm": "control_canary",
                "delay_blocks": None,
                "pre_command": "forward",
                "post_command": "forward",
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "calibration", "validation", "test", "all"],
        default=os.environ.get("RTWM_V2_STAGE", "stage1"),
    )
    parser.add_argument("--seeds", default=None, help="comma-separated override")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=V2 / "results" / "matched_scoring" / "gate_report.json",
        help="hash-bound gate required by stage2 or confirmatory test generation",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    config = protocol.data
    model_config = config["model"]
    experiment_config = config.get("pilot", config.get("confirmatory"))
    change_frame = int(experiment_config["change_frame"])
    n_frames = int(model_config["n_frames"])
    seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else None
    specs = build_specs(config, args.stage, seeds)
    if args.limit is not None:
        specs = specs[: args.limit]

    source = git_source_identity(GAMMA_REPO)
    require_source(protocol, source)
    manifest_path = args.output / "manifest.json"
    if "pilot" in config and args.stage in ("stage2", "all"):
        require_stage2_gate(args.gate_report, protocol, manifest_path=manifest_path)
    if "confirmatory" in config and args.stage in ("test", "all"):
        require_confirmatory_gate(args.gate_report, protocol)

    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("SAFESWM_DUMP_LATENTS", "1")
    sys.path.insert(0, str(GAMMA_REPO))
    sys.path.insert(0, str(GAMMA_REPO / "scripts"))
    sys.path.insert(0, str(SAFESWM))

    import numpy as np
    import torch
    import cv2
    from PIL import Image

    from gamma.actions import commands_to_action_json
    from gamma_world._src.gamma_world.inference.inference_i2v import (
        I2VInference,
        IS_PREPROCESSED_KEY,
        to_with_skip_tensor,
    )
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from gamma_world._src.imaginaire.visualize.video import save_img_or_video
    from inference import format_hydra_value

    determinism = configure_determinism(torch)
    determinism["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    torch.set_grad_enabled(False)
    spec = MODEL_SPECS["causal_few_step"]
    experiment_opts = [f"{key}={format_hydra_value(value)}" for key, value in spec.config_overrides.items()]
    engine = I2VInference(
        experiment_name=spec.experiment,
        ckpt_path=str(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
        config_file=spec.config_file,
        guidance=float(model_config["guidance"]),
        shift=None,
        num_sampling_steps=spec.default_num_steps or 35,
        seed=1,
        context_parallel_size=1,
        experiment_opts=experiment_opts,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = int(model_config["fps"])
    model = engine.model
    nfpb = int(model.num_frame_per_block)
    change_latent = change_frame // int(model_config["latent_stride_pixels"])
    change_block = change_latent // nfpb

    checkpoint_path = MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"
    tokenizer_path = MODELS / "gamma-world" / "tokenizer.pth"
    model_artifacts = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    configured_scenes = (
        set(config["pilot"]["stage1_scenes"]) | set(config["pilot"]["stage2_scenes"])
        if "pilot" in config
        else set(config["confirmatory"]["scenes"])
    )
    scene_hashes = {
        scene: file_sha256(GAMMA_REPO / "data" / scene / "first_frame.png")
        for scene in sorted(configured_scenes)
    }
    manifest_base: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "experiment": config["experiment"],
        "config_path": str(protocol.path),
        **protocol.identity,
        "gamma_source": source,
        "checkpoint": model_config["checkpoint"],
        "model_artifacts": model_artifacts,
        "scene_hashes": scene_hashes,
        "n_frames": n_frames,
        "fps": int(model_config["fps"]),
        "change_frame": change_frame,
        "nfpb": nfpb,
        "latent_stride_pixels": int(model_config["latent_stride_pixels"]),
        "determinism": determinism,
        "rollouts": [],
    }
    manifest = load_manifest(manifest_path, manifest_base)
    manifest["determinism"] = determinism
    atomic_write_json(manifest_path, manifest)

    def action_tensors(pre: str, post: str):
        commands = [pre] * change_frame + [post] * (n_frames - change_frame)
        action = commands_to_action_json(commands)
        keyboard = torch.tensor(action["keyboard"], dtype=torch.float32).unsqueeze(0)
        camera = torch.tensor(action["camera"], dtype=torch.float32).unsqueeze(0)
        return keyboard, camera

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
    for run_index, run_spec in enumerate(specs, start=1):
        scene = run_spec["scene"]
        seed = int(run_spec["seed"])
        arm = run_spec["arm"]
        rollout_id = f"{scene}__seed{seed}__{arm}"
        output_dir = args.output / scene / f"seed{seed}" / arm
        video_path = output_dir / "generated.mp4"
        latent_path = output_dir / "latent.pt"
        decoded_u8_path = output_dir / "decoded_u8.npy"
        detector_frames_path = output_dir / "detector_frames.npz"
        first_frame = GAMMA_REPO / "data" / scene / "first_frame.png"
        scene_sha256 = file_sha256(first_frame)
        delay = run_spec["delay_blocks"]
        explicit_admission = run_spec.get("admission_latent")
        admission_latent = (
            int(explicit_admission)
            if explicit_admission is not None
            else (change_block + int(delay or 0)) * nfpb
        )
        admission_frame = int(admission_latent * model_config["latent_stride_pixels"])
        if arm.startswith("control"):
            admission_frame = change_frame
            admission_latent = change_latent
        existing = next(
            (record for record in manifest["rollouts"] if record["rollout_id"] == rollout_id),
            None,
        )
        resume_identity = {
            "rollout_id": rollout_id,
            "pair_key": f"{scene}__seed{seed}",
            "split": run_spec["split"],
            "scene": scene,
            "scene_sha256": scene_sha256,
            "seed": seed,
            "arm": arm,
            "delay_blocks": run_spec["delay_blocks"],
            "pre_command": run_spec["pre_command"],
            "post_command": run_spec["post_command"],
            "change_frame": change_frame,
            "admission_latent": int(admission_latent),
            "admission_frame": admission_frame,
        }
        if not args.force and existing is not None and existing.get("status") == "complete":
            try:
                validate_resume_record(existing, resume_identity, args.output)
            except ArtifactError as error:
                raise RuntimeError(
                    f"refusing unsafe resume for {rollout_id}: {error}; "
                    "use a new output root or --force to regenerate it"
                ) from error
            print(f"[v2] skip verified complete {rollout_id}", flush=True)
            continue

        image = Image.open(first_frame)
        half = image.width // 2
        images = [
            np.asarray(image.crop((0, 0, half, image.height))).copy(),
            np.asarray(image.crop((half, 0, image.width, image.height))).copy(),
        ]

        record = {
            "rollout_id": rollout_id,
            "pair_key": f"{scene}__seed{seed}",
            "split": run_spec["split"],
            "scene": scene,
            "scene_sha256": scene_sha256,
            "seed": seed,
            "delay_blocks": delay,
            "arm": arm,
            "pre_command": run_spec["pre_command"],
            "post_command": run_spec["post_command"],
            "change_frame": change_frame,
            "admission_latent": int(admission_latent),
            "admission_frame": admission_frame,
            "video_path": str(video_path.relative_to(args.output)),
            "latent_path": str(latent_path.relative_to(args.output)),
            "decoded_u8_path": str(decoded_u8_path.relative_to(args.output)),
            "detector_frames_path": str(detector_frames_path.relative_to(args.output)),
            "status": "running",
            "run_index": run_index,
        }
        upsert_rollout(manifest, record)
        atomic_write_json(manifest_path, manifest)

        keyboard_pre, camera_pre = action_tensors("forward", "forward")
        keyboard_post, camera_post = action_tensors(
            run_spec["pre_command"],
            run_spec["post_command"],
        )

        def make_batch(keyboard, camera):
            return engine.build_inference_batch(
                images,
                PROMPT,
                [(keyboard, camera), (keyboard, camera)],
                num_frames=n_frames,
                num_conditional_frames=1,
            )

        engine.clear_cache()
        post_batch = preprocess(make_batch(keyboard_post, camera_post))

        def patched_get(data_batch, **kwargs):
            pre_fn = original_get(data_batch, **kwargs)
            post_fn = original_get(post_batch, **kwargs)

            def dispatcher(
                noisy_image_or_video,
                timestep,
                kv_cache=None,
                crossattn_cache=None,
                current_start=None,
                current_end=None,
                start_frame_for_rope=None,
            ):
                use_post = (
                    start_frame_for_rope is not None
                    and start_frame_for_rope >= admission_latent
                )
                prediction = post_fn if use_post else pre_fn
                return prediction(
                    noisy_image_or_video,
                    timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    start_frame_for_rope=start_frame_for_rope,
                )

            return dispatcher

        model.get_x0_fn_from_batch = patched_get
        started = time.perf_counter()
        try:
            batch = make_batch(keyboard_pre, camera_pre)
            video = engine.generate_from_batch(
                batch,
                guidance=float(model_config["guidance"]),
                seed=seed,
            )
        except Exception as error:
            record.update(status="failed", error=repr(error))
            upsert_rollout(manifest, record)
            atomic_write_json(manifest_path, manifest)
            raise
        finally:
            model.get_x0_fn_from_batch = original_get
        elapsed = time.perf_counter() - started

        output_dir.mkdir(parents=True, exist_ok=True)
        normalized = ((video + 1.0) / 2.0).clamp(0, 1)
        try:
            latent = getattr(engine, "last_sample_latent", None)
            if latent is None:
                raise RuntimeError(
                    "SAFESWM_DUMP_LATENTS was set but engine.last_sample_latent is missing"
                )

            with temporary_path(video_path, suffix=".mp4") as temporary_video:
                temporary_video.unlink()
                save_img_or_video(
                    normalized[0],
                    str(temporary_video.with_suffix("")),
                    fps=int(model_config["fps"]),
                )
                commit_temporary(temporary_video, video_path)

            decoded_u8 = (
                normalized[0]
                .mul(255)
                .floor()
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )
            if decoded_u8.shape[2] % 2:
                raise RuntimeError(
                    f"decoded side-by-side width is not even: {decoded_u8.shape}"
                )
            half_width = decoded_u8.shape[2] // 2
            detector_views = np.empty(
                (
                    2,
                    decoded_u8.shape[0],
                    int(config["detector"]["resize_height"]),
                    int(config["detector"]["resize_width"]),
                ),
                dtype=np.uint8,
            )
            for frame_index, frame in enumerate(decoded_u8):
                for view_index, view in enumerate(
                    (frame[:, :half_width], frame[:, half_width:])
                ):
                    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
                    detector_views[view_index, frame_index] = cv2.resize(
                        gray,
                        (
                            int(config["detector"]["resize_width"]),
                            int(config["detector"]["resize_height"]),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

            atomic_numpy_save(decoded_u8_path, decoded_u8, np)
            atomic_numpy_savez(
                detector_frames_path,
                np,
                views=detector_views,
            )
            atomic_torch_save(latent_path, latent, torch)
            artifacts = {
                "video": artifact_metadata(
                    video_path,
                    relative_to=args.output,
                    required=False,
                ),
                "latent": artifact_metadata(
                    latent_path,
                    relative_to=args.output,
                    shape=list(latent.shape),
                    dtype=str(latent.dtype),
                ),
                "decoded_u8": artifact_metadata(
                    decoded_u8_path,
                    relative_to=args.output,
                    shape=list(decoded_u8.shape),
                    dtype=str(decoded_u8.dtype),
                ),
                "detector_frames": artifact_metadata(
                    detector_frames_path,
                    relative_to=args.output,
                    shape=list(detector_views.shape),
                    dtype=str(detector_views.dtype),
                ),
            }
        except Exception as error:
            record.update(status="failed", error=f"artifact write failed: {error!r}")
            upsert_rollout(manifest, record)
            atomic_write_json(manifest_path, manifest)
            raise

        record.update(
            status="complete",
            total_s=elapsed,
            artifacts=artifacts,
            video_sha256=artifacts["video"]["sha256"],
            latent_sha256=artifacts["latent"]["sha256"],
            decoded_u8_sha256=artifacts["decoded_u8"]["sha256"],
            detector_frames_sha256=artifacts["detector_frames"]["sha256"],
            latent_shape=list(latent.shape),
            latent_dtype=str(latent.dtype),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        upsert_rollout(manifest, record)
        atomic_write_json(manifest_path, manifest)
        print(
            f"[v2] complete {run_index}/{len(specs)} {rollout_id} "
            f"admission={admission_frame} total={elapsed:.1f}s",
            flush=True,
        )

    print(f"[v2] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
