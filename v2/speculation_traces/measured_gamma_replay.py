"""Measured single-H200 Gamma systems replay for frozen trace decisions.

This is a systems-cost experiment. It serially generates every requested
candidate through the exact one-block delta path and separately labels worker
capacity estimates as simulations.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import (
    ROOT,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
)
from speculation_frontier import (
    BLOCK_FRAMES,
    PredictorSet,
    assert_no_action_latency_schema,
    construct_targets,
    fit_intent_thresholds,
    materialize_intent,
    simulate_readiness,
    _load_episodes,
)

RESEARCH = ROOT.parents[2]
RTWM = ROOT.parents[1]
SAFESWM = RESEARCH / "safeswm"
GAMMA_REPO = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
DRIVER_PATH = RTWM / "driver_v2.py"
SCENE = "buildTower_normal"
PROMPT = "Two Minecraft players exploring the world"
N_FRAMES = 189
PREFIX_BLOCKS = 3
LEAD_WINDOW_MS = 750.0
BUDGETS = (0, 1, 2, 4)
CAPACITIES = (1, 2, 4)
SEED = 1234


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_identity(selection_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"
    tokenizer = MODELS / "gamma-world" / "tokenizer.pth"
    identity: dict[str, Any] = {
        "schema_version": 1,
        "selection_sha256": sha256_file(selection_path),
        "sources": {
            "measured_gamma_replay.py": sha256_file(ROOT / "measured_gamma_replay.py"),
            "speculation_frontier.py": sha256_file(ROOT / "speculation_frontier.py"),
            "driver_v2.py": sha256_file(DRIVER_PATH),
            "gamma_source_commit": _git_commit(GAMMA_REPO),
        },
        "model_artifacts": {
            "checkpoint": {
                "label": "models/gamma-world/causal-few-step/model.safetensors",
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            "tokenizer": {
                "label": "models/gamma-world/tokenizer.pth",
                "sha256": sha256_file(tokenizer),
                "bytes": tokenizer.stat().st_size,
            },
            "text_encoder_label": "models/Cosmos-Reason1-7B",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "protocol": {
            "execution": "measured serial, one H200, one process",
            "scene": SCENE,
            "pixel_frames": N_FRAMES,
            "prefix_temporal_blocks": PREFIX_BLOCKS,
            "branch_pixel_frames": BLOCK_FRAMES,
            "lead_window_seconds": LEAD_WINDOW_MS / 1000.0,
            "budgets": list(BUDGETS),
            "simulated_worker_capacities": list(CAPACITIES),
            "seed": SEED,
        },
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    return identity


def _build_engine() -> Any:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    sys.path.insert(0, str(GAMMA_REPO))
    sys.path.insert(0, str(GAMMA_REPO / "scripts"))
    import torch
    from gamma_world._src.gamma_world.inference.inference_i2v import I2VInference
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from inference import format_hydra_value

    torch.set_grad_enabled(False)
    spec = MODEL_SPECS["causal_few_step"]
    options = [
        f"{key}={format_hydra_value(value)}"
        for key, value in spec.config_overrides.items()
    ]
    engine = I2VInference(
        experiment_name=spec.experiment,
        ckpt_path=str(
            MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"
        ),
        config_file=spec.config_file,
        guidance=5.0,
        shift=None,
        num_sampling_steps=spec.default_num_steps or 35,
        seed=1,
        context_parallel_size=1,
        experiment_opts=options,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = 16
    return engine


def _load_scene() -> list[np.ndarray]:
    from PIL import Image

    image = Image.open(GAMMA_REPO / "data" / SCENE / "first_frame.png")
    width = image.width // 2
    return [
        np.asarray(image.crop((0, 0, width, image.height))),
        np.asarray(image.crop((width, 0, image.width, image.height))),
    ]


def _action_stream(
    previous: tuple[np.ndarray, np.ndarray],
    branch: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    keyboard = np.repeat(previous[0][:1], N_FRAMES, axis=0).astype(np.float32)
    camera = np.repeat(previous[1][:1], N_FRAMES, axis=0).astype(np.float32)
    start = PREFIX_BLOCKS * BLOCK_FRAMES
    keyboard[start : start + BLOCK_FRAMES] = branch[0]
    camera[start : start + BLOCK_FRAMES] = branch[1]
    keyboard[start + BLOCK_FRAMES :] = branch[0][-1]
    camera[start + BLOCK_FRAMES :] = 0.0
    return keyboard, camera


def _make_x0(
    engine: Any,
    images: Sequence[np.ndarray],
    keyboard: np.ndarray,
    camera: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    import torch
    from gamma_world._src.gamma_world.inference.inference_i2v import (
        IS_PREPROCESSED_KEY,
        to_with_skip_tensor,
    )

    keyboard_tensor = torch.tensor(keyboard, dtype=torch.float32).unsqueeze(0)
    camera_tensor = torch.tensor(camera, dtype=torch.float32).unsqueeze(0)
    actions = [
        (keyboard_tensor, camera_tensor),
        (keyboard_tensor.clone(), camera_tensor.clone()),
    ]
    batch = engine.build_inference_batch(
        list(images),
        PROMPT,
        actions,
        num_frames=N_FRAMES,
        num_conditional_frames=1,
    )
    model = engine.model
    batch["video"] = batch["video"].float()
    if not batch.get(IS_PREPROCESSED_KEY, False):
        batch["video"] = batch["video"] / 127.5 - 1.0
    batch["video"] = torch.clamp(batch["video"], -1, 1)
    batch[IS_PREPROCESSED_KEY] = True
    batch = to_with_skip_tensor(batch, **model.tensor_kwargs)
    engine.inplace_compute_text_embeddings_online(batch, use_negative_prompt=True)
    batch = model.get_data_batch_with_latent_view_indices(batch)
    model._normalize_video_databatch_inplace(batch)
    x0 = model.get_x0_fn_from_batch(
        batch,
        n_views=2,
        guidance=5.0,
        is_negative_prompt=True,
    )
    return batch, x0


def _reseed() -> None:
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _timed(operation: Any) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def _run_decision(
    engine: Any,
    images: Sequence[np.ndarray],
    decision: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    sys.path.insert(0, str(RTWM))
    from driver_v2 import BlockwiseSession, snapshot_nbytes

    previous = materialize_intent(decision["previous_intent"], thresholds)
    actual = (
        np.asarray(decision["actual_keyboard"], dtype=np.float32),
        np.asarray(decision["actual_camera_degrees"], dtype=np.float32),
    )
    intent_actions = {
        token: materialize_intent(token, thresholds)
        for token in decision["candidate_intents_ranked"]
    }
    streams = {
        "prefix": _action_stream(previous, previous),
        "actual": _action_stream(previous, actual),
    }
    streams.update(
        {
            f"candidate:{token}": _action_stream(previous, action)
            for token, action in intent_actions.items()
        }
    )
    prepared = {
        name: _make_x0(engine, images, *stream)
        for name, stream in streams.items()
    }
    batch, prefix_x0 = prepared["prefix"]
    session = BlockwiseSession(engine, batch, seed=1)
    if session.nfpb * 4 != BLOCK_FRAMES:
        raise RuntimeError("Gamma latent block no longer maps to 12 pixel frames")
    prefix_measurements: list[float] = []
    for _ in range(PREFIX_BLOCKS):
        _reseed()
        _, elapsed = _timed(lambda: session.step_block(prefix_x0))
        prefix_measurements.append(elapsed)

    scenarios: list[dict[str, Any]] = []
    for budget in BUDGETS:
        torch.cuda.synchronize()
        start_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        snapshot, fork_ms = _timed(session.fork_delta)
        fork_bytes = snapshot_nbytes(snapshot["kv"])
        candidates = (
            ["__actual_baseline__"]
            if budget == 0
            else list(decision["candidate_intents_ranked"][:budget])
        )
        records: list[dict[str, Any]] = []
        captures: list[Any] = []
        capture: Any | None = None
        accepted_capture: Any | None = None
        restore_ms: list[float] = []
        serial_elapsed = fork_ms
        for candidate_index, token in enumerate(candidates):
            if candidate_index > 0:
                _, elapsed = _timed(lambda: session.restore_delta(snapshot))
                restore_ms.append(elapsed)
                serial_elapsed += elapsed
            x0 = (
                prepared["actual"][1]
                if token == "__actual_baseline__"
                else prepared[f"candidate:{token}"][1]
            )
            _reseed()
            result, generation_ms = _timed(lambda: session.step_block(x0))
            if budget == 0:
                capture_ms = 0.0
                capture_bytes = 0
            else:
                capture, capture_ms = _timed(session.capture_branch)
                captures.append(capture)
                capture_bytes = snapshot_nbytes(capture["kv"]) + (
                    capture["out_block"].numel()
                    * capture["out_block"].element_size()
                )
            serial_elapsed += generation_ms + capture_ms
            records.append(
                {
                    "candidate_index": candidate_index,
                    "kind": (
                        "on_demand_baseline"
                        if budget == 0
                        else "speculative_candidate"
                    ),
                    "intent": (
                        decision["actual_intent"]
                        if token == "__actual_baseline__"
                        else token
                    ),
                    "generation_ms": generation_ms,
                    "denoise_ms": result.denoise_ms,
                    "commit_ms": result.commit_ms,
                    "capture_ms": capture_ms,
                    "capture_bytes": capture_bytes,
                    "charged": True,
                }
            )

        hit_index = (
            candidates.index(decision["actual_intent"])
            if budget > 0 and decision["actual_intent"] in candidates
            else None
        )
        fallback_generated = False
        if budget > 0 and hit_index is None:
            _, elapsed = _timed(lambda: session.restore_delta(snapshot))
            restore_ms.append(elapsed)
            serial_elapsed += elapsed
            _reseed()
            result, generation_ms = _timed(
                lambda: session.step_block(prepared["actual"][1])
            )
            capture, capture_ms = _timed(session.capture_branch)
            serial_elapsed += generation_ms + capture_ms
            fallback_generated = True
            captures.append(capture)
            records.append(
                {
                    "candidate_index": len(candidates),
                    "kind": "miss_fallback_actual",
                    "intent": decision["actual_intent"],
                    "generation_ms": generation_ms,
                    "denoise_ms": result.denoise_ms,
                    "commit_ms": result.commit_ms,
                    "capture_ms": capture_ms,
                    "capture_bytes": snapshot_nbytes(capture["kv"])
                    + capture["out_block"].numel()
                    * capture["out_block"].element_size(),
                    "charged": True,
                }
            )
            accepted_capture = capture
        elif budget > 0:
            accepted_capture = captures[int(hit_index)]

        if accepted_capture is None:
            accept_ms = 0.0
        else:
            _, accept_ms = _timed(lambda: session.apply_branch(accepted_capture))
            serial_elapsed += accept_ms
        peak_allocated = torch.cuda.max_memory_allocated()
        _, reset_restore_ms = _timed(lambda: session.restore_delta(snapshot))
        scenario = {
            "category": decision["category"],
            "episode_id": decision["episode_id"],
            "block_index": decision["block_index"],
            "budget": budget,
            "measured_mode": "serial_one_h200",
            "lead_window_ms": LEAD_WINDOW_MS,
            "prefix_generation_ms": prefix_measurements,
            "fork_ms": fork_ms,
            "fork_snapshot_bytes": fork_bytes,
            "restore_ms": restore_ms,
            "accept_ms": accept_ms,
            "benchmark_reset_restore_ms_excluded": reset_restore_ms,
            "candidate_records": records,
            "generated_speculative_candidates": 0 if budget == 0 else budget,
            "generated_fallback_candidates": int(fallback_generated),
            "hit_count": int(budget > 0 and hit_index is not None),
            "miss_count": int(budget > 0 and hit_index is None),
            "hit_rank": None if hit_index is None else hit_index + 1,
            "ready_by_lead_window": bool(
                budget > 0
                and hit_index is not None
                and sum(
                    record["generation_ms"] + record["capture_ms"]
                    for record in records[: hit_index + 1]
                )
                + fork_ms
                + sum(restore_ms[:hit_index])
                <= LEAD_WINDOW_MS
            ),
            "total_gpu_work_ms": serial_elapsed,
            "start_allocated_bytes": start_allocated,
            "peak_allocated_bytes": peak_allocated,
            "incremental_peak_allocated_bytes": max(
                0, peak_allocated - start_allocated
            ),
            "total_retained_snapshot_bytes": fork_bytes
            + sum(record["capture_bytes"] for record in records),
            "all_generated_work_charged": all(
                record["charged"] for record in records
            ),
        }
        scenarios.append(scenario)
        del captures, capture, accepted_capture, result, snapshot
        torch.cuda.empty_cache()
    del prepared, session
    torch.cuda.empty_cache()
    return scenarios


def _trace_projection(
    scenarios: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    block_manifest = load_json(ROOT / "manifests" / "blocks.json")
    episodes = _load_episodes(block_manifest)
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    train_targets = construct_targets(train, thresholds)
    test_targets = construct_targets(test, thresholds)
    model = PredictorSet(train_targets, "intent")
    measured = [
        float(record["generation_ms"]) + float(record["capture_ms"])
        for scenario in scenarios
        if int(scenario["budget"]) > 0
        for record in scenario["candidate_records"]
        if record["kind"] == "speculative_candidate"
    ]
    measured_candidate_ms = float(np.median(np.asarray(measured)))
    fork_ms = float(
        np.median(
            np.asarray(
                [
                    scenario["fork_ms"]
                    for scenario in scenarios
                    if int(scenario["budget"]) > 0
                ]
            )
        )
    )
    capture_bytes = int(
        np.median(
            np.asarray(
                [
                    record["capture_bytes"]
                    for scenario in scenarios
                    if int(scenario["budget"]) > 0
                    for record in scenario["candidate_records"]
                    if record["kind"] == "speculative_candidate"
                ]
            )
        )
    )
    fork_bytes = max(int(scenario["fork_snapshot_bytes"]) for scenario in scenarios)
    rows: list[dict[str, Any]] = []
    for budget in (1, 2, 4):
        for capacity in CAPACITIES:
            completion = simulate_readiness(
                [measured_candidate_ms] * budget,
                capacity=capacity,
                lead_window_ms=LEAD_WINDOW_MS,
            )
            for subset in ("all", "intent_change"):
                eligible = 0
                hits = 0
                ready = 0
                for episode in test_targets:
                    for index in range(1, len(episode.intent)):
                        changed = episode.intent[index] != episode.intent[index - 1]
                        if subset == "intent_change" and not changed:
                            continue
                        eligible += 1
                        candidates = model.predict(
                            "markov_history_3", episode.intent[:index], budget
                        )
                        if episode.intent[index] in candidates:
                            hits += 1
                            rank = candidates.index(episode.intent[index])
                            ready += int(completion[rank] + fork_ms <= LEAD_WINDOW_MS)
                rows.append(
                    {
                        "projection": "simulation_from_measured_serial_branch_times",
                        "target": "intent",
                        "predictor": "markov_history_3",
                        "subset": subset,
                        "budget": budget,
                        "worker_capacity": capacity,
                        "lead_window_ms": LEAD_WINDOW_MS,
                        "eligible_block_count": eligible,
                        "hit_count": hits,
                        "miss_count": eligible - hits,
                        "hit_rate": hits / eligible,
                        "miss_rate": 1.0 - hits / eligible,
                        "readiness_count": ready,
                        "readiness_rate": ready / eligible,
                        "compute_candidates": eligible * budget,
                        "compute_ms": eligible * budget * measured_candidate_ms,
                        "extra_state_bytes": fork_bytes + budget * capture_bytes,
                        "measured_candidate_time_estimator": "median",
                        "measured_candidate_ms": measured_candidate_ms,
                    }
                )
    assert_no_action_latency_schema(rows)
    return rows


def run_replay(
    *,
    selection_path: Path,
    output_path: Path,
    resume: bool,
    max_decisions: int | None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for measured Gamma replay")
    identity = _source_identity(selection_path)
    selection = load_json(selection_path)
    decisions = selection["decisions"]
    if max_decisions is not None:
        decisions = decisions[:max_decisions]
    existing: dict[str, Any] | None = None
    if output_path.exists():
        if not resume:
            raise FileExistsError("replay artifact exists; pass --resume")
        existing = load_json(output_path)
        if existing["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("strict resume identity mismatch")
    completed = {
        (row["category"], int(row["budget"]))
        for row in (existing or {}).get("scenarios", [])
    }
    report: dict[str, Any] = existing or {
        "schema_version": 1,
        "identity_sha256": identity["identity_sha256"],
        "identity": identity,
        "selection_sha256": selection["selection_sha256"],
        "claim_scope": (
            "systems cost and readiness only; no semantic visual-response metric"
        ),
        "timing_model": {
            "primary": (
                "measured serial execution on one H200; every generated "
                "speculative and miss-fallback branch is charged"
            ),
            "projection": (
                "worker capacities 1,2,4 are simulations using measured serial "
                "branch times and a fixed 0.75 second action lead window"
            ),
        },
        "delta_path": (
            "BlockwiseSession.fork_delta, step_block, capture_branch, "
            "restore_delta, apply_branch; one generated temporal block per branch"
        ),
        "canonical_actions": (
            "direct float32 tensors from frozen 23-key arrays and canonical camera "
            "degrees; intent candidates use train-fit representatives"
        ),
        "controlled_allocator": {
            "status": "not_used_in_gamma_replay",
            "note": (
                "paged copy-on-write allocator measurements are separate controlled "
                "results and are not substituted into this replay"
            ),
        },
        "scenarios": [],
        "trace_projections": [],
        "status": "running",
    }
    engine = _build_engine()
    images = _load_scene()
    thresholds = load_json(ROOT / "results" / "intent_fit.json")
    started = time.perf_counter()
    for decision in decisions:
        missing = [
            budget
            for budget in BUDGETS
            if (decision["category"], budget) not in completed
        ]
        if not missing:
            continue
        # A decision is run as an atomic four-budget unit so all budgets share
        # the identical live prefix. Partial decision rows are never resumed.
        if len(missing) != len(BUDGETS):
            raise RuntimeError("strict resume found a partial replay decision")
        rows = _run_decision(engine, images, decision, thresholds)
        report["scenarios"].extend(rows)
        report["measured_gpu_runtime_ms"] = sum(
            row["total_gpu_work_ms"] for row in report["scenarios"]
        )
        report["wall_runtime_seconds_current_process"] = time.perf_counter() - started
        atomic_write_json(output_path, report)
    report["trace_projections"] = _trace_projection(report["scenarios"], thresholds)
    report["measured_gpu_runtime_ms"] = sum(
        row["total_gpu_work_ms"] for row in report["scenarios"]
    )
    report["status"] = (
        "complete" if len(decisions) == len(selection["decisions"]) else "bounded_subset"
    )
    report["limitations"] = [
        "single GPU and one fixed Gamma scene",
        "two hash-frozen decisions, one stable/common and one intent change",
        "intent branches are train-derived representatives rather than exact future test blocks",
        "worker capacity results are projections, not concurrent measurements",
        "systems replay does not measure semantic visual response",
    ]
    assert_no_action_latency_schema(report)
    atomic_write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "results" / "replay_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "gamma_replay.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-decisions", type=int)
    args = parser.parse_args()
    report = run_replay(
        selection_path=args.selection,
        output_path=args.output,
        resume=args.resume,
        max_decisions=args.max_decisions,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scenario_count": len(report["scenarios"]),
                "measured_gpu_runtime_ms": report["measured_gpu_runtime_ms"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
