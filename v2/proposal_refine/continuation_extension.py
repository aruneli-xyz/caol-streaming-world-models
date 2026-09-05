"""Run the steady-roll eight-block exactness gate on a 201-frame horizon.

The stock 189-frame input contains only 16 complete temporal blocks, so a
steady-roll parent at block 9 cannot have eight successors.  This isolated
extension changes only the allocation horizon; schedule, model, transaction,
actions, and exact comparisons are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

if __package__:
    from . import gpu_experiment as experiment
    from .conditioning import make_shallow_conditioning_x0
else:
    import gpu_experiment as experiment
    from conditioning import make_shallow_conditioning_x0


PIXEL_FRAMES = 201
PHASE = "steady_roll"
PHASE_BLOCK = 9


def collect(output: Path, *, resume: bool) -> dict:
    experiment.N_FRAMES = PIXEL_FRAMES
    config = json.loads(experiment.CONFIG.read_text())
    identity = experiment.source_identity()
    identity["continuation_extension_source_sha256"] = experiment.sha256_file(
        Path(__file__)
    )
    identity["protocol_override"] = {
        "reason": (
            "189 frames provide 16 complete blocks; block 9 plus eight "
            "continuations requires at least 17 complete blocks"
        ),
        "pixel_frames": PIXEL_FRAMES,
        "phase": PHASE,
        "parent_block_index": PHASE_BLOCK,
        "only_allocation_horizon_changed": True,
    }
    identity["identity_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(identity)
    ).hexdigest()

    existing = None
    if output.exists():
        if not resume:
            raise FileExistsError("output exists; pass --resume")
        existing = json.loads(output.read_text())
        if existing["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("strict resume identity mismatch")
    report = existing or {
        "schema_version": "rtwm-v2-proposal-continuation-extension-1",
        "status": "running",
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "gates": [],
    }
    completed = {row["action"] for row in report["gates"]}

    engine = experiment.build_engine(config)
    required_latent_frames = (PIXEL_FRAMES - 1) // 4 + 1
    available_normalization_frames = int(
        engine.model.tokenizer.model.video_mean.shape[2]
    )
    if required_latent_frames > available_normalization_frames:
        report.update(
            {
                "status": "blocked_by_stock_tokenizer_horizon",
                "all_exact": False,
                "stock_horizon": {
                    "required_pixel_frames": PIXEL_FRAMES,
                    "required_latent_frames": required_latent_frames,
                    "available_normalization_frames": (
                        available_normalization_frames
                    ),
                    "unsafe_patch_applied": False,
                    "reason": (
                        "extending or extrapolating checkpoint video_mean/video_std "
                        "would no longer be an exact stock-model comparison"
                    ),
                },
            }
        )
        experiment.atomic_json(output, report)
        return report
    images = experiment.load_images()
    prefix_keyboard, prefix_camera = experiment.action_arrays("unchanged", 0)
    prefix_batch, prefix_stock_x0, prefix_tensors, _ = experiment.make_x0(
        engine, images, prefix_keyboard, prefix_camera
    )
    prefix_x0, _ = make_shallow_conditioning_x0(prefix_stock_x0)
    source_hashes = identity["sources"]
    started = time.perf_counter()
    for action in config["correctness"]["actions"]:
        if action in completed:
            continue
        keyboard, camera = experiment.action_arrays(action, PHASE_BLOCK)
        _, stock_x0, tensors, _ = experiment.make_x0(
            engine, images, keyboard, camera
        )
        action_x0, _ = make_shallow_conditioning_x0(stock_x0)
        gate = experiment.exact_chain_gate(
            engine,
            prefix_batch,
            prefix_x0,
            prefix_tensors,
            action_x0,
            tensors,
            PHASE_BLOCK,
            PHASE,
            action,
            source_hashes,
            int(config["model"]["seed"]),
        )
        report["gates"].append(gate)
        report["wall_runtime_seconds_current_process"] = (
            time.perf_counter() - started
        )
        experiment.atomic_json(output, report)

    report["all_exact"] = all(
        row["all_exact"]
        and any(
            item["continuation_blocks"] == 8 and item["exact"]
            for item in row["continuation"]
        )
        for row in report["gates"]
    )
    report["status"] = "complete"
    report["wall_runtime_seconds_current_process"] = time.perf_counter() - started
    experiment.atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "continuation_extension.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(args.output, resume=args.resume)
    print(
        json.dumps(
            {
                "status": report["status"],
                "all_exact": report["all_exact"],
                "gate_count": len(report["gates"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
