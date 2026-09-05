"""Measure Gamma-World's raw decoder temporal receptive field.

This probe consumes the matched pre-decode latents produced by
``matched_counterfactual.py``.  It deliberately compares raw uint8 decoder
outputs before any MP4 encoding.

Run from the Gamma-World checkout with its virtual environment:

    .venv/bin/python /path/to/rtwm/v2/measure_decoder_rf.py \
        --scene buildTower_normal --seed 1 --delay 0 \
        --hybrid-indices differing --allow-dense

The dense command decodes one hybrid clip per differing temporal latent index
and is intentionally opt-in.  A one-decode plumbing check is available with
``--smoke``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


V2 = Path(__file__).resolve().parent
RTWM = V2.parent
SAFESWM = RTWM.parent / "safeswm"
GAMMA_REPO = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
DEFAULT_CONFIG = V2 / "config" / "pilot.json"
DEFAULT_MATCHED = V2 / "results" / "matched"
DEFAULT_OUTPUT = V2 / "results" / "decoder_rf"
DEFAULT_TOKENIZER = MODELS / "gamma-world" / "tokenizer.pth"

MANIFEST_SCHEMA = "rtwm-v2-decoder-rf-manifest-1"
SMOKE_SCHEMA = "rtwm-v2-decoder-rf-smoke-1"
DETERMINISM_SCHEMA = "rtwm-v2-decoder-rf-determinism-1"
FULL_SCHEMA = "rtwm-v2-decoder-rf-full-1"
HYBRID_SCHEMA = "rtwm-v2-decoder-rf-hybrid-1"
MP4_SCHEMA = "rtwm-v2-decoder-rf-mp4-diagnostic-1"
METRICS_SCHEMA = "rtwm-v2-decoder-rf-frame-metrics-1"


def expected_first_pixel_frame(latent_index: int, temporal_stride: int = 4) -> int:
    """Return the first pixel frame nominally introduced by a latent index.

    Wan's frame-count map is ``P(T) = 1 + (T - 1) * stride``.  Therefore
    latent zero owns frame zero, while latent ``t > 0`` first owns
    ``stride*t - (stride - 1)`` (``4*t - 3`` for Gamma-World).
    """

    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    if latent_index == 0:
        return 0
    return temporal_stride * latent_index - (temporal_stride - 1)


def expected_native_pixel_support(
    latent_index: int,
    total_pixel_frames: int | None = None,
    temporal_stride: int = 4,
) -> dict[str, int]:
    """Return the inclusive nominal output window owned by one latent."""

    start = expected_first_pixel_frame(latent_index, temporal_stride)
    end = 0 if latent_index == 0 else temporal_stride * latent_index
    if total_pixel_frames is not None:
        if total_pixel_frames < 1:
            raise ValueError("total_pixel_frames must be positive")
        end = min(end, total_pixel_frames - 1)
    return {"start": start, "end": end, "count": max(0, end - start + 1)}


def contiguous_spans(indices: Iterable[int]) -> list[list[int]]:
    """Collapse integer indices into inclusive ``[start, end]`` spans."""

    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    spans: list[list[int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            spans.append([start, previous])
            start = index
        previous = index
    spans.append([start, previous])
    return spans


def summarize_temporal_support(
    changed_by_frame: Sequence[bool],
    *,
    latent_index: int | None = None,
    temporal_stride: int = 4,
) -> dict[str, Any]:
    """Summarize changed frames and decoder look-ahead/tail bookkeeping."""

    changed = [index for index, value in enumerate(changed_by_frame) if bool(value)]
    result: dict[str, Any] = {
        "frame_count": len(changed_by_frame),
        "changed_frame_count": len(changed),
        "changed_frame_spans": contiguous_spans(changed),
        "earliest_changed_frame": changed[0] if changed else None,
        "latest_changed_frame": changed[-1] if changed else None,
    }
    if latent_index is None:
        return result

    expected = expected_native_pixel_support(
        latent_index,
        total_pixel_frames=len(changed_by_frame),
        temporal_stride=temporal_stride,
    )
    early = [frame for frame in changed if frame < expected["start"]]
    late = [frame for frame in changed if frame > expected["end"]]
    result.update(
        {
            "latent_index": latent_index,
            "expected_native_support": expected,
            "expected_first_pixel_formula": (
                f"{temporal_stride}*t-{temporal_stride - 1}"
                if temporal_stride > 1
                else "t"
            ),
            "lookahead_frames": (
                max(0, expected["start"] - changed[0]) if changed else None
            ),
            "early_changed_frame_count": len(early),
            "early_changed_frame_spans": contiguous_spans(early),
            "late_changed_frame_count": len(late),
            "late_changed_frame_spans": contiguous_spans(late),
        }
    )
    return result


def parse_index_spec(
    specification: str,
    *,
    latent_count: int,
    differing_indices: Sequence[int],
) -> list[int]:
    """Parse ``none``, ``all``, ``differing``, or comma/range index specs."""

    if latent_count < 1:
        raise ValueError("latent_count must be positive")
    normalized = specification.strip().lower()
    if normalized in {"", "none"}:
        return []
    if normalized == "all":
        return list(range(latent_count))
    if normalized == "differing":
        return sorted(set(int(index) for index in differing_indices))
    if normalized == "first-differing":
        return [min(differing_indices)] if differing_indices else []

    selected: set[int] = set()
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty token in hybrid index spec {specification!r}")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending hybrid range {token!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    invalid = sorted(index for index in selected if not 0 <= index < latent_count)
    if invalid:
        raise ValueError(
            f"hybrid indices outside [0, {latent_count}): {invalid}"
        )
    return sorted(selected)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    """Hash dtype, shape, and C-order bytes of a NumPy array."""

    import numpy as np

    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def configure_determinism(torch_module: Any) -> dict[str, Any]:
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms_warn_only": True,
    }


def load_tensor(torch_module: Any, path: Path) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def split_views(latent: Any, n_views: int) -> Any:
    """Convert ``[B,C,V*T,H,W]`` to ``[B,V,C,T,H,W]``."""

    if latent.ndim != 5:
        raise ValueError(f"expected a 5D latent, got shape {list(latent.shape)}")
    if n_views < 1 or latent.shape[2] % n_views:
        raise ValueError(
            f"combined latent time {latent.shape[2]} is not divisible by "
            f"n_views={n_views}"
        )
    batch, channels, combined_time, height, width = latent.shape
    temporal = combined_time // n_views
    return (
        latent.reshape(batch, channels, n_views, temporal, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .contiguous()
    )


def combine_views(view_latent: Any) -> Any:
    """Convert ``[B,V,C,T,H,W]`` to ``[B,C,V*T,H,W]``."""

    if view_latent.ndim != 6:
        raise ValueError(
            f"expected a 6D view latent, got shape {list(view_latent.shape)}"
        )
    batch, views, channels, temporal, height, width = view_latent.shape
    return (
        view_latent.permute(0, 2, 1, 3, 4, 5)
        .contiguous()
        .reshape(batch, channels, views * temporal, height, width)
    )


def latent_difference_summary(
    control_views: Any,
    stop_views: Any,
) -> dict[str, Any]:
    import torch

    if tuple(control_views.shape) != tuple(stop_views.shape):
        raise ValueError("control and STOP latent shapes differ")
    difference = (control_views.float() - stop_views.float()).abs()
    per_view_time = difference.amax(dim=(0, 2, 4, 5))
    views: list[dict[str, Any]] = []
    differing_union: set[int] = set()
    for view in range(per_view_time.shape[0]):
        indices = torch.nonzero(per_view_time[view] > 0).flatten().tolist()
        differing_union.update(int(index) for index in indices)
        views.append(
            {
                "view_index": view,
                "differing_indices": [int(index) for index in indices],
                "first_differing_index": int(indices[0]) if indices else None,
                "last_differing_index": int(indices[-1]) if indices else None,
                "maximum_absolute_error": float(per_view_time[view].max()),
            }
        )
    return {
        "shape_bvcthw": list(control_views.shape),
        "differing_indices_union": sorted(differing_union),
        "views": views,
    }


class GammaWorldDecoder:
    """Thin adapter around Gamma-World's actual Wan2.1 tokenizer decoder."""

    def __init__(self, tokenizer_path: Path, n_views: int, torch_module: Any):
        from gamma_world._src.predict2.tokenizers.wan2pt1 import (
            Wan2pt1VAEInterface,
        )

        self.torch = torch_module
        self.n_views = n_views
        self.tokenizer = Wan2pt1VAEInterface(vae_pth=str(tokenizer_path))

    def decode_uint8(self, combined_latent: Any) -> Any:
        """Decode to ``[V,T,H,W,C]`` uint8, before video encoding."""

        torch = self.torch
        view_latent = split_views(combined_latent, self.n_views)
        batch, views, channels, temporal, height, width = view_latent.shape
        flat = view_latent.reshape(
            batch * views, channels, temporal, height, width
        ).to("cuda")
        with torch.inference_mode():
            decoded = self.tokenizer.decode(flat)
            if decoded.ndim != 5 or decoded.shape[1] != 3:
                raise RuntimeError(
                    f"unexpected tokenizer output shape {list(decoded.shape)}"
                )
            raw = (
                ((decoded.float() + 1.0) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
            )
        if batch != 1:
            raise ValueError(f"only batch size one is supported, got {batch}")
        return (
            raw.reshape(batch, views, 3, raw.shape[2], raw.shape[3], raw.shape[4])
            .permute(0, 1, 3, 4, 5, 2)
            .contiguous()[0]
            .numpy()
        )


def frame_metrics(first: Any, second: Any) -> dict[str, Any]:
    """Compute bounded-memory per-view/per-frame metrics on uint8 arrays."""

    import numpy as np

    first = np.asarray(first)
    second = np.asarray(second)
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise TypeError("frame metrics require uint8 inputs")
    if first.shape != second.shape or first.ndim != 5:
        raise ValueError(
            "frame metrics require equal [V,T,H,W,C] arrays; "
            f"got {first.shape} and {second.shape}"
        )
    views, frames = first.shape[:2]
    maximum = np.zeros((views, frames), dtype=np.uint8)
    mean = np.zeros((views, frames), dtype=np.float64)
    changed_values = np.zeros((views, frames), dtype=np.uint64)
    changed_pixels = np.zeros((views, frames), dtype=np.uint64)
    sum_absolute = np.zeros((views, frames), dtype=np.uint64)
    for view in range(views):
        for frame in range(frames):
            difference = np.abs(
                first[view, frame].astype(np.int16)
                - second[view, frame].astype(np.int16)
            )
            maximum[view, frame] = difference.max()
            mean[view, frame] = difference.mean(dtype=np.float64)
            changed_values[view, frame] = np.count_nonzero(difference)
            changed_pixels[view, frame] = np.count_nonzero(
                np.any(difference != 0, axis=-1)
            )
            sum_absolute[view, frame] = difference.sum(dtype=np.uint64)
    return {
        "max_abs": maximum,
        "mean_abs": mean,
        "changed_values": changed_values,
        "changed_pixels": changed_pixels,
        "sum_abs": sum_absolute,
    }


def save_metrics(path: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **metrics)
    temporary.replace(path)
    return {
        "schema_version": METRICS_SCHEMA,
        "path": str(Path(path.parent.name) / path.name),
        "sha256": sha256_file(path),
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in metrics.items()
        },
    }


def comparison_payload(
    *,
    first: Any,
    second: Any,
    first_name: str,
    second_name: str,
    metrics_path: Path,
    latent_indices_by_view: Sequence[int | None] | None = None,
    temporal_stride: int = 4,
) -> dict[str, Any]:
    metrics = frame_metrics(first, second)
    artifact = save_metrics(metrics_path, metrics)
    view_summaries: list[dict[str, Any]] = []
    for view in range(metrics["max_abs"].shape[0]):
        latent_index = (
            latent_indices_by_view[view]
            if latent_indices_by_view is not None
            else None
        )
        summary = summarize_temporal_support(
            metrics["max_abs"][view] > 0,
            latent_index=latent_index,
            temporal_stride=temporal_stride,
        )
        summary.update(
            {
                "view_index": view,
                "maximum_absolute_error": int(metrics["max_abs"][view].max()),
                "total_changed_pixels": int(
                    metrics["changed_pixels"][view].sum()
                ),
                "total_changed_values": int(
                    metrics["changed_values"][view].sum()
                ),
                "total_absolute_error": int(metrics["sum_abs"][view].sum()),
            }
        )
        view_summaries.append(summary)
    return {
        "comparison_domain": "raw_uint8_pre_mp4",
        "first": first_name,
        "second": second_name,
        "first_raw_sha256": sha256_array(first),
        "second_raw_sha256": sha256_array(second),
        "raw_shape_vthwc": list(first.shape),
        "raw_dtype": "uint8",
        "exact": bool(all(row["changed_frame_count"] == 0 for row in view_summaries)),
        "frame_metrics": artifact,
        "views": view_summaries,
    }


def save_raw_decode(path: Path, decoded: Any) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, decoded, allow_pickle=False)
    temporary.replace(path)
    return {
        "path": path.name,
        "file_sha256": sha256_file(path),
        "array_sha256": sha256_array(decoded),
        "dtype": str(decoded.dtype),
        "shape_vthwc": list(decoded.shape),
    }


def read_mp4_views(path: Path, n_views: int) -> Any:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    video = np.stack(frames)
    if video.shape[2] % n_views:
        raise ValueError(
            f"MP4 width {video.shape[2]} is not divisible by n_views={n_views}"
        )
    return np.stack(np.split(video, n_views, axis=2), axis=0)


def mp4_diagnostic(
    *,
    run_dir: Path,
    matched_root: Path,
    control_record: dict[str, Any],
    stop_record: dict[str, Any],
    control_raw: Any,
    stop_raw: Any,
) -> dict[str, Any]:
    control_path = matched_root / control_record["video_path"]
    stop_path = matched_root / stop_record["video_path"]
    control_mp4 = read_mp4_views(control_path, control_raw.shape[0])
    stop_mp4 = read_mp4_views(stop_path, stop_raw.shape[0])
    if control_mp4.shape != control_raw.shape or stop_mp4.shape != stop_raw.shape:
        return {
            "schema_version": MP4_SCHEMA,
            "status": "shape_mismatch",
            "raw_shapes": {
                "control": list(control_raw.shape),
                "stop": list(stop_raw.shape),
            },
            "mp4_shapes": {
                "control": list(control_mp4.shape),
                "stop": list(stop_mp4.shape),
            },
            "note": (
                "The saved latents are FP16 copies of the original sampling "
                "output, so this diagnostic combines latent downcast and MP4 "
                "encoding effects."
            ),
        }

    raw_pair = comparison_payload(
        first=control_raw,
        second=stop_raw,
        first_name="control_raw",
        second_name="stop_raw",
        metrics_path=run_dir / "metrics" / "mp4_diag_raw_pair.npz",
    )
    mp4_pair = comparison_payload(
        first=control_mp4,
        second=stop_mp4,
        first_name="control_mp4_decode",
        second_name="stop_mp4_decode",
        metrics_path=run_dir / "metrics" / "mp4_diag_encoded_pair.npz",
    )
    return {
        "schema_version": MP4_SCHEMA,
        "status": "ok",
        "note": (
            "Raw-to-MP4 differences include both the saved FP16 latent "
            "downcast (generation decoded the unsaved sampler tensor) and "
            "lossy MP4 encoding; they are not a codec-only estimate."
        ),
        "control_raw_vs_mp4": comparison_payload(
            first=control_raw,
            second=control_mp4,
            first_name="control_raw",
            second_name="control_mp4_decode",
            metrics_path=run_dir / "metrics" / "mp4_diag_control.npz",
        ),
        "stop_raw_vs_mp4": comparison_payload(
            first=stop_raw,
            second=stop_mp4,
            first_name="stop_raw",
            second_name="stop_mp4_decode",
            metrics_path=run_dir / "metrics" / "mp4_diag_stop.npz",
        ),
        "raw_stop_control": raw_pair,
        "mp4_stop_control": mp4_pair,
        "earliest_pair_difference_shift_by_view": [
            (
                None
                if raw_view["earliest_changed_frame"] is None
                or mp4_view["earliest_changed_frame"] is None
                else mp4_view["earliest_changed_frame"]
                - raw_view["earliest_changed_frame"]
            )
            for raw_view, mp4_view in zip(
                raw_pair["views"], mp4_pair["views"]
            )
        ],
    }


def find_pair(
    manifest: dict[str, Any],
    scene: str | None,
    seed: int,
    delay: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete = [
        record
        for record in manifest["rollouts"]
        if record.get("status") == "complete"
    ]
    scenes = sorted({record["scene"] for record in complete})
    if scene is None:
        if not scenes:
            raise RuntimeError("matched manifest has no complete scenes")
        scene = scenes[0]
    by_key = {
        (record["scene"], int(record["seed"]), record["arm"]): record
        for record in complete
    }
    control = by_key.get((scene, seed, "control"))
    stop = by_key.get((scene, seed, f"stop_d{delay}"))
    if control is None or stop is None:
        raise RuntimeError(
            f"missing complete matched pair for {scene}, seed={seed}, "
            f"delay={delay}"
        )
    return control, stop


def source_record(
    matched_root: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    latent_relative = record.get("latent_path")
    if not latent_relative:
        raise RuntimeError(
            f"{record['rollout_id']} has no persisted latent; rerun that arm "
            "with SAFESWM_DUMP_LATENTS instrumentation"
        )
    latent_path = matched_root / latent_relative
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    actual_hash = sha256_file(latent_path)
    recorded_hash = record.get("latent_sha256")
    if recorded_hash is not None and actual_hash != recorded_hash:
        raise RuntimeError(
            f"latent hash mismatch for {record['rollout_id']}: "
            f"{actual_hash} != {recorded_hash}"
        )
    video_path = matched_root / record["video_path"]
    video_hash = sha256_file(video_path) if video_path.exists() else None
    recorded_video_hash = record.get("video_sha256")
    if (
        video_hash is not None
        and recorded_video_hash is not None
        and video_hash != recorded_video_hash
    ):
        raise RuntimeError(
            f"video hash mismatch for {record['rollout_id']}: "
            f"{video_hash} != {recorded_video_hash}"
        )
    return {
        "rollout_id": record["rollout_id"],
        "arm": record["arm"],
        "latent_path": str(latent_path),
        "latent_sha256": actual_hash,
        "recorded_latent_sha256": recorded_hash,
        "latent_shape": record.get("latent_shape"),
        "latent_dtype": record.get("latent_dtype"),
        "video_path": str(video_path),
        "video_sha256": video_hash,
        "recorded_video_sha256": recorded_video_hash,
    }


def artifact_record(run_dir: Path, path: Path, schema: str) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw Gamma-World decoder temporal support"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--n-views", type=int, default=2)
    parser.add_argument(
        "--hybrid-indices",
        default="none",
        help="none, differing, first-differing, all, or e.g. 24,26-28",
    )
    parser.add_argument(
        "--allow-dense",
        action="store_true",
        help="required when more than one hybrid decode is requested",
    )
    parser.add_argument(
        "--save-raw-decodes",
        action="store_true",
        help="save large [V,T,H,W,C] uint8 .npy decoder outputs",
    )
    parser.add_argument(
        "--mp4-diagnostic",
        action="store_true",
        help="compare raw re-decodes with existing lossy MP4s",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="decode control once, write hashes, and stop",
    )
    parser.add_argument(
        "--allow-cross-protocol-source",
        action="store_true",
        help=(
            "allow exploratory matched latents generated under a different "
            "protocol; records both hashes and uses them only for decoder support"
        ),
    )
    args = parser.parse_args()

    if args.n_views < 1:
        parser.error("--n-views must be positive")
    for path in (
        args.config,
        args.matched / "manifest.json",
        args.tokenizer,
    ):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")

    protocol = json.loads(args.config.read_text())
    matched_manifest_path = args.matched / "manifest.json"
    matched_manifest = json.loads(matched_manifest_path.read_text())
    config_hash = sha256_file(args.config)
    recorded_config_hash = matched_manifest.get("config_sha256")
    config_mismatch = (
        recorded_config_hash is not None
        and config_hash != recorded_config_hash
    )
    if config_mismatch and not args.allow_cross_protocol_source:
        raise RuntimeError(
            f"config hash differs from matched manifest: {config_hash} != "
            f"{recorded_config_hash}"
        )
    control_record, stop_record = find_pair(
        matched_manifest, args.scene, args.seed, args.delay
    )
    scene = control_record["scene"]
    run_dir = args.output / f"{scene}__seed{args.seed}__d{args.delay}"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(GAMMA_REPO))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Gamma-World tokenizer decoding requires CUDA")
    torch.set_grad_enabled(False)
    determinism = configure_determinism(torch)

    control_source = source_record(args.matched, control_record)
    stop_source = source_record(args.matched, stop_record)
    control_latent = load_tensor(
        torch, Path(control_source["latent_path"])
    )
    stop_latent = load_tensor(torch, Path(stop_source["latent_path"]))
    if tuple(control_latent.shape) != tuple(stop_latent.shape):
        raise ValueError(
            f"matched latent shapes differ: {list(control_latent.shape)} "
            f"versus {list(stop_latent.shape)}"
        )
    control_views = split_views(control_latent, args.n_views)
    stop_views = split_views(stop_latent, args.n_views)
    latent_summary = latent_difference_summary(control_views, stop_views)
    latent_count = int(control_views.shape[3])
    temporal_stride = int(protocol["model"]["latent_stride_pixels"])
    expected_pixel_frames = 1 + (latent_count - 1) * temporal_stride
    selected_indices = parse_index_spec(
        args.hybrid_indices,
        latent_count=latent_count,
        differing_indices=latent_summary["differing_indices_union"],
    )
    if len(selected_indices) > 1 and not args.allow_dense:
        parser.error(
            f"{len(selected_indices)} hybrid decodes requested; pass "
            "--allow-dense to acknowledge the expensive dense experiment"
        )

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "raw_decoder_receptive_field",
        "pair": {
            "scene": scene,
            "seed": args.seed,
            "delay_blocks": args.delay,
            "control_rollout_id": control_record["rollout_id"],
            "stop_rollout_id": stop_record["rollout_id"],
        },
        "source": {
            "config": {
                "path": str(args.config),
                "sha256": config_hash,
            },
            "matched_source_config_sha256": recorded_config_hash,
            "cross_protocol_source_allowed": bool(
                config_mismatch and args.allow_cross_protocol_source
            ),
            "cross_protocol_scope": (
                "decoder temporal support only; no detector calibration or "
                "intervention outcome estimate"
                if config_mismatch
                else None
            ),
            "matched_manifest": {
                "path": str(matched_manifest_path),
                "sha256": sha256_file(matched_manifest_path),
                "schema_version": matched_manifest.get("schema_version"),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "gamma_world": git_metadata(GAMMA_REPO),
            "tokenizer_checkpoint": {
                "path": str(args.tokenizer),
                "sha256": sha256_file(args.tokenizer),
            },
            "control": control_source,
            "stop": stop_source,
        },
        "decoder": {
            "entrypoint": (
                "gamma_world._src.predict2.tokenizers.wan2pt1."
                "Wan2pt1VAEInterface.decode"
            ),
            "equivalence": (
                "Gamma-World model.decode splits V*T into independent views "
                "and calls this tokenizer decoder"
            ),
            "input_layout": "B,C,V*T,H,W",
            "analysis_layout": "V,T,H,W,C",
            "comparison_domain": "raw_uint8_pre_mp4",
            "uint8_conversion": "clamp((decode_float + 1) * 127.5, 0, 255).to(uint8)",
            "saved_latent_precision_warning": (
                "matched latents were persisted as FP16; generation decoded "
                "the sampler tensor before this downcast"
            ),
            "n_views": args.n_views,
            "latent_frames_per_view": latent_count,
            "expected_pixel_frames": expected_pixel_frames,
            "temporal_stride": temporal_stride,
            "expected_first_support": (
                f"{temporal_stride}*t-{temporal_stride - 1} for t>0; 0 for t=0"
            ),
        },
        "determinism_settings": determinism,
        "latent_stop_control": latent_summary,
        "requested_hybrid_indices": selected_indices,
        "artifacts": {},
    }
    atomic_write_json(manifest_path, manifest)

    decoder = GammaWorldDecoder(args.tokenizer, args.n_views, torch)
    control_raw = decoder.decode_uint8(control_latent)
    if control_raw.shape[1] != expected_pixel_frames:
        raise RuntimeError(
            f"decoder returned {control_raw.shape[1]} frames, expected "
            f"{expected_pixel_frames}"
        )

    if args.save_raw_decodes:
        manifest["artifacts"]["control_raw"] = save_raw_decode(
            run_dir / "control_raw_uint8.npy", control_raw
        )

    if args.smoke:
        smoke_path = run_dir / "smoke.json"
        smoke = {
            "schema_version": SMOKE_SCHEMA,
            "status": "complete",
            "source_latent_sha256": control_source["latent_sha256"],
            "raw_dtype": str(control_raw.dtype),
            "raw_shape_vthwc": list(control_raw.shape),
            "raw_sha256": sha256_array(control_raw),
            "raw_min": int(control_raw.min()),
            "raw_max": int(control_raw.max()),
        }
        atomic_write_json(smoke_path, smoke)
        manifest["status"] = "smoke_complete"
        manifest["completed_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        manifest["artifacts"]["smoke"] = artifact_record(
            run_dir, smoke_path, SMOKE_SCHEMA
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_text(
            run_dir / "manifest.sha256", sha256_file(manifest_path) + "\n"
        )
        print(f"smoke decode saved -> {run_dir}", flush=True)
        return

    repeated_control_raw = decoder.decode_uint8(control_latent)
    determinism_comparison = comparison_payload(
        first=control_raw,
        second=repeated_control_raw,
        first_name="control_decode_1",
        second_name="control_decode_2",
        metrics_path=run_dir / "metrics" / "determinism.npz",
    )
    determinism_payload = {
        "schema_version": DETERMINISM_SCHEMA,
        "source_latent_sha256": control_source["latent_sha256"],
        **determinism_comparison,
    }
    determinism_path = run_dir / "determinism.json"
    atomic_write_json(determinism_path, determinism_payload)
    manifest["artifacts"]["determinism"] = artifact_record(
        run_dir, determinism_path, DETERMINISM_SCHEMA
    )
    del repeated_control_raw

    stop_raw = decoder.decode_uint8(stop_latent)
    if args.save_raw_decodes:
        manifest["artifacts"]["stop_raw"] = save_raw_decode(
            run_dir / "stop_raw_uint8.npy", stop_raw
        )

    first_latent_indices = [
        row["first_differing_index"] for row in latent_summary["views"]
    ]
    full_comparison = comparison_payload(
        first=control_raw,
        second=stop_raw,
        first_name="control",
        second_name="stop",
        metrics_path=run_dir / "metrics" / "full_stop_control.npz",
        latent_indices_by_view=first_latent_indices,
        temporal_stride=temporal_stride,
    )
    full_payload = {
        "schema_version": FULL_SCHEMA,
        "latent_stop_control": latent_summary,
        **full_comparison,
    }
    full_path = run_dir / "full_stop_control.json"
    atomic_write_json(full_path, full_payload)
    manifest["artifacts"]["full_stop_control"] = artifact_record(
        run_dir, full_path, FULL_SCHEMA
    )

    hybrid_path = run_dir / "hybrid_results.jsonl"
    hybrid_rows: list[dict[str, Any]] = []
    for position, latent_index in enumerate(selected_indices, start=1):
        hybrid_views = control_views.clone()
        hybrid_views[:, :, :, latent_index] = stop_views[
            :, :, :, latent_index
        ]
        replaced_difference = (
            control_views[:, :, :, latent_index].float()
            - stop_views[:, :, :, latent_index].float()
        ).abs()
        started = time.perf_counter()
        hybrid_raw = decoder.decode_uint8(combine_views(hybrid_views))
        elapsed = time.perf_counter() - started
        comparison = comparison_payload(
            first=control_raw,
            second=hybrid_raw,
            first_name="control",
            second_name=f"hybrid_t{latent_index}",
            metrics_path=(
                run_dir / "metrics" / f"hybrid_t{latent_index:03d}.npz"
            ),
            latent_indices_by_view=[latent_index] * args.n_views,
            temporal_stride=temporal_stride,
        )
        row = {
            "schema_version": HYBRID_SCHEMA,
            "latent_index": latent_index,
            "replacement": (
                "replace STOP values at temporal index in every view of "
                "the control latent"
            ),
            "replacement_views": list(range(args.n_views)),
            "source_index_differs": bool(replaced_difference.max() > 0),
            "source_index_max_abs_error": float(replaced_difference.max()),
            "source_index_changed_values": int(
                torch.count_nonzero(replaced_difference)
            ),
            "decode_elapsed_s": elapsed,
            **comparison,
        }
        if args.save_raw_decodes:
            row["raw_artifact"] = save_raw_decode(
                run_dir / f"hybrid_t{latent_index:03d}_raw_uint8.npy",
                hybrid_raw,
            )
        hybrid_rows.append(row)
        atomic_write_jsonl(hybrid_path, hybrid_rows)
        del hybrid_raw, hybrid_views
        print(
            f"hybrid {position}/{len(selected_indices)} t={latent_index} "
            f"decoded in {elapsed:.1f}s",
            flush=True,
        )

    if hybrid_rows or selected_indices == []:
        atomic_write_jsonl(hybrid_path, hybrid_rows)
        manifest["artifacts"]["hybrid_results"] = artifact_record(
            run_dir, hybrid_path, HYBRID_SCHEMA
        )

    if args.mp4_diagnostic:
        diagnostic = mp4_diagnostic(
            run_dir=run_dir,
            matched_root=args.matched,
            control_record=control_record,
            stop_record=stop_record,
            control_raw=control_raw,
            stop_raw=stop_raw,
        )
        diagnostic_path = run_dir / "mp4_diagnostic.json"
        atomic_write_json(diagnostic_path, diagnostic)
        manifest["artifacts"]["mp4_diagnostic"] = artifact_record(
            run_dir, diagnostic_path, MP4_SCHEMA
        )

    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    manifest["determinism_exact"] = determinism_payload["exact"]
    manifest["full_comparison_exact"] = full_payload["exact"]
    manifest["hybrid_count"] = len(hybrid_rows)
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(
        run_dir / "manifest.sha256", sha256_file(manifest_path) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(run_dir),
                "determinism_exact": manifest["determinism_exact"],
                "full_comparison_exact": manifest["full_comparison_exact"],
                "hybrid_count": manifest["hybrid_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
