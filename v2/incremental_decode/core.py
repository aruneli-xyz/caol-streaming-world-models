"""GPU-independent contracts for the Wan2.1 incremental decode probe."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "rtwm-v2-incremental-decode-1"
MANIFEST_HASH_KEY = "manifest_sha256"


class ProbeError(RuntimeError):
    """Raised when provenance, resume, or exactness requirements fail."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash an array's dtype, shape, and C-order bytes."""
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {"dtype": str(value.dtype), "shape": [int(x) for x in value.shape]}
        )
    )
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    encoded = json.dumps(
        payload, allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, encoded)


def atomic_save_npy(path: str | Path, array: Any) -> None:
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def split_gamma_views(latent: Any, n_views: int = 2) -> Any:
    """Convert Gamma ``[B,C,V*T,H,W]`` to ``[B*V,C,T,H,W]``."""
    if getattr(latent, "ndim", None) != 5:
        raise ValueError("latent must have shape [B,C,V*T,H,W]")
    batch, channels, combined_time, height, width = latent.shape
    if n_views < 1 or combined_time % n_views:
        raise ValueError(
            f"combined temporal axis {combined_time} is not divisible by {n_views} views"
        )
    time_per_view = combined_time // n_views
    return (
        latent.reshape(batch, channels, n_views, time_per_view, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * n_views, channels, time_per_view, height, width)
        .contiguous()
    )


def validate_block_plan(
    total_latents: int, *, block_latents: int = 3, max_blocks: int | None = None
) -> list[tuple[int, int]]:
    if total_latents < 1:
        raise ValueError("total_latents must be positive")
    if block_latents != 3:
        raise ValueError("Gamma incremental decode requires 3-latent blocks")
    block_count = total_latents // block_latents
    if max_blocks is not None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        block_count = min(block_count, max_blocks)
    if block_count < 1:
        raise ValueError("input has fewer than one complete 3-latent block")
    return [
        (index * block_latents, (index + 1) * block_latents)
        for index in range(block_count)
    ]


def global_normalization_slice(
    latent_chunk: Any,
    video_mean: Any,
    video_std: Any,
    *,
    global_start: int,
) -> Any:
    """Undo normalization with the chunk's global, not local, time indices."""
    if getattr(latent_chunk, "ndim", None) != 5:
        raise ValueError("latent chunk must be [B*V,C,T,H,W]")
    if global_start < 0:
        raise ValueError("global_start must be non-negative")
    end = global_start + latent_chunk.shape[2]
    if video_mean.shape[2] < end or video_std.shape[2] < end:
        raise ValueError(
            f"normalization tables have {video_mean.shape[2]} frames, need {end}"
        )
    mean = video_mean[:, :, global_start:end].to(
        device=latent_chunk.device, dtype=latent_chunk.dtype
    )
    std = video_std[:, :, global_start:end].to(
        device=latent_chunk.device, dtype=latent_chunk.dtype
    )
    return (latent_chunk * std + mean).contiguous()


def decoded_to_uint8_views(decoded: Any, *, batch: int, n_views: int) -> Any:
    """Convert ``[B*V,3,T,H,W]`` decoder output to ``[V,T,H,W,3]``."""
    import torch

    if decoded.ndim != 5 or decoded.shape[0] != batch * n_views:
        raise ValueError(f"unexpected decoded shape {tuple(decoded.shape)}")
    if decoded.shape[1] != 3 or batch != 1:
        raise ValueError("probe requires batch=1 and three output channels")
    raw = ((decoded.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return (
        raw.reshape(batch, n_views, 3, *raw.shape[2:])
        .permute(0, 1, 3, 4, 5, 2)
        .contiguous()[0]
    )


def compare_exact(first: Any, second: Any) -> dict[str, Any]:
    import numpy as np

    first = np.asarray(first)
    second = np.asarray(second)
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise TypeError("exact raw comparison requires uint8 arrays")
    if first.shape != second.shape or first.ndim != 5:
        raise ValueError(
            f"expected equal [V,T,H,W,C] arrays, got {first.shape} and {second.shape}"
        )
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    views = []
    for view in range(first.shape[0]):
        per_frame = difference[view].reshape(first.shape[1], -1).max(axis=1)
        views.append(
            {
                "view": view,
                "frames": int(first.shape[1]),
                "max_error": int(per_frame.max(initial=0)),
                "changed_frames": [
                    int(index) for index in np.flatnonzero(per_frame)
                ],
                "exact": bool(not np.any(per_frame)),
            }
        )
    maximum = int(difference.max(initial=0))
    return {
        "domain": "raw_uint8_pre_encoding",
        "shape_vthwc": [int(x) for x in first.shape],
        "first_sha256": array_sha256(first),
        "second_sha256": array_sha256(second),
        "max_error": maximum,
        "exact": maximum == 0,
        "views": views,
    }


def artifact_metadata(path: Path, *, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ProbeError(f"retained artifact path escapes output root: {path}") from error
    return {
        "path": relative.as_posix(),
        "bytes": resolved_path.stat().st_size,
        "sha256": file_sha256(resolved_path),
    }


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if MANIFEST_HASH_KEY in payload:
        raise ValueError(f"{MANIFEST_HASH_KEY} must not be supplied")
    sealed = dict(payload)
    sealed[MANIFEST_HASH_KEY] = canonical_sha256(payload)
    return sealed


def verify_sealed_manifest(payload: Mapping[str, Any]) -> None:
    expected = payload.get(MANIFEST_HASH_KEY)
    if not isinstance(expected, str):
        raise ProbeError("manifest is not sealed")
    body = {key: value for key, value in payload.items() if key != MANIFEST_HASH_KEY}
    actual = canonical_sha256(body)
    if actual != expected:
        raise ProbeError(f"manifest seal mismatch: expected {expected}, got {actual}")


def verify_artifacts(root: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    for label, metadata in artifacts.items():
        relative = Path(str(metadata.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ProbeError(f"{label} does not use a safe relative path")
        path = root / relative
        if not path.is_file():
            raise ProbeError(f"missing retained artifact {label}: {relative}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ProbeError(f"size mismatch for retained artifact {label}")
        if file_sha256(path) != metadata["sha256"]:
            raise ProbeError(f"hash mismatch for retained artifact {label}")


def strict_resume(
    manifest_path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    verify_sealed_manifest(payload)
    if payload.get("identity") != dict(expected_identity):
        raise ProbeError("existing immutable manifest identity differs")
    verify_artifacts(manifest_path.parent, payload.get("artifacts", {}))
    if payload.get("status") != "complete":
        raise ProbeError("existing immutable manifest is not complete")
    return payload


def percentile_summary(values: Iterable[float]) -> dict[str, float | int]:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min_ms": float(array.min()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "max_ms": float(array.max()),
        "mean_ms": float(array.mean()),
    }
