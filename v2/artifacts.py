"""Atomic writes and artifact verification for RTWM v2."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from protocol import file_sha256


class ArtifactError(RuntimeError):
    """Raised when a required artifact is absent, stale, or corrupt."""


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def temporary_path(target: str | Path, *, suffix: str | None = None) -> Iterator[Path]:
    """Yield a same-directory temporary path and remove it on failure."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_suffix = destination.suffix if suffix is None else suffix
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".tmp{final_suffix}",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def commit_temporary(temporary: str | Path, destination: str | Path) -> None:
    source = Path(temporary)
    target = Path(destination)
    if not source.is_file():
        raise ArtifactError(f"writer did not create temporary artifact: {source}")
    with source.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(source, target)
    _fsync_directory(target.parent)


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    with temporary_path(path) as temporary:
        writer(temporary)
        commit_temporary(temporary, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    def write(temporary: Path) -> None:
        temporary.write_text(encoded)

    atomic_write(path, write)


def atomic_torch_save(path: str | Path, payload: Any, torch_module: Any) -> None:
    atomic_write(path, lambda temporary: torch_module.save(payload, temporary))


def atomic_numpy_save(path: str | Path, payload: Any, numpy_module: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            numpy_module.save(handle, payload, allow_pickle=False)

    atomic_write(path, write)


def atomic_numpy_savez(path: str | Path, numpy_module: Any, **arrays: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            numpy_module.savez_compressed(handle, **arrays)

    atomic_write(path, write)


def artifact_metadata(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    required: bool = True,
    shape: list[int] | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactError(f"missing artifact: {artifact}")
    displayed_path = artifact
    if relative_to is not None:
        displayed_path = artifact.relative_to(Path(relative_to))
    metadata: dict[str, Any] = {
        "path": str(displayed_path),
        "sha256": file_sha256(artifact),
        "bytes": artifact.stat().st_size,
        "required": required,
    }
    if shape is not None:
        metadata["shape"] = shape
    if dtype is not None:
        metadata["dtype"] = dtype
    return metadata


def verify_artifact(
    root: str | Path,
    metadata: Mapping[str, Any],
    *,
    label: str = "artifact",
) -> Path:
    if not metadata.get("path"):
        raise ArtifactError(f"{label} has no path")
    path = Path(root) / str(metadata["path"])
    if not path.is_file():
        raise ArtifactError(f"missing {label}: {path}")
    expected_size = metadata.get("bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ArtifactError(f"{label} size mismatch: {path}")
    expected_hash = metadata.get("sha256")
    if not expected_hash:
        raise ArtifactError(f"{label} has no sha256")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ArtifactError(
            f"{label} sha256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return path


LEGACY_ARTIFACT_FIELDS = {
    "video": ("video_path", "video_sha256"),
    "latent": ("latent_path", "latent_sha256"),
    "decoded_u8": ("decoded_u8_path", "decoded_u8_sha256"),
    "detector_frames": ("detector_frames_path", "detector_frames_sha256"),
}


def record_artifacts(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return normalized artifact metadata from new or legacy manifests."""
    artifacts = record.get("artifacts")
    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(artifacts, Mapping):
        for name, metadata in artifacts.items():
            if isinstance(metadata, Mapping):
                normalized[str(name)] = dict(metadata)
    for name, (path_key, hash_key) in LEGACY_ARTIFACT_FIELDS.items():
        if name not in normalized and record.get(path_key):
            normalized[name] = {
                "path": record[path_key],
                "sha256": record.get(hash_key),
                "required": name != "video",
            }
    return normalized


def verify_rollout_artifacts(
    root: str | Path,
    record: Mapping[str, Any],
    *,
    required: tuple[str, ...] = ("latent", "decoded_u8", "detector_frames"),
) -> dict[str, Path]:
    artifacts = record_artifacts(record)
    verified: dict[str, Path] = {}
    for name in required:
        if name not in artifacts:
            raise ArtifactError(f"rollout {record.get('rollout_id')} missing required {name}")
        verified[name] = verify_artifact(root, artifacts[name], label=name)
    if "video" in artifacts:
        verified["video"] = verify_artifact(root, artifacts["video"], label="video")
    return verified
