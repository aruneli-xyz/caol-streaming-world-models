"""Shared deterministic I/O and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "manifests" / "protocol.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def converter_sha256() -> str:
    """Hash every source file that defines canonical conversion semantics."""

    digest = hashlib.sha256()
    for name in ("actions.py", "convert.py"):
        path = ROOT / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def protocol_sha256(protocol_path: Path) -> str:
    return sha256_file(protocol_path)


def selected_episodes(
    protocol: dict[str, Any], splits: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    allowed = set(splits or ("train", "test"))
    return [episode for episode in protocol["episodes"] if episode["split"] in allowed]


def assert_protocol_current(protocol: dict[str, Any]) -> None:
    expected = protocol["converter_sha256"]
    actual = converter_sha256()
    if actual != expected:
        raise RuntimeError(
            "converter hash does not match frozen protocol: "
            f"expected {expected}, got {actual}; regenerate and review protocol.json"
        )

