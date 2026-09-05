"""Convert downloaded VPT JSONL into canonical 16 FPS action arrays."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actions import KEYBOARD_KEYS, VPTActionState, convert_vpt_record
from common import (
    DEFAULT_PROTOCOL,
    assert_protocol_current,
    atomic_write_json,
    canonical_json_bytes,
    converter_sha256,
    load_json,
    protocol_sha256,
    selected_episodes,
    sha256_bytes,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    utf8_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "windows-1252"):
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding=encoding) as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON: {error}"
                        ) from error
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{path}:{line_number}: expected a JSON object"
                        )
                    records.append(value)
            return records
        except UnicodeDecodeError as error:
            if encoding == "windows-1252":
                raise
            utf8_error = error
    raise AssertionError(f"unreachable encoding fallback: {utf8_error}")


def convert_records(
    records: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    state = VPTActionState()
    keyboard: list[list[int]] = []
    camera: list[list[float]] = []
    for record in records:
        keys, movement = convert_vpt_record(record, state)
        keyboard.append(keys)
        camera.append(movement)
    return (
        np.asarray(keyboard, dtype=np.uint8).reshape((-1, len(KEYBOARD_KEYS))),
        np.asarray(camera, dtype=np.float32).reshape((-1, 2)),
    )


def resample_20_to_16(
    keyboard_20: np.ndarray, camera_20: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rational, timestamp-anchored resampling with camera conservation.

    Keyboard state at each 16 FPS timestamp is held from source index
    ``floor(5*j/4)``.  Camera deltas are extensive values, so every source
    delta is accumulated into target bin ``floor(4*i/5)``.  No interpolation,
    wall-clock arithmetic, or cross-episode state is used.
    """

    if keyboard_20.ndim != 2 or keyboard_20.shape[1] != len(KEYBOARD_KEYS):
        raise ValueError("keyboard_20 must have shape [frames, 23]")
    if camera_20.shape != (keyboard_20.shape[0], 2):
        raise ValueError("camera_20 must have shape [frames, 2]")
    source_count = keyboard_20.shape[0]
    if source_count == 0:
        return (
            keyboard_20.copy(),
            camera_20.copy(),
            np.empty((0,), dtype=np.int64),
        )

    target_count = (source_count * 4 + 4) // 5
    source_indices = (
        np.arange(target_count, dtype=np.int64) * 5 // 4
    )
    keyboard_16 = keyboard_20[source_indices].copy()
    camera_16 = np.zeros((target_count, 2), dtype=np.float32)
    target_bins = np.arange(source_count, dtype=np.int64) * 4 // 5
    np.add.at(camera_16, target_bins, camera_20)
    return keyboard_16, camera_16, source_indices


def pack_keyboard_array(keyboard: np.ndarray) -> np.ndarray:
    if keyboard.ndim != 2 or keyboard.shape[1] != len(KEYBOARD_KEYS):
        raise ValueError("keyboard must have shape [frames, 23]")
    if not np.all((keyboard == 0) | (keyboard == 1)):
        raise ValueError("keyboard is not binary")
    weights = (np.uint32(1) << np.arange(len(KEYBOARD_KEYS), dtype=np.uint32))
    return (keyboard.astype(np.uint32) * weights).sum(axis=1, dtype=np.uint32)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_entries(download_manifest: dict[str, Any]) -> dict[tuple[str, int], Any]:
    return {
        (entry["split"], int(entry["episode_id"])): entry
        for entry in download_manifest["files"]
    }


def convert_downloads(
    protocol_path: Path,
    download_manifest_path: Path,
    output_dir: Path,
    conversion_manifest_path: Path,
    splits: tuple[str, ...],
    limit: int | None,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    assert_protocol_current(protocol)
    protocol_hash = protocol_sha256(protocol_path)
    downloads = load_json(download_manifest_path)
    if downloads["protocol_sha256"] != protocol_hash:
        raise ValueError("download manifest is bound to a different protocol")
    downloaded = _download_entries(downloads)
    episodes = selected_episodes(protocol, splits)
    if limit is not None:
        episodes = episodes[:limit]

    converted: list[dict[str, Any]] = []
    for episode in episodes:
        key = (episode["split"], int(episode["episode_id"]))
        if key not in downloaded:
            continue
        download = downloaded[key]
        raw_path = Path(download["local_path"])
        raw_hash = sha256_file(raw_path)
        if raw_hash != download["sha256"]:
            raise ValueError(f"raw hash mismatch for {raw_path}")
        records = read_jsonl(raw_path)
        if len(records) != episode["frame_count"]:
            raise ValueError(
                f"{key}: metadata says {episode['frame_count']} frames, "
                f"JSONL has {len(records)} records"
            )
        keyboard_20, camera_20 = convert_records(records)
        keyboard, camera, source_indices = resample_20_to_16(
            keyboard_20, camera_20
        )
        output_path = (
            output_dir / episode["split"] / f"{int(episode['episode_id']):06d}.npz"
        )
        atomic_savez(
            output_path,
            keyboard=keyboard,
            keyboard_token=pack_keyboard_array(keyboard),
            camera_degrees=camera,
            source_indices_20fps=source_indices,
            source_frame_count=np.asarray(episode["frame_count"], dtype=np.int64),
            source_fps=np.asarray(20, dtype=np.int16),
            target_fps=np.asarray(16, dtype=np.int16),
        )
        converted.append(
            {
                "split": episode["split"],
                "episode_id": episode["episode_id"],
                "source_sha256": raw_hash,
                "source_frames": len(records),
                "output_frames": int(keyboard.shape[0]),
                "local_path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "converter_sha256": converter_sha256(),
        "source_fps": 20,
        "target_fps": 16,
        "resampling": {
            "keyboard": "source_index=floor(5*target_index/4)",
            "camera": "sum source deltas by target_bin=floor(4*source_index/5)",
        },
        "files": sorted(
            converted, key=lambda item: (item["split"], item["episode_id"])
        ),
    }
    atomic_write_json(conversion_manifest_path, manifest)
    return manifest


def fit_camera_quantizer(
    protocol_path: Path,
    conversion_manifest_path: Path,
    output_path: Path,
    bins: int = 256,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    assert_protocol_current(protocol)
    protocol_hash = protocol_sha256(protocol_path)
    conversion = load_json(conversion_manifest_path)
    if conversion["protocol_sha256"] != protocol_hash:
        raise ValueError("conversion manifest is bound to a different protocol")

    expected = {
        int(episode["episode_id"])
        for episode in selected_episodes(protocol, ("train",))
    }
    train_entries = {
        int(entry["episode_id"]): entry
        for entry in conversion["files"]
        if entry["split"] == "train"
    }
    missing = sorted(expected - set(train_entries))
    if missing:
        raise ValueError(
            f"quantizer fit requires every selected train episode; missing {len(missing)}"
        )

    cameras: list[np.ndarray] = []
    fit_files: list[dict[str, Any]] = []
    for episode_id in sorted(expected):
        entry = train_entries[episode_id]
        path = Path(entry["local_path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"converted hash mismatch for {path}")
        with np.load(path, allow_pickle=False) as arrays:
            cameras.append(arrays["camera_degrees"].astype(np.float64))
        fit_files.append({"episode_id": episode_id, "sha256": entry["sha256"]})
    values = np.concatenate(cameras, axis=0)
    probabilities = np.arange(1, bins, dtype=np.float64) / bins
    edges = np.quantile(values, probabilities, axis=0, method="linear")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "fit_split": "train",
        "fit_episode_count": len(expected),
        "fit_frame_count": int(values.shape[0]),
        "bins": bins,
        "algorithm": "per-axis equal-frequency edges; numpy.quantile(method=linear)",
        "axis_order": ["cameraX_yaw_degrees", "cameraY_pitch_degrees"],
        "edges": {
            "yaw": edges[:, 0].tolist(),
            "pitch": edges[:, 1].tolist(),
        },
        "numpy_version": np.__version__,
        "fit_files": fit_files,
    }
    payload["quantizer_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_json(output_path, payload)
    return payload


def quantize_camera(camera: np.ndarray, quantizer: dict[str, Any]) -> np.ndarray:
    if camera.ndim != 2 or camera.shape[1] != 2:
        raise ValueError("camera must have shape [frames, 2]")
    output = np.empty(camera.shape, dtype=np.uint16)
    output[:, 0] = np.searchsorted(
        np.asarray(quantizer["edges"]["yaw"]), camera[:, 0], side="right"
    )
    output[:, 1] = np.searchsorted(
        np.asarray(quantizer["edges"]["pitch"]), camera[:, 1], side="right"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    convert_parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("manifests/downloads.json"),
    )
    convert_parser.add_argument("--output-dir", type=Path, default=Path("arrays"))
    convert_parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/conversion.json")
    )
    convert_parser.add_argument(
        "--split", choices=("train", "test", "all"), default="all"
    )
    convert_parser.add_argument("--limit", type=int)

    fit_parser = subparsers.add_parser("fit-quantizer")
    fit_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    fit_parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=Path("manifests/conversion.json"),
    )
    fit_parser.add_argument(
        "--output", type=Path, default=Path("manifests/camera_quantizer.json")
    )
    fit_parser.add_argument("--bins", type=int, default=256)

    args = parser.parse_args()
    if args.command == "convert":
        splits = ("train", "test") if args.split == "all" else (args.split,)
        manifest = convert_downloads(
            args.protocol,
            args.download_manifest,
            args.output_dir,
            args.manifest,
            splits,
            args.limit,
        )
        print(json.dumps({"converted": len(manifest["files"])}, sort_keys=True))
    else:
        quantizer = fit_camera_quantizer(
            args.protocol, args.conversion_manifest, args.output, args.bins
        )
        print(
            json.dumps(
                {
                    "fit_episodes": quantizer["fit_episode_count"],
                    "fit_frames": quantizer["fit_frame_count"],
                    "quantizer_sha256": quantizer["quantizer_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

