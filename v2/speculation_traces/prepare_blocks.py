"""Form episode-local 12-frame Gamma action blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_PROTOCOL,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    protocol_sha256,
    sha256_bytes,
    sha256_file,
)
from convert import atomic_savez, quantize_camera


def prepare_blocks(
    protocol_path: Path,
    conversion_manifest_path: Path,
    quantizer_path: Path,
    output_dir: Path,
    manifest_path: Path,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    protocol_hash = protocol_sha256(protocol_path)
    conversion = load_json(conversion_manifest_path)
    quantizer = load_json(quantizer_path)
    if conversion["protocol_sha256"] != protocol_hash:
        raise ValueError("conversion manifest is bound to another protocol")
    if quantizer["protocol_sha256"] != protocol_hash:
        raise ValueError("quantizer is bound to another protocol")
    quantizer_copy = dict(quantizer)
    recorded_quantizer_hash = quantizer_copy.pop("quantizer_sha256")
    if sha256_bytes(canonical_json_bytes(quantizer_copy)) != recorded_quantizer_hash:
        raise ValueError("quantizer content hash mismatch")

    block_frames = int(protocol["action_spec"]["block_frames"])
    entries: list[dict[str, Any]] = []
    for converted in conversion["files"]:
        if converted["split"] not in splits:
            continue
        source_path = Path(converted["local_path"])
        if sha256_file(source_path) != converted["sha256"]:
            raise ValueError(f"converted hash mismatch for {source_path}")
        with np.load(source_path, allow_pickle=False) as arrays:
            keyboard = arrays["keyboard"]
            keyboard_token = arrays["keyboard_token"]
            camera = arrays["camera_degrees"]
            source_indices = arrays["source_indices_20fps"]
        block_count = keyboard.shape[0] // block_frames
        kept = block_count * block_frames
        camera_token = quantize_camera(camera[:kept], quantizer)
        output_path = (
            output_dir
            / converted["split"]
            / f"{int(converted['episode_id']):06d}.npz"
        )
        atomic_savez(
            output_path,
            keyboard=keyboard[:kept].reshape(block_count, block_frames, 23),
            keyboard_token=keyboard_token[:kept].reshape(block_count, block_frames),
            camera_degrees=camera[:kept].reshape(block_count, block_frames, 2),
            camera_token=camera_token.reshape(block_count, block_frames, 2),
            source_indices_20fps=source_indices[:kept].reshape(
                block_count, block_frames
            ),
        )
        entries.append(
            {
                "split": converted["split"],
                "episode_id": converted["episode_id"],
                "source_converted_sha256": converted["sha256"],
                "block_count": block_count,
                "block_frames": block_frames,
                "dropped_tail_frames": int(keyboard.shape[0] - kept),
                "local_path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "conversion_manifest_sha256": sha256_file(conversion_manifest_path),
        "quantizer_sha256": recorded_quantizer_hash,
        "episode_boundary_policy": "blocks never cross episodes; incomplete tails dropped",
        "block_frames": block_frames,
        "files": sorted(entries, key=lambda item: (item["split"], item["episode_id"])),
    }
    manifest["summary"] = {
        "episode_count": len(entries),
        "block_count": sum(entry["block_count"] for entry in entries),
        "dropped_tail_frames": sum(
            entry["dropped_tail_frames"] for entry in entries
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=Path("manifests/conversion.json"),
    )
    parser.add_argument(
        "--quantizer",
        type=Path,
        default=Path("manifests/camera_quantizer.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("blocks"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/blocks.json")
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()
    splits = ("train", "test") if args.split == "all" else (args.split,)
    manifest = prepare_blocks(
        args.protocol,
        args.conversion_manifest,
        args.quantizer,
        args.output_dir,
        args.manifest,
        splits,
    )
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

