"""Validate manifest bindings and optionally every generated artifact hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_PROTOCOL,
    assert_protocol_current,
    canonical_json_bytes,
    load_json,
    protocol_sha256,
    sha256_bytes,
    sha256_file,
)


def _key(entry: dict[str, Any]) -> tuple[str, int]:
    return entry["split"], int(entry["episode_id"])


def validate_all(root: Path, protocol_path: Path, deep: bool) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    assert_protocol_current(protocol)
    protocol_hash = protocol_sha256(protocol_path)
    episodes = {_key(entry): entry for entry in protocol["episodes"]}
    if len(episodes) != 597:
        raise ValueError("protocol does not contain 597 unique split/episode pairs")

    downloads_path = root / "manifests/downloads.json"
    downloads = load_json(downloads_path)
    if downloads["protocol_sha256"] != protocol_hash:
        raise ValueError("download manifest protocol binding mismatch")
    downloaded = {_key(entry): entry for entry in downloads["files"]}
    if downloaded.keys() != episodes.keys():
        raise ValueError("download manifest does not exactly cover the protocol")
    for key, entry in downloaded.items():
        episode = episodes[key]
        if entry["source_url"] != episode["source_url"]:
            raise ValueError(f"{key}: source URL changed")
        if entry["jsonl_records"] != episode["frame_count"]:
            raise ValueError(f"{key}: source frame count mismatch")
        if not entry["source_url"].endswith(".jsonl") or ".mp4" in entry["source_url"]:
            raise ValueError(f"{key}: non-action source in download manifest")
        if deep and sha256_file(Path(entry["local_path"])) != entry["sha256"]:
            raise ValueError(f"{key}: downloaded file hash mismatch")

    conversion_path = root / "manifests/conversion.json"
    conversion = load_json(conversion_path)
    if conversion["protocol_sha256"] != protocol_hash:
        raise ValueError("conversion manifest protocol binding mismatch")
    converted = {_key(entry): entry for entry in conversion["files"]}
    if converted.keys() != episodes.keys():
        raise ValueError("conversion manifest does not exactly cover the protocol")
    for key, entry in converted.items():
        episode = episodes[key]
        expected_frames = (episode["frame_count"] * 4 + 4) // 5
        if entry["source_sha256"] != downloaded[key]["sha256"]:
            raise ValueError(f"{key}: conversion source hash mismatch")
        if entry["output_frames"] != expected_frames:
            raise ValueError(f"{key}: converted frame count mismatch")
        if deep:
            path = Path(entry["local_path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{key}: converted file hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                if arrays["keyboard"].shape != (expected_frames, 23):
                    raise ValueError(f"{key}: keyboard shape mismatch")
                if arrays["keyboard"].dtype != np.uint8:
                    raise ValueError(f"{key}: keyboard dtype mismatch")
                if arrays["camera_degrees"].shape != (expected_frames, 2):
                    raise ValueError(f"{key}: camera shape mismatch")
                if arrays["keyboard_token"].dtype != np.uint32:
                    raise ValueError(f"{key}: keyboard token dtype mismatch")

    quantizer = load_json(root / "manifests/camera_quantizer.json")
    quantizer_copy = dict(quantizer)
    quantizer_hash = quantizer_copy.pop("quantizer_sha256")
    if sha256_bytes(canonical_json_bytes(quantizer_copy)) != quantizer_hash:
        raise ValueError("camera quantizer hash mismatch")
    if quantizer["protocol_sha256"] != protocol_hash:
        raise ValueError("camera quantizer protocol binding mismatch")
    train_keys = {key for key in episodes if key[0] == "train"}
    fit_ids = {("train", int(entry["episode_id"])) for entry in quantizer["fit_files"]}
    if fit_ids != train_keys or quantizer["fit_split"] != "train":
        raise ValueError("camera quantizer was not fit on exactly the train split")
    for fit in quantizer["fit_files"]:
        key = ("train", int(fit["episode_id"]))
        if fit["sha256"] != converted[key]["sha256"]:
            raise ValueError(f"{key}: quantizer fit hash mismatch")

    blocks_path = root / "manifests/blocks.json"
    blocks = load_json(blocks_path)
    if blocks["protocol_sha256"] != protocol_hash:
        raise ValueError("block manifest protocol binding mismatch")
    if blocks["conversion_manifest_sha256"] != sha256_file(conversion_path):
        raise ValueError("block conversion-manifest hash mismatch")
    if blocks["quantizer_sha256"] != quantizer_hash:
        raise ValueError("block quantizer hash mismatch")
    blocked = {_key(entry): entry for entry in blocks["files"]}
    if blocked.keys() != episodes.keys():
        raise ValueError("block manifest does not exactly cover the protocol")
    for key, entry in blocked.items():
        expected_blocks = converted[key]["output_frames"] // 12
        if entry["block_count"] != expected_blocks:
            raise ValueError(f"{key}: block count mismatch")
        if entry["source_converted_sha256"] != converted[key]["sha256"]:
            raise ValueError(f"{key}: block source hash mismatch")
        if deep:
            path = Path(entry["local_path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{key}: block file hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                expected_shape = (expected_blocks, 12)
                if arrays["keyboard"].shape != (*expected_shape, 23):
                    raise ValueError(f"{key}: block keyboard shape mismatch")
                if arrays["camera_token"].shape != (*expected_shape, 2):
                    raise ValueError(f"{key}: block camera-token shape mismatch")

    report = {
        "protocol_sha256": protocol_hash,
        "downloaded_files": len(downloaded),
        "downloaded_bytes": sum(entry["bytes"] for entry in downloaded.values()),
        "source_frames": sum(entry["frame_count"] for entry in episodes.values()),
        "converted_files": len(converted),
        "converted_frames": sum(entry["output_frames"] for entry in converted.values()),
        "quantizer_fit_split": quantizer["fit_split"],
        "quantizer_fit_episodes": quantizer["fit_episode_count"],
        "quantizer_fit_frames": quantizer["fit_frame_count"],
        "block_files": len(blocked),
        "blocks": sum(entry["block_count"] for entry in blocked.values()),
        "deep_hash_validation": deep,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_all(args.root, args.protocol, args.deep), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

