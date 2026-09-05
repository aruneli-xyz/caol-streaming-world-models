"""Build or verify the compact frozen episode-selection protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin

from actions import ACTION_KEYS, CAMERA_SCALER, KEYBOARD_KEYS
from common import (
    DEFAULT_PROTOCOL,
    ROOT,
    atomic_write_json,
    canonical_json_bytes,
    converter_sha256,
    load_json,
    sha256_bytes,
    sha256_file,
)


def _version(index_path: Path) -> str:
    part = index_path.name.split("_")[1]
    if not part.endswith("xx") or not part[:-2].isdigit():
        raise ValueError(f"cannot determine major version from {index_path.name}")
    return part[:-2]


def _source_index(
    paths: list[Path],
) -> tuple[dict[tuple[str, str], list[dict[str, str]]], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, str]]] = {}
    provenance: list[dict[str, Any]] = []
    for path in sorted(paths):
        data = load_json(path)
        basedir = data["basedir"]
        version = _version(path)
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "basedir": basedir,
                "relpath_count": len(data["relpaths"]),
            }
        )
        for relpath in data["relpaths"]:
            stem = PurePosixPath(relpath).stem
            item = {
                "index_file": str(path.resolve()),
                "index_file_sha256": provenance[-1]["sha256"],
                "basedir": basedir,
                "relpath": str(PurePosixPath(relpath).with_suffix(".jsonl")),
            }
            values = lookup.setdefault((version, stem), [])
            if item not in values:
                values.append(item)
    return lookup, provenance


def _rank(protocol_id: str, episode: dict[str, Any]) -> str:
    identity = "\0".join(
        (
            protocol_id,
            "train",
            str(episode["episode_id"]),
            episode["actions_path"],
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_protocol(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    source = config["source"]
    metadata_dir = Path(source["metadata_dir"])
    train_path = metadata_dir / source["train_metadata"]
    test_path = metadata_dir / source["test_metadata"]
    index_paths = sorted(metadata_dir.glob(source["indices_glob"]))
    if not index_paths:
        raise FileNotFoundError("no VPT index files matched")
    lookup, index_provenance = _source_index(index_paths)

    metadata_provenance = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in (train_path, test_path)
    ]
    protocol_id = config["protocol_id"]

    def enrich(
        episode: dict[str, Any], split: str, selection_rank: str | None
    ) -> dict[str, Any]:
        version = episode["actions_path"].split("/", 1)[0]
        if not version.startswith("v"):
            raise ValueError(f"unexpected actions path: {episode['actions_path']}")
        key = (version[1:], PurePosixPath(episode["actions_path"]).stem)
        candidates = lookup.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f"{key} has {len(candidates)} source mappings")
        source_item = candidates[0]
        return {
            "split": split,
            "episode_id": int(episode["episode_id"]),
            "frame_count": int(episode["length"]),
            "metadata_actions_path": episode["actions_path"],
            "metadata_video_path": episode["video_path"],
            "source_index_file": source_item["index_file"],
            "source_index_file_sha256": source_item["index_file_sha256"],
            "source_relpath": source_item["relpath"],
            "source_url": urljoin(source_item["basedir"], source_item["relpath"]),
            "selection_rank": selection_rank,
        }

    train_metadata = load_json(train_path)["episodes"]
    ranked: list[tuple[str, dict[str, Any]]] = []
    ambiguous_count = 0
    for episode in train_metadata:
        version = episode["actions_path"].split("/", 1)[0][1:]
        key = (version, PurePosixPath(episode["actions_path"]).stem)
        if len(lookup.get(key, [])) != 1:
            ambiguous_count += 1
            continue
        ranked.append((_rank(protocol_id, episode), episode))
    ranked.sort(key=lambda item: (item[0], int(item[1]["episode_id"])))
    train_count = int(config["selection"]["train_count"])
    if len(ranked) < train_count:
        raise ValueError("not enough unambiguous train episodes")
    train = [enrich(episode, "train", rank) for rank, episode in ranked[:train_count]]

    test_metadata = sorted(
        load_json(test_path)["episodes"], key=lambda episode: int(episode["episode_id"])
    )
    test = [enrich(episode, "test", None) for episode in test_metadata]
    if len(test) != 85:
        raise ValueError(f"frozen protocol requires 85 test episodes, found {len(test)}")

    action_config = config["actions"]
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "config_sha256": sha256_file(config_path),
        "selection": {
            **config["selection"],
            "eligible_train_count": len(ranked),
            "excluded_ambiguous_train_count": ambiguous_count,
            "selected_train_count": len(train),
            "selected_test_count": len(test),
            "test_expected_count": 85,
        },
        "source_metadata": {
            "episode_files": metadata_provenance,
            "index_files": index_provenance,
        },
        "action_spec": {
            "ordering": list(ACTION_KEYS),
            "keyboard_ordering": list(KEYBOARD_KEYS),
            "camera_ordering": ["cameraX_yaw", "cameraY_pitch"],
            "camera_scaler_degrees_per_pixel": CAMERA_SCALER,
            "source_fps": action_config["source_fps"],
            "target_fps": action_config["target_fps"],
            "block_frames": action_config["block_frames"],
            "camera_quantization_bins": action_config["camera_quantization_bins"],
            "semantics_source": str(
                (
                    metadata_dir.parent / "src" / "data" / "minecraft.py"
                ).resolve()
            ),
            "semantics_source_sha256": sha256_file(
                metadata_dir.parent / "src" / "data" / "minecraft.py"
            ),
        },
        "converter_sha256": converter_sha256(),
        "licensing_notes": [
            "Actions are fetched from OpenAI's public minecraft-rl blob endpoint; videos are never requested.",
            "The local Solaris code is Apache-2.0, but that software license does not by itself license the VPT recordings or action data.",
            "The bundled metadata provides no dataset license text. Downstream users must verify OpenAI/Minecraft dataset terms before redistribution or use.",
            "Frozen manifests contain metadata, URLs, and hashes; generated downloads and arrays are excluded from version control.",
        ],
        "episodes": sorted(train + test, key=lambda item: (item["split"], item["episode_id"])),
    }
    protocol["episodes_sha256"] = sha256_bytes(
        canonical_json_bytes(protocol["episodes"])
    )
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--check", action="store_true", help="fail unless output is byte-identical"
    )
    args = parser.parse_args()
    protocol = build_protocol(args.config)
    encoded = canonical_json_bytes(protocol)
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != encoded:
            raise SystemExit(f"{args.output} is not current")
    else:
        atomic_write_json(args.output, protocol)
    print(
        json.dumps(
            {
                "episodes": len(protocol["episodes"]),
                "train": protocol["selection"]["selected_train_count"],
                "test": protocol["selection"]["selected_test_count"],
                "protocol_content_sha256": sha256_bytes(encoded),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

