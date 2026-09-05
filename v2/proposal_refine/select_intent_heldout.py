"""Freeze the held-out approximate-intent systems sample before GPU scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "speculation_traces"
if __package__:
    from .pre_roll_confidence import atomic_json, fit_counts, predict_with_confidence
    from speculation_traces.common import canonical_json_bytes, load_json, sha256_file
    from speculation_traces.speculation_frontier import (
        _load_episodes,
        canonical_strict_hash,
        construct_targets,
        decode_intent,
        fit_intent_thresholds,
        materialize_intent,
    )
else:
    sys.path.insert(0, str(TRACES))
    from common import canonical_json_bytes, load_json, sha256_file
    from pre_roll_confidence import atomic_json, fit_counts, predict_with_confidence
    from speculation_frontier import (
        _load_episodes,
        canonical_strict_hash,
        construct_targets,
        decode_intent,
        fit_intent_thresholds,
        materialize_intent,
    )


def _selection_rank(config_sha256: str, episode_id: int, index: int) -> str:
    payload = (
        config_sha256
        + "\0rtwm-v2-intent-pre-roll-heldout-1\0"
        + str(episode_id)
        + "\0"
        + str(index)
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_array(value: np.ndarray) -> list[Any]:
    return value.tolist()


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    config_sha = sha256_file(config_path)
    blocks_path = TRACES / "manifests" / "blocks.json"
    episodes = _load_episodes(load_json(blocks_path))
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    thresholds = fit_intent_thresholds(train)
    train_targets = construct_targets(train, thresholds)
    test_targets = construct_targets(test, thresholds)
    model = fit_counts(train_targets, "intent")
    episode_by_id = {episode.episode_id: episode for episode in test}

    candidates = []
    for target in test_targets:
        episode = episode_by_id[target.episode_id]
        upper = min(8, len(target.intent) - 4)
        for index in range(1, upper):
            prediction = predict_with_confidence(model, target.intent[:index])
            if prediction["prediction"] != target.intent[index]:
                continue
            keyboard, camera = materialize_intent(prediction["prediction"], thresholds)
            weights = np.uint32(1) << np.arange(23, dtype=np.uint32)
            keyboard_token = (
                keyboard.astype(np.uint32) * weights
            ).sum(axis=1, dtype=np.uint32)
            representative_hash = canonical_strict_hash(keyboard_token, camera)
            if representative_hash == target.strict_hash[index]:
                continue
            packed, yaw, pitch = decode_intent(target.intent[index])
            candidates.append(
                {
                    "rank": _selection_rank(config_sha, target.episode_id, index),
                    "episode_id": target.episode_id,
                    "block_index": index,
                    "confidence": prediction["confidence"],
                    "context_order": prediction["context_order"],
                    "intent": target.intent[index],
                    "movement_camera_group": f"{int(packed != 0)}:{yaw}:{pitch}",
                    "actual_strict_hash": target.strict_hash[index],
                    "representative_strict_hash": representative_hash,
                    "source_block_file_sha256": episode.source_sha256,
                    "prefix_and_continuation_keyboard": _json_array(
                        episode.keyboard[: index + 5]
                    ),
                    "prefix_and_continuation_camera_degrees": _json_array(
                        episode.camera_degrees[: index + 5]
                    ),
                    "representative_keyboard": _json_array(keyboard),
                    "representative_camera_degrees": _json_array(camera),
                }
            )

    count = int(config["held_out_selection"]["sample_count"])
    selected = []
    remaining = sorted(candidates, key=lambda row: row["rank"])
    seen_intents: set[str] = set()
    seen_groups: set[str] = set()
    while remaining and len(selected) < count:
        best = min(
            remaining,
            key=lambda row: (
                -(
                    int(row["intent"] not in seen_intents)
                    + int(row["movement_camera_group"] not in seen_groups)
                ),
                -int(row["intent"] not in seen_intents),
                -int(row["movement_camera_group"] not in seen_groups),
                row["rank"],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        seen_intents.add(best["intent"])
        seen_groups.add(best["movement_camera_group"])
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} qualifying decisions")
    movement_states = {
        row["movement_camera_group"].split(":")[0] for row in selected
    }
    camera_states = {
        ":".join(row["movement_camera_group"].split(":")[1:]) for row in selected
    }
    if len(seen_intents) < 3 or len(movement_states) < 2 or len(camera_states) < 2:
        raise RuntimeError("selected sample does not span camera/movement states")

    for ordinal, row in enumerate(selected):
        row["sample_id"] = ordinal
    report = {
        "schema_version": "rtwm-v2-intent-heldout-selection-1",
        "status": "frozen_before_gpu_scoring",
        "identity": {
            "config_sha256": config_sha,
            "blocks_manifest_sha256": sha256_file(blocks_path),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "training_only_intent_thresholds": thresholds,
        "candidate_count": len(candidates),
        "sample_count": len(selected),
        "coverage": {
            "distinct_intent_count": len(seen_intents),
            "movement_state_count": len(movement_states),
            "camera_state_count": len(camera_states),
            "movement_states": sorted(movement_states),
            "camera_states": sorted(camera_states),
        },
        "decisions": selected,
    }
    report["identity"]["selection_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "intent_pre_roll_config.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "manifests" / "intent_heldout_selection.json",
    )
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "sample_count": report["sample_count"],
                "coverage": report["coverage"],
                "selection_sha256": report["identity"]["selection_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
