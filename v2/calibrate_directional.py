"""Fit fresh canonical null thresholds and validate seed 104 before test scoring."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import artifact_metadata, verify_rollout_artifacts  # noqa: E402
from directional_study import (  # noqa: E402
    action_protocol_sha256,
    detector_config,
    load_directional_protocol,
)
from flow_detector import first_sustained_strict_crossing, paired_flow_field_signals  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def load_frames(root: Path, record: dict[str, Any]) -> np.ndarray:
    paths = verify_rollout_artifacts(
        root, record, required=("latent", "decoded_u8", "detector_frames", "action_tensors")
    )
    with np.load(paths["detector_frames"]) as payload:
        return np.asarray(payload["views"])


def sustained_statistics(
    signal: np.ndarray, anchors: list[int], window: int, persistence: int
) -> list[dict[str, Any]]:
    rows = []
    for anchor in anchors:
        end = min(len(signal), anchor + window)
        values = [
            float(np.min(signal[index:index + persistence]))
            for index in range(anchor, max(anchor, end - persistence + 1))
            if signal[index:index + persistence].size == persistence
            and np.isfinite(signal[index:index + persistence]).all()
        ]
        if not values:
            raise RuntimeError(f"no finite null windows at anchor {anchor}")
        rows.append({
            "anchor": anchor,
            "search_end": end,
            "maximum_sustained_min": max(values),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "results" / "directional_d0" / "manifest.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "directional_d0")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "directional_d0_calibration")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable calibration: {args.output}")

    protocol = load_directional_protocol(args.config)
    manifest = json.loads(args.manifest.read_text())
    canonical_hash = action_protocol_sha256(protocol)
    if (
        manifest.get("config_sha256") != protocol.file_sha256
        or manifest.get("protocol_sha256") != protocol.canonical_sha256
        or manifest.get("action_protocol_sha256") != canonical_hash
        or manifest.get("planned_rollouts") != 52
    ):
        raise RuntimeError("fresh canonical manifest identity mismatch")
    records = {
        (row["stage"], row["scene"], int(row["seed"]), row["arm"]): row
        for row in manifest["rollouts"]
        if row.get("status") == "complete"
    }
    config = detector_config(protocol)
    null = protocol.data["null_calibration"]
    anchors = [int(value) for value in null["anchors"]]
    window = int(null["window_frames"])
    persistence = int(null["persistence"])
    scenes = protocol.data["design"]["scenes"]
    calibration_seeds = protocol.data["design"]["fresh_grid"]["calibration"]["seeds"]
    validation_seeds = protocol.data["design"]["fresh_grid"]["validation"]["seeds"]
    locks: dict[str, dict[str, Any]] = {}
    validation_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    for scene in scenes:
        calibration_signals: dict[int, list[np.ndarray]] = {0: [], 1: []}
        bindings = []
        for seed in calibration_seeds:
            first = records[("calibration", scene, int(seed), "null_a")]
            second = records[("calibration", scene, int(seed), "null_b")]
            for record in (first, second):
                if record.get("action_protocol_sha256") != canonical_hash:
                    raise RuntimeError("null record action protocol mismatch")
                evidence.extend(record["artifacts"].values())
            first_frames, second_frames = load_frames(args.root, first), load_frames(args.root, second)
            for view in (0, 1):
                paired = paired_flow_field_signals(
                    list(first_frames[view]), list(second_frames[view]), config
                )["flow_field_l2"]
                calibration_signals[view].append(paired)
                bindings.append({
                    "scene": scene, "seed": seed, "view_index": view,
                    "null_a": first["rollout_id"], "null_b": second["rollout_id"],
                    "statistics": sustained_statistics(paired, anchors, window, persistence),
                })
        for view in (0, 1):
            selected = [
                row["maximum_sustained_min"]
                for binding in bindings if binding["view_index"] == view
                for row in binding["statistics"]
            ]
            threshold = float(max(selected))
            payload = {
                "schema_version": "rtwm-v2-directional-null-lock-1",
                **protocol.identity,
                "action_protocol_sha256": canonical_hash,
                "scene": scene,
                "view_index": view,
                "threshold": threshold,
                "statistic": null["statistic"],
                "anchors": anchors,
                "window_frames": window,
                "persistence": persistence,
                "calibration_seeds": calibration_seeds,
                "bindings": [row for row in bindings if row["view_index"] == view],
            }
            payload["payload_sha256"] = canonical_sha256(payload)
            locks[f"{scene}__view{view}.json"] = payload

        for seed in validation_seeds:
            first = records[("validation", scene, int(seed), "null_a")]
            second = records[("validation", scene, int(seed), "null_b")]
            first_frames, second_frames = load_frames(args.root, first), load_frames(args.root, second)
            for view in (0, 1):
                signal = paired_flow_field_signals(
                    list(first_frames[view]), list(second_frames[view]), config
                )["flow_field_l2"]
                threshold = float(locks[f"{scene}__view{view}.json"]["threshold"])
                crossings = []
                for anchor in anchors:
                    crossing = first_sustained_strict_crossing(
                        signal, threshold, "rise", anchor,
                        min(len(signal), anchor + window), persistence,
                    )
                    crossings.append(crossing)
                validation_rows.append({
                    "scene": scene, "seed": seed, "view_index": view,
                    "threshold": threshold, "crossings": crossings,
                    "zero_crossings": all(value is None for value in crossings),
                })

    checks = {
        "all_12_calibration_rollouts_present": sum(
            key[0] == "calibration" for key in records
        ) == 12,
        "all_4_validation_rollouts_present": sum(
            key[0] == "validation" for key in records
        ) == 4,
        "four_scene_view_locks": len(locks) == 4,
        "validation_zero_strict_crossings": all(row["zero_crossings"] for row in validation_rows),
        "canonical_action_protocol_bound": all(
            row.get("action_protocol_sha256") == canonical_hash
            for row in manifest["rollouts"]
            if row.get("stage") in {"calibration", "validation"}
        ),
    }
    block_reasons = sorted(name for name, passed in checks.items() if not passed)
    gate = {
        "schema_version": "rtwm-v2-directional-null-gate-1",
        **protocol.identity,
        "action_protocol_sha256": canonical_hash,
        "manifest_sha256": file_sha256(args.manifest),
        "passed": not block_reasons,
        "checks": checks,
        "block_reasons": block_reasons,
        "validation": validation_rows,
        "lock_payload_sha256": {
            name: payload["payload_sha256"] for name, payload in locks.items()
        },
    }
    gate["gate_sha256"] = canonical_sha256(gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (temporary / "locks").mkdir()
        for name, payload in locks.items():
            path = temporary / "locks" / name
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            path.chmod(0o444)
        gate_path = temporary / "gate_report.json"
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        gate_path.chmod(0o444)
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(gate, indent=2))
    if block_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
