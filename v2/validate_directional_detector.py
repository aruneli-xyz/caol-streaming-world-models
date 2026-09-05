"""Bounded held-out validation of transition-specific flow detectors.

The detector configuration is fixed before this script runs. Conditions at
change frames 48 and 96 form the calibration audit; frame 144 is held out.
No parameter is retuned from these outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


V2 = Path(__file__).resolve().parent
RTWM = V2.parent
DEFAULT_CONFIG = V2 / "config" / "confirmatory.json"
DEFAULT_OUTPUT = V2 / "results" / "directional_validation"

sys.path.insert(0, str(V2))

from artifacts import atomic_write_json  # noqa: E402
from flow_detector import DetectorConfig, flow_signals  # noqa: E402
from protocol import canonical_sha256, file_sha256, load_protocol  # noqa: E402
from score_detector import score_intrinsic, write_jsonl  # noqa: E402


def textured_frame(config: DetectorConfig, seed: int = 260819) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros(
        (config.resize_height, config.resize_width),
        dtype=np.uint8,
    )
    for x, y, value in zip(
        rng.integers(10, config.resize_width - 10, 500),
        rng.integers(10, config.resize_height - 10, 500),
        rng.integers(80, 256, 500),
    ):
        cv2.circle(frame, (int(x), int(y)), 2, int(value), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def translated(frame: np.ndarray, dx: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def scaled(
    frame: np.ndarray,
    scale: float,
    config: DetectorConfig,
) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        (config.radial_center_x, config.radial_center_y),
        0,
        scale,
    )
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def synthetic_checks(config: DetectorConfig) -> dict[str, bool]:
    frame = textured_frame(config)
    positive_horizontal = flow_signals(
        [frame, translated(frame, 2)],
        config,
    )["horizontal"][0]
    expansion = flow_signals(
        [frame, scaled(frame, 1.04, config)],
        config,
    )["radial"][0]
    contraction = flow_signals(
        [frame, scaled(frame, 0.96, config)],
        config,
    )["radial"][0]
    return {
        "positive_translation_has_positive_horizontal_flow": bool(
            positive_horizontal > 0.5
        ),
        "expansion_has_positive_radial_flow": bool(expansion > 0.1),
        "contraction_has_negative_radial_flow": bool(contraction < -0.1),
    }


def directional_delta(
    signal: np.ndarray,
    change_frame: int,
) -> float:
    baseline = signal[max(0, change_frame - 16):max(0, change_frame - 2)]
    post = signal[
        change_frame + 4:min(len(signal), change_frame + 20)
    ]
    if not baseline.size or not post.size:
        return float("nan")
    return float(np.median(post) - np.median(baseline))


def summarize_split(
    rows: list[dict[str, Any]],
    signals: dict[str, np.ndarray],
    transition: str,
    frames: set[int],
    config: DetectorConfig,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["transition"] == transition
        and int(row["change_frame"]) in frames
        and row["detector"] == "v2_preonly_transition_aware"
    ]
    spec = config.transitions[transition]
    signal_name = str(spec["signal"])
    direction = str(spec["direction"])
    deltas: list[float] = []
    for row in selected:
        key = f"{row['sample']}__v{row['view_index']}__{signal_name}"
        deltas.append(
            directional_delta(signals[key], int(row["change_frame"]))
        )
    finite = [value for value in deltas if np.isfinite(value)]
    agreements = [
        value > 0 if direction == "rise" else value < 0
        for value in finite
    ]
    detected = sum(row["status"] == "detected" for row in selected)
    return {
        "transition": transition,
        "signal": signal_name,
        "expected_direction": direction,
        "change_frames": sorted(frames),
        "n_views": len(selected),
        "n_detected": detected,
        "detection_fraction": detected / len(selected) if selected else None,
        "finite_directional_deltas": len(finite),
        "expected_sign_fraction": (
            float(np.mean(agreements)) if agreements else None
        ),
        "median_post_minus_pre": (
            float(np.median(finite)) if finite else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    config = DetectorConfig.from_json(args.config)
    source_manifest_path = RTWM / "results" / "a2e" / "manifest.json"
    source_items = json.loads(source_manifest_path.read_text())
    video_artifacts = []
    for item in source_items:
        video = (
            RTWM
            / "results"
            / "a2e"
            / "out"
            / item["sample"]
            / item["sample"]
            / "generated.mp4"
        )
        if video.exists():
            video_artifacts.append(
                {
                    "sample": item["sample"],
                    "path": str(video),
                    "sha256": file_sha256(video),
                }
            )
    signal_store: dict[str, np.ndarray] = {}
    view_rows, pair_rows = score_intrinsic(config, signal_store)
    calibration_frames = {48, 96}
    validation_frames = {144}
    checks = synthetic_checks(config)

    transitions: dict[str, Any] = {}
    for transition in ("reverse", "left"):
        calibration = summarize_split(
            view_rows,
            signal_store,
            transition,
            calibration_frames,
            config,
        )
        validation = summarize_split(
            view_rows,
            signal_store,
            transition,
            validation_frames,
            config,
        )
        supported = bool(
            all(checks.values())
            and validation["n_views"] >= 4
            and validation["detection_fraction"] is not None
            and validation["detection_fraction"] >= 0.75
            and validation["expected_sign_fraction"] is not None
            and validation["expected_sign_fraction"] >= 0.75
        )
        transitions[transition] = {
            "calibration_audit": calibration,
            "held_out_validation": validation,
            "supported_for_response_claims": supported,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "view_results.jsonl", view_rows)
    write_jsonl(args.output / "pair_results.jsonl", pair_rows)
    np.savez_compressed(args.output / "signals.npz", **signal_store)
    report = {
        "schema_version": "rtwm-v2-directional-validation-1",
        **protocol.identity,
        "protocol_status": (
            "retrospective bounded validation; split and decision rule were "
            "fixed before this rerun but were not preregistered"
        ),
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": file_sha256(source_manifest_path),
        },
        "video_artifacts": video_artifacts,
        "split": {
            "calibration_change_frames": sorted(calibration_frames),
            "held_out_change_frames": sorted(validation_frames),
            "unit": "view; paired views are not independent seeds",
        },
        "decision_rule": {
            "minimum_held_out_views": 4,
            "minimum_detection_fraction": 0.75,
            "minimum_expected_sign_fraction": 0.75,
            "no_outcome_based_retuning": True,
        },
        "synthetic_checks": checks,
        "transitions": transitions,
        "claim_scope": (
            "Restrict confirmatory causal claims to forward-to-stop unless a "
            "directional transition passes the held-out rule."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output / "validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
