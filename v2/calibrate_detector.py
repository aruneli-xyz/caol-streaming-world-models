"""Fit and lock paired-detector thresholds using null duplicate rollouts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from artifacts import record_artifacts, verify_artifact
from flow_detector import (
    DetectorConfig,
    PairedDetectorConfig,
    first_sustained_strict_crossing,
    paired_flow_field_signals,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "confirmatory.json"
DEFAULT_OUTPUT = HERE / "results" / "paired_calibration"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def records_from_manifest(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        records = manifest
    elif isinstance(manifest, dict) and isinstance(manifest.get("rollouts"), list):
        records = manifest["rollouts"]
    else:
        raise ValueError("manifest must be a rollout list or contain a rollout list")
    return [
        dict(record)
        for record in records
        if record.get("status", "complete") == "complete"
    ]


def fit_null_threshold(
    signals: list[np.ndarray],
    config: PairedDetectorConfig,
) -> float:
    """Fit a threshold from null values without consulting interventions."""
    if not signals:
        raise ValueError("no null signals supplied for threshold fitting")
    values = np.concatenate(
        [np.asarray(signal, dtype=np.float64).reshape(-1) for signal in signals]
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("null calibration signals must be nonempty and finite")
    if config.threshold_statistic == "maximum":
        return float(np.max(values))
    return float(
        np.quantile(values, config.threshold_quantile, method="higher")
    )


def null_window_statistics(
    signal: np.ndarray,
    anchors: list[int],
    window: int,
    persistence: int,
) -> list[dict[str, Any]]:
    """Maximum sustained-min statistic at predeclared pseudo-admission anchors."""
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        end = min(len(signal), anchor + window)
        values = []
        for index in range(anchor, max(anchor, end - persistence + 1)):
            chunk = signal[index:index + persistence]
            if chunk.size == persistence and np.isfinite(chunk).all():
                values.append(float(np.min(chunk)))
        if not values:
            raise ValueError(f"no finite persistence windows at anchor {anchor}")
        rows.append(
            {
                "anchor": anchor,
                "search_end": end,
                "maximum_sustained_min": max(values),
            }
        )
    return rows


def validate_protocol(protocol: dict[str, Any]) -> tuple[list[str], list[int], int, list[int], list[str]]:
    confirmatory = protocol["confirmatory"]
    scenes = [str(scene) for scene in confirmatory["scenes"]]
    calibration_seeds = [
        int(seed) for seed in confirmatory["calibration"]["seeds"]
    ]
    validation_seeds = [
        int(seed) for seed in confirmatory["validation"]["seeds"]
    ]
    test_seeds = [int(seed) for seed in confirmatory["test"]["seeds"]]
    null_arms = [
        str(arm) for arm in confirmatory["calibration"]["arms"]
    ]
    if len(scenes) != 2 or len(set(scenes)) != 2:
        raise ValueError("confirmatory protocol requires exactly two scenes")
    if calibration_seeds != [101, 102, 103]:
        raise ValueError("calibration seeds must be exactly 101, 102, and 103")
    if validation_seeds != [104]:
        raise ValueError("validation seed must be exactly 104")
    if test_seeds != [201, 202, 203, 204, 205, 206]:
        raise ValueError("test seeds must be exactly 201 through 206")
    if set(calibration_seeds) & set(validation_seeds + test_seeds):
        raise ValueError("calibration, validation, and test seeds must be disjoint")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("validation and test seeds must be disjoint")
    if len(null_arms) != 2 or len(set(null_arms)) != 2:
        raise ValueError("exactly two distinct null duplicate arms are required")
    null_commands = confirmatory["calibration"]["commands"]
    if any(
        null_commands.get(arm) != {"pre": "forward", "post": "forward"}
        for arm in null_arms
    ):
        raise ValueError("calibration arms must both be unchanged-forward nulls")
    validation_arms = [
        str(arm) for arm in confirmatory["validation"]["arms"]
    ]
    if validation_arms != null_arms:
        raise ValueError("validation must use the same two null duplicate arms")
    return scenes, calibration_seeds, validation_seeds[0], test_seeds, null_arms


def index_required_nulls(
    records: list[dict[str, Any]],
    scenes: list[str],
    seeds: list[int],
    arms: list[str],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    required = {
        (scene, seed, arm)
        for scene in scenes
        for seed in seeds
        for arm in arms
    }
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        scene = str(record.get("scene"))
        arm = str(record.get("arm"))
        if scene not in scenes or arm not in arms:
            continue
        try:
            seed = int(record.get("seed"))
        except (TypeError, ValueError):
            continue
        key = (
            scene,
            seed,
            arm,
        )
        if key not in required:
            continue
        if key in indexed:
            raise ValueError(f"duplicate manifest record for {key}")
        if record.get("pre_command", "forward") != "forward":
            raise ValueError(f"null input {key} has a non-forward pre command")
        if record.get("post_command", "forward") != "forward":
            raise ValueError(f"intervention record cannot calibrate detector: {key}")
        indexed[key] = record
    missing = sorted(required - set(indexed))
    if missing:
        raise ValueError(f"missing required null duplicate records: {missing}")
    return indexed


def bind_detector_frames(
    record: dict[str, Any],
    root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    artifacts = record_artifacts(record)
    if "detector_frames" not in artifacts:
        raise ValueError("every calibration input requires detector_frames")
    path = verify_artifact(root, artifacts["detector_frames"], label="detector_frames")
    with np.load(path) as payload:
        if "views" not in payload:
            raise ValueError(f"detector frame artifact lacks views: {path}")
        views = np.asarray(payload["views"])
    if views.dtype != np.uint8 or views.ndim != 4 or views.shape[0] != 2:
        raise ValueError(f"invalid detector frame payload: {path} {views.shape} {views.dtype}")
    binding = {
        "rollout_id": record.get("rollout_id"),
        "scene": str(record["scene"]),
        "seed": int(record["seed"]),
        "arm": str(record["arm"]),
        "detector_frames_path": artifacts["detector_frames"]["path"],
        "detector_frames_sha256": artifacts["detector_frames"]["sha256"],
    }
    return views, binding


def add_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def write_immutable_bundle(
    output: Path,
    gate_report: dict[str, Any],
    locks: dict[str, dict[str, Any]],
) -> None:
    """Atomically create a read-only calibration bundle and never overwrite."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    report_path = temporary / "gate_report.json"
    try:
        (temporary / "locks").mkdir()
        report_path.write_text(
            json.dumps(gate_report, indent=2, sort_keys=True) + "\n"
        )
        report_path.chmod(0o444)
        for name, payload in locks.items():
            lock_path = temporary / "locks" / name
            lock_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            lock_path.chmod(0o444)
        (temporary / "locks").chmod(0o555)
        temporary.chmod(0o555)
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o755)
            locks_dir = temporary / "locks"
            if locks_dir.exists():
                locks_dir.chmod(0o755)
                for path in locks_dir.iterdir():
                    path.chmod(0o644)
            report_path = temporary / "gate_report.json"
            if report_path.exists():
                report_path.chmod(0o644)
            shutil.rmtree(temporary)
        raise


def calibrate(
    config_path: Path,
    manifest_path: Path,
    root: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    detector_config = DetectorConfig.from_json(config_path)
    paired_config = PairedDetectorConfig.from_json(config_path)
    if paired_config.signal != "flow_field_l2":
        raise ValueError("calibration requires the direct flow_field_l2 signal")
    scenes, calibration_seeds, validation_seed, test_seeds, null_arms = (
        validate_protocol(protocol)
    )
    records = records_from_manifest(manifest)
    indexed = index_required_nulls(
        records,
        scenes,
        calibration_seeds + [validation_seed],
        null_arms,
    )
    expected_frames = int(protocol["validation"]["expected_frames"])
    minimum_finite = float(
        protocol["validation"]["minimum_finite_flow_fraction"]
    )
    threshold_fit = protocol["paired_detector"]["threshold_fit"]
    anchors = [int(value) for value in threshold_fit["anchors"]]
    calibration_window = int(threshold_fit["window"])
    config_hash = sha256_file(config_path)
    manifest_hash = sha256_file(manifest_path)

    frame_cache: dict[tuple[str, int, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    bindings: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key, record in indexed.items():
        views, binding = bind_detector_frames(record, root)
        if views.shape[1] != expected_frames:
            raise ValueError(
                f"unexpected frame count for {binding['detector_frames_path']}: "
                f"{views.shape[1]} != {expected_frames}"
            )
        frame_cache[key] = (list(views[0]), list(views[1]))
        bindings[key] = binding

    thresholds: dict[tuple[str, int], float] = {}
    calibration_details: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for scene in scenes:
        for view in (0, 1):
            null_signals: list[np.ndarray] = []
            details: list[dict[str, Any]] = []
            for seed in calibration_seeds:
                first_key = (scene, seed, null_arms[0])
                second_key = (scene, seed, null_arms[1])
                paired = paired_flow_field_signals(
                    frame_cache[first_key][view],
                    frame_cache[second_key][view],
                    detector_config,
                )
                finite_min = (
                    float(np.min(paired["finite_fraction"]))
                    if paired["finite_fraction"].size
                    else 0.0
                )
                if finite_min < minimum_finite:
                    raise ValueError(
                        f"insufficient finite flow for {scene} seed {seed} view {view}"
                    )
                anchor_statistics = null_window_statistics(
                    paired[paired_config.signal],
                    anchors,
                    calibration_window,
                    paired_config.persistence,
                )
                null_signals.append(
                    np.asarray(
                        [row["maximum_sustained_min"] for row in anchor_statistics],
                        dtype=np.float64,
                    )
                )
                details.append(
                    {
                        "seed": seed,
                        "sample_count": int(paired[paired_config.signal].size),
                        "maximum": float(np.max(paired[paired_config.signal])),
                        "anchor_statistics": anchor_statistics,
                        "minimum_finite_fraction": finite_min,
                    }
                )
            thresholds[(scene, view)] = fit_null_threshold(
                null_signals, paired_config
            )
            calibration_details[(scene, view)] = details

    validation_results: list[dict[str, Any]] = []
    validation_passed = True
    for scene in scenes:
        for view in (0, 1):
            first_key = (scene, validation_seed, null_arms[0])
            second_key = (scene, validation_seed, null_arms[1])
            paired = paired_flow_field_signals(
                frame_cache[first_key][view],
                frame_cache[second_key][view],
                detector_config,
            )
            signal = paired[paired_config.signal]
            finite_min = (
                float(np.min(paired["finite_fraction"]))
                if paired["finite_fraction"].size
                else 0.0
            )
            crossings = [
                {
                    "anchor": anchor,
                    "crossing_index": first_sustained_strict_crossing(
                        signal,
                        thresholds[(scene, view)],
                        paired_config.direction,
                        anchor,
                        min(len(signal), anchor + calibration_window),
                        paired_config.persistence,
                    ),
                }
                for anchor in anchors
            ]
            passed = (
                all(row["crossing_index"] is None for row in crossings)
                and finite_min >= minimum_finite
            )
            validation_passed = validation_passed and passed
            validation_results.append(
                {
                    "scene": scene,
                    "seed": validation_seed,
                    "view_index": view,
                    "threshold": thresholds[(scene, view)],
                    "strict_crossings": crossings,
                    "maximum": float(np.max(signal)),
                    "minimum_finite_fraction": finite_min,
                    "passed": passed,
                }
            )

    common_binding = {
        "schema_version": "rtwm-v2-paired-calibration-lock-1",
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "signal": paired_config.signal,
        "direction": paired_config.direction,
        "crossing_operator": "strict",
        "persistence": paired_config.persistence,
        "threshold_statistic": paired_config.threshold_statistic,
        "threshold_quantile": paired_config.threshold_quantile,
        "anchors": anchors,
        "calibration_window": calibration_window,
        "calibration_seeds": calibration_seeds,
        "validation_seed": validation_seed,
        "excluded_test_seeds": test_seeds,
    }
    locks: dict[str, dict[str, Any]] = {}
    if validation_passed:
        for scene in scenes:
            for view in (0, 1):
                source_keys = [
                    (scene, seed, arm)
                    for seed in calibration_seeds + [validation_seed]
                    for arm in null_arms
                ]
                payload = {
                    **common_binding,
                    "scene": scene,
                    "view_index": view,
                    "threshold": thresholds[(scene, view)],
                    "calibration_details": calibration_details[(scene, view)],
                    "validation": next(
                        result
                        for result in validation_results
                        if result["scene"] == scene
                        and result["view_index"] == view
                    ),
                    "source_videos": [bindings[key] for key in source_keys],
                }
                locks[f"{scene}__view{view}.json"] = add_payload_hash(payload)

    lock_index = {
        name: payload["payload_sha256"]
        for name, payload in sorted(locks.items())
    }
    gate_payload = {
        "schema_version": "rtwm-v2-paired-calibration-gate-1",
        "passed": validation_passed,
        "checks": {
            "protocol_partition_exact": True,
            "calibration_uses_null_duplicates_only": True,
            "all_input_hashes_match": True,
            "all_frame_counts_match": True,
            "seed104_has_zero_strict_crossings": validation_passed,
        },
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "calibration_seeds": calibration_seeds,
        "validation_seed": validation_seed,
        "excluded_test_seeds": test_seeds,
        "validation_results": validation_results,
        "lock_payload_sha256": lock_index,
    }
    gate_report = add_payload_hash(gate_payload)
    write_immutable_bundle(output, gate_report, locks)
    if not validation_passed:
        raise RuntimeError(
            f"seed 104 validation failed; no calibration locks written ({output})"
        )
    return gate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="root for manifest-relative video paths (default: manifest directory)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.manifest.parent if args.root is None else args.root
    report = calibrate(args.config, args.manifest, root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"locked paired detector calibration -> {args.output}")


if __name__ == "__main__":
    main()
