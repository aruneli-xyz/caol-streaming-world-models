"""Validity-first scoring for the locked two-scene confirmatory experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from artifacts import atomic_write_json, record_artifacts, verify_rollout_artifacts
from flow_detector import (
    DetectorConfig,
    PairedDetectorConfig,
    aggregate_pair,
    detect_calibrated_paired_divergence,
    paired_flow_field_signals,
)
from gates import require_confirmatory_gate
from protocol import canonical_sha256, file_sha256, load_protocol


HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def load_artifacts(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    paths = verify_rollout_artifacts(root, record)
    latent = torch.load(paths["latent"], map_location="cpu")
    decoded = np.load(paths["decoded_u8"], mmap_mode="r")
    with np.load(paths["detector_frames"]) as payload:
        detector = np.asarray(payload["views"])
    return {
        "paths": paths,
        "latent": latent,
        "decoded": decoded,
        "detector": detector,
    }


def split_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[2] % 2:
        raise ValueError(f"invalid latent shape {list(latent.shape)}")
    batch, channels, combined, height, width = latent.shape
    return latent.reshape(batch, channels, 2, combined // 2, height, width)


def raw_view(decoded: np.ndarray, view: int) -> np.ndarray:
    if decoded.ndim != 4 or decoded.shape[2] % 2:
        raise ValueError(f"invalid decoded shape {list(decoded.shape)}")
    width = decoded.shape[2] // 2
    return np.asarray(decoded[:, :, view * width:(view + 1) * width])


def rf_effect_starts(rf_manifest_path: Path) -> dict[tuple[int, int], int]:
    manifest = load_json(rf_manifest_path)
    artifact = manifest["artifacts"]["hybrid_results"]
    hybrid_path = rf_manifest_path.parent / artifact["path"]
    rows = read_jsonl(hybrid_path)
    starts: dict[tuple[int, int], int] = {}
    for row in rows:
        index = int(row["latent_index"])
        for view in row["views"]:
            if view["early_changed_frame_count"] != 0:
                raise RuntimeError(f"RF probe t={index} has pre-support raw effects")
            starts[(index, int(view["view_index"]))] = int(
                view["earliest_changed_frame"]
            )
    return starts


def load_thresholds(calibration: Path) -> dict[tuple[str, int], float]:
    thresholds = {}
    for lock_path in (calibration / "locks").glob("*.json"):
        lock = load_json(lock_path)
        thresholds[(str(lock["scene"]), int(lock["view_index"]))] = float(
            lock["threshold"]
        )
    if len(thresholds) != 4:
        raise RuntimeError("expected four scene/view calibration locks")
    return thresholds


def require_protocol_identity(
    label: str,
    payload: dict[str, Any],
    protocol: Any,
) -> None:
    for key, expected in protocol.identity.items():
        actual = payload.get(key)
        if actual != expected:
            raise RuntimeError(
                f"{label} {key} mismatch: {actual!r} != {expected!r}"
            )


def require_gate_bound_artifact(
    gate: dict[str, Any],
    gate_path: Path,
    artifact_path: Path,
) -> None:
    target = artifact_path.resolve()
    for metadata in gate.get("evidence_artifacts", []):
        recorded = Path(metadata["path"])
        if not recorded.is_absolute():
            recorded = gate_path.parent / recorded
        if recorded.resolve() != target:
            continue
        actual = file_sha256(target)
        if actual != metadata.get("sha256"):
            raise RuntimeError(
                f"gate-bound artifact hash mismatch for {target}: "
                f"{actual!r} != {metadata.get('sha256')!r}"
            )
        return
    raise RuntimeError(f"artifact is not bound by confirmatory gate: {target}")


def validate_manifest_identity(
    manifest: dict[str, Any],
    protocol: Any,
) -> None:
    require_protocol_identity("manifest", manifest, protocol)
    data = protocol.data
    expected = {
        "experiment": data["experiment"],
        "checkpoint": data["model"]["checkpoint"],
        "n_frames": int(data["model"]["n_frames"]),
        "fps": int(data["model"]["fps"]),
        "latent_stride_pixels": int(data["model"]["latent_stride_pixels"]),
        "nfpb": int(data["model"]["latent_frames_per_block"]),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"manifest {key} mismatch: {manifest.get(key)!r} != {value!r}"
            )
    source = manifest.get("gamma_source", {})
    expected_commit = data["model"]["source_commit"]
    if source.get("commit") != expected_commit:
        raise RuntimeError(
            "manifest source commit mismatch: "
            f"{source.get('commit')!r} != {expected_commit!r}"
        )
    if source.get("dirty") and not source.get("diff_sha256"):
        raise RuntimeError("dirty source manifest lacks diff_sha256 provenance")


def signal_window_statistics(
    signal: np.ndarray,
    start: int,
    end: int,
    threshold: float,
) -> dict[str, float | int | None]:
    values = np.asarray(signal[max(0, start):min(len(signal), end)], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "signal_window_count": 0,
            "signal_peak": None,
            "signal_mean": None,
            "signal_auc": None,
            "signal_peak_above_threshold": None,
        }
    peak = float(np.max(values))
    return {
        "signal_window_count": int(values.size),
        "signal_peak": peak,
        "signal_mean": float(np.mean(values)),
        "signal_auc": float(np.sum(values)),
        "signal_peak_above_threshold": float(peak - threshold),
    }


def support_aligned_search_range(
    effect_start_frame: int,
    horizon: int,
) -> tuple[int, int]:
    """Map first affected frame F to eligible flow indices starting at F-1."""
    if effect_start_frame < 0 or horizon <= 0:
        raise ValueError("effect start must be nonnegative and horizon positive")
    return max(0, effect_start_frame - 1), effect_start_frame + horizon


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    samples: int = 20_000,
) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not finite.size:
        return {"n_seeds": 0, "mean": None, "ci95": [None, None]}
    if finite.size == 1:
        value = float(finite[0])
        return {"n_seeds": 1, "mean": value, "ci95": [value, value]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    bootstrapped = finite[indices].mean(axis=1)
    return {
        "n_seeds": int(finite.size),
        "mean": float(finite.mean()),
        "ci95": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
    }


def seed_level_statistics(
    pair_rows: list[dict[str, Any]],
    *,
    change_frame: int,
    n_frames: int,
) -> dict[str, Any]:
    valid = [row for row in pair_rows if row.get("pair_valid")]
    delays = sorted({str(row["delay"]) for row in valid})
    per_delay: dict[str, Any] = {}
    seed_metrics: dict[str, dict[int, dict[str, float]]] = {}
    for delay_index, delay in enumerate(delays):
        selected = [row for row in valid if row["delay"] == delay]
        seeds = sorted({int(row["seed"]) for row in selected})
        by_seed: dict[int, dict[str, float]] = {}
        for seed in seeds:
            rows = [row for row in selected if int(row["seed"]) == seed]
            both_lags = [
                float(row["both_detected_onset_mean"]) - change_frame
                for row in rows
                if row.get("both_detected_onset_mean") is not None
            ]
            by_seed[seed] = {
                "both_view_detection_fraction": float(
                    np.mean([row["pair_status"] == "both" for row in rows])
                ),
                "any_view_detection_fraction": float(
                    np.mean([int(row["detected_count"]) > 0 for row in rows])
                ),
                "causal_onset_lag_frames": (
                    float(np.mean(both_lags))
                    if len(both_lags) == len(rows)
                    else float("nan")
                ),
                "flow_divergence_peak": float(
                    np.mean([row["pair_signal_peak_mean"] for row in rows])
                ),
                "effective_generation_fps": float(
                    np.mean(
                        [
                            n_frames / float(row["intervention_total_s"])
                            for row in rows
                        ]
                    )
                ),
            }
        seed_metrics[delay] = by_seed
        metric_rows: dict[str, Any] = {}
        for metric_index, metric in enumerate(next(iter(by_seed.values()), {})):
            metric_rows[metric] = bootstrap_mean_ci(
                [values[metric] for values in by_seed.values()],
                seed=20260819 + 100 * delay_index + metric_index,
            )
        per_delay[delay] = {
            "seed_count": len(by_seed),
            "scene_pair_count": len(selected),
            "metrics": metric_rows,
        }

    paired: dict[str, Any] = {}
    if {"d0", "d1"}.issubset(seed_metrics):
        common = sorted(set(seed_metrics["d0"]) & set(seed_metrics["d1"]))
        for metric_index, metric in enumerate(
            next(iter(seed_metrics["d0"].values()), {})
        ):
            differences = []
            for seed in common:
                first = seed_metrics["d0"][seed][metric]
                second = seed_metrics["d1"][seed][metric]
                if math.isfinite(first) and math.isfinite(second):
                    differences.append(second - first)
            paired[f"d1_minus_d0_{metric}"] = bootstrap_mean_ci(
                differences,
                seed=20261819 + metric_index,
            )
        d0_fps = per_delay["d0"]["metrics"]["effective_generation_fps"]["mean"]
        d1_fps = per_delay["d1"]["metrics"]["effective_generation_fps"]["mean"]
        pooled = (d0_fps + d1_fps) / 2 if d0_fps is not None and d1_fps is not None else None
        relative = (
            abs(d1_fps - d0_fps) / pooled
            if pooled is not None and pooled > 0
            else None
        )
        paired["throughput_tolerance_fraction"] = 0.05
        paired["relative_throughput_difference"] = relative
        paired["similar_throughput"] = relative is not None and relative <= 0.05

    return {
        "schema_version": "rtwm-v2-confirmatory-statistics-1",
        "statistical_unit": "seed; scenes and views are repeated observations",
        "bootstrap": {
            "samples": 20_000,
            "interval": "percentile 95%",
            "missing_onsets": (
                "seed onset is reported only when both views detect in every "
                "planned scene; reliability is reported separately"
            ),
        },
        "by_delay": per_delay,
        "paired_delay_comparison": paired,
    }


def invalidate(result: dict[str, Any], reasons: list[str]) -> None:
    result.update(
        status="invalid",
        onset_signal_index=None,
        onset_frame=None,
        lag_from_change=None,
        offset_from_admission=None,
        invalid_reasons=sorted(set(reasons)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=HERE / "config" / "confirmatory.json"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rf-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-patch",
        type=Path,
        default=HERE / "results" / "provenance" / "gamma_world_source.patch",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "confirmatory_scoring"
    )
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    gate_report = require_confirmatory_gate(args.gate, protocol)
    detector_config = DetectorConfig.from_json(args.config)
    paired_config = PairedDetectorConfig.from_json(args.config)
    require_gate_bound_artifact(
        gate_report,
        args.gate,
        args.calibration / "gate_report.json",
    )
    for lock_path in sorted((args.calibration / "locks").glob("*.json")):
        require_gate_bound_artifact(gate_report, args.gate, lock_path)
    require_gate_bound_artifact(gate_report, args.gate, args.rf_manifest)
    thresholds = load_thresholds(args.calibration)
    effect_starts = rf_effect_starts(args.rf_manifest)
    manifest = load_json(args.manifest)
    validate_manifest_identity(manifest, protocol)
    source_patch_sha256 = file_sha256(args.source_patch)
    expected_patch_sha256 = manifest["gamma_source"]["diff_sha256"]
    if source_patch_sha256 != expected_patch_sha256:
        raise RuntimeError(
            "archived source patch mismatch: "
            f"{source_patch_sha256!r} != {expected_patch_sha256!r}"
        )
    calibration_gate = load_json(args.calibration / "gate_report.json")
    if calibration_gate.get("config_sha256") != protocol.file_sha256:
        raise RuntimeError(
            "calibration gate config mismatch: "
            f"{calibration_gate.get('config_sha256')!r} != "
            f"{protocol.file_sha256!r}"
        )
    if not calibration_gate.get("passed"):
        raise RuntimeError("calibration gate did not pass")
    rf_manifest = load_json(args.rf_manifest)
    rf_config_sha256 = (
        rf_manifest.get("source", {}).get("config", {}).get("sha256")
    )
    if rf_config_sha256 != protocol.file_sha256:
        raise RuntimeError(
            "RF manifest config mismatch: "
            f"{rf_config_sha256!r} != {protocol.file_sha256!r}"
        )
    rf_source_commit = (
        rf_manifest.get("source", {}).get("gamma_world", {}).get("commit")
    )
    if rf_source_commit != protocol.data["model"]["source_commit"]:
        raise RuntimeError(
            "RF manifest source commit mismatch: "
            f"{rf_source_commit!r} != "
            f"{protocol.data['model']['source_commit']!r}"
        )
    records = [
        record
        for record in manifest["rollouts"]
        if record.get("split") == "test" and record.get("status") == "complete"
    ]
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record["scene"], int(record["seed"]), record["arm"])
        if key in indexed:
            raise RuntimeError(f"duplicate complete rollout record: {key}")
        indexed[key] = record
    expected_pairs = 0
    view_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for scene in protocol.data["confirmatory"]["scenes"]:
        for seed in protocol.data["confirmatory"]["test"]["seeds"]:
            control_record = indexed.get((scene, int(seed), "control"))
            for delay in protocol.data["confirmatory"]["test"]["delays"]:
                expected_pairs += 1
                delay_name = str(delay["name"])
                stop_record = indexed.get((scene, int(seed), f"stop_{delay_name}"))
                pair_id = f"{scene}__seed{seed}__{delay_name}"
                bundle_reasons: list[str] = []
                if control_record is None:
                    bundle_reasons.append("missing_control")
                if stop_record is None:
                    bundle_reasons.append("missing_intervention")
                if bundle_reasons:
                    pair_rows.append(
                        {
                            "pair_id": pair_id,
                            "scene": scene,
                            "seed": seed,
                            "delay": delay_name,
                            "pair_valid": False,
                            "pair_status": "invalid",
                            "invalid_reasons": bundle_reasons,
                        }
                    )
                    continue

                try:
                    control = load_artifacts(args.root, control_record)
                    stop = load_artifacts(args.root, stop_record)
                except Exception as error:
                    bundle_reasons.append(f"artifact_error:{error}")
                    pair_rows.append(
                        {
                            "pair_id": pair_id,
                            "scene": scene,
                            "seed": seed,
                            "delay": delay_name,
                            "pair_valid": False,
                            "pair_status": "invalid",
                            "invalid_reasons": bundle_reasons,
                        }
                    )
                    continue

                admission_latent = int(stop_record["admission_latent"])
                admission_frame = int(stop_record["admission_frame"])
                control_latent = split_latent(control["latent"])
                stop_latent = split_latent(stop["latent"])
                if control_latent.shape != stop_latent.shape:
                    bundle_reasons.append("latent_shape_mismatch")
                    latent_prefix_error = None
                else:
                    latent_prefix_error = float(
                        (control_latent[:, :, :, :admission_latent]
                         - stop_latent[:, :, :, :admission_latent]).abs().max()
                    )
                    if latent_prefix_error != 0:
                        bundle_reasons.append("latent_prefix_mismatch")

                paired_results = []
                for view in (0, 1):
                    view_reasons = list(bundle_reasons)
                    effect_start = effect_starts.get((admission_latent, view))
                    if effect_start is None:
                        view_reasons.append("missing_rf_effect_start")
                        effect_start = admission_frame

                    control_raw = raw_view(control["decoded"], view)
                    stop_raw = raw_view(stop["decoded"], view)
                    raw_prefix_error = float(
                        np.max(
                            np.abs(
                                control_raw[:effect_start].astype(np.int16)
                                - stop_raw[:effect_start].astype(np.int16)
                            )
                        )
                    )
                    if raw_prefix_error != 0:
                        view_reasons.append("raw_prefix_mismatch")

                    control_frames = list(control["detector"][view])
                    stop_frames = list(stop["detector"][view])
                    paired = paired_flow_field_signals(
                        control_frames, stop_frames, detector_config
                    )
                    finite = (
                        float(np.min(paired["finite_fraction"]))
                        if paired["finite_fraction"].size
                        else 0.0
                    )
                    if finite < float(
                        protocol.data["validation"]["minimum_finite_flow_fraction"]
                    ):
                        view_reasons.append("insufficient_finite_flow")

                    detector_search_start, detector_search_end = (
                        support_aligned_search_range(
                            effect_start,
                            paired_config.search_horizon_after_admission,
                        )
                    )
                    result = detect_calibrated_paired_divergence(
                        paired[paired_config.signal],
                        thresholds[(scene, view)],
                        detector_search_start,
                        paired_config,
                        change_frame=int(protocol.data["confirmatory"]["change_frame"]),
                        search_end=detector_search_end,
                    )
                    signal_stats = signal_window_statistics(
                        paired[paired_config.signal],
                        detector_search_start,
                        detector_search_end,
                        thresholds[(scene, view)],
                    )
                    if view_reasons:
                        invalidate(result, view_reasons)
                    onset_frame = result["onset_frame"]
                    offset_from_effect_start = (
                        int(onset_frame - effect_start)
                        if onset_frame is not None
                        else None
                    )
                    offset_from_admission = (
                        int(onset_frame - admission_frame)
                        if onset_frame is not None
                        else None
                    )
                    result.update(
                        {
                            "schema_version": "rtwm-v2-confirmatory-view-1",
                            "pair_id": pair_id,
                            "scene": scene,
                            "seed": seed,
                            "delay": delay_name,
                            "view_index": view,
                            "pair_valid": not view_reasons,
                            "invalid_reasons": sorted(set(view_reasons)),
                            "admission_latent": admission_latent,
                            "admission_frame": admission_frame,
                            "effect_start_frame": effect_start,
                            "detector_search_start_signal_index": detector_search_start,
                            "offset_from_effect_start": offset_from_effect_start,
                            "offset_from_admission": offset_from_admission,
                            "latent_prefix_max_error": latent_prefix_error,
                            "raw_prefix_max_error": raw_prefix_error,
                            "minimum_finite_flow_fraction": finite,
                            "calibration_threshold": thresholds[(scene, view)],
                            **signal_stats,
                            "control_artifacts": record_artifacts(control_record),
                            "intervention_artifacts": record_artifacts(stop_record),
                        }
                    )
                    paired_results.append(result)
                    view_rows.append(result)
                    summary_rows.append(
                        {
                            "pair_id": pair_id,
                            "scene": scene,
                            "seed": seed,
                            "delay": delay_name,
                            "view": view,
                            "status": result["status"],
                            "pair_valid": not view_reasons,
                            "onset_frame": result["onset_frame"],
                            "lag_from_change": result["lag_from_change"],
                            "offset_from_effect_start": result["offset_from_effect_start"],
                            "latent_prefix_max_error": latent_prefix_error,
                            "raw_prefix_max_error": raw_prefix_error,
                            "finite_fraction": finite,
                        }
                    )

                pair_valid = all(result["pair_valid"] for result in paired_results)
                aggregate = aggregate_pair(paired_results, detector_config)
                if not pair_valid:
                    aggregate.update(
                        pair_status="invalid",
                        view_statuses=["invalid", "invalid"],
                        view_onsets=[None, None],
                        view_lags=[None, None],
                        detected_count=0,
                        both_detected_onset_mean=None,
                        onset_disagreement_frames=None,
                    )
                pair_peaks = [
                    float(result["signal_peak"])
                    for result in paired_results
                    if result.get("signal_peak") is not None
                ]
                pair_effect_starts = [
                    int(result["effect_start_frame"]) for result in paired_results
                ]
                pair_rows.append(
                    {
                        "schema_version": "rtwm-v2-confirmatory-pair-1",
                        "pair_id": pair_id,
                        "scene": scene,
                        "seed": seed,
                        "delay": delay_name,
                        "n_frames": int(protocol.data["model"]["n_frames"]),
                        "change_frame": int(
                            protocol.data["confirmatory"]["change_frame"]
                        ),
                        "admission_frame": admission_frame,
                        "admission_lag_frames": admission_frame
                        - int(protocol.data["confirmatory"]["change_frame"]),
                        "effect_start_frames": pair_effect_starts,
                        "pair_signal_peak_mean": (
                            float(np.mean(pair_peaks)) if pair_peaks else None
                        ),
                        "control_total_s": float(control_record["total_s"]),
                        "intervention_total_s": float(stop_record["total_s"]),
                        "pair_valid": pair_valid,
                        "invalid_reasons": sorted(
                            {
                                reason
                                for result in paired_results
                                for reason in result["invalid_reasons"]
                            }
                        ),
                        **aggregate,
                    }
                )

    valid_pairs = [row for row in pair_rows if row.get("pair_valid")]
    analysis_gate = {
        "schema_version": "rtwm-v2-confirmatory-analysis-gate-1",
        **protocol.identity,
        "expected_pairs": expected_pairs,
        "observed_pairs": len(pair_rows),
        "valid_pairs": len(valid_pairs),
        "invalid_pairs": len(pair_rows) - len(valid_pairs),
        "complete": len(pair_rows) == expected_pairs and len(valid_pairs) == expected_pairs,
        "source_patch": {
            "path": str(args.source_patch.resolve().relative_to(HERE)),
            "sha256": source_patch_sha256,
        },
    }
    statistics = seed_level_statistics(
        pair_rows,
        change_frame=int(protocol.data["confirmatory"]["change_frame"]),
        n_frames=int(protocol.data["model"]["n_frames"]),
    )
    statistics.update(protocol.identity)
    statistics["statistics_sha256"] = canonical_sha256(statistics)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "view_results.jsonl", view_rows)
    write_jsonl(args.output / "pair_results.jsonl", pair_rows)
    atomic_write_json(args.output / "statistics.json", statistics)
    analysis_gate["outputs"] = {
        "view_results.jsonl": file_sha256(args.output / "view_results.jsonl"),
        "pair_results.jsonl": file_sha256(args.output / "pair_results.jsonl"),
        "statistics.json": file_sha256(args.output / "statistics.json"),
    }
    analysis_gate["gate_sha256"] = canonical_sha256(analysis_gate)
    atomic_write_json(args.output / "analysis_gate.json", analysis_gate)
    if summary_rows:
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(summary_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
    print(json.dumps(analysis_gate, indent=2))


if __name__ == "__main__":
    main()
