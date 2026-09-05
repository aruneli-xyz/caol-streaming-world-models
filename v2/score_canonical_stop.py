"""Score fresh canonical STOP pairs against fresh canonical controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import atomic_write, atomic_write_json, verify_rollout_artifacts  # noqa: E402
from canonical_stop import load_stop_protocol, stop_specs, validate_canonical_evidence  # noqa: E402
from directional_study import detector_config  # noqa: E402
from flow_detector import (  # noqa: E402
    PairedDetectorConfig, aggregate_pair, detect_calibrated_paired_divergence,
    paired_flow_field_signals,
)
from protocol import canonical_sha256, file_sha256  # noqa: E402
from score_confirmatory import (  # noqa: E402
    bootstrap_mean_ci, seed_level_statistics, signal_window_statistics,
    support_aligned_search_range,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        lambda temporary: temporary.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        ),
    )


def scene_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for delay_index, delay in enumerate(("d0", "d1")):
        result[delay] = {}
        for scene_index, scene in enumerate(("buildTower_normal", "buildHouse_flat")):
            selected = [row for row in rows if row["delay"] == delay and row["scene"] == scene and row["pair_valid"]]
            seeds = sorted({int(row["seed"]) for row in selected})
            both, any_view, lags, peaks, fps = [], [], [], [], []
            for seed in seeds:
                row = next(item for item in selected if int(item["seed"]) == seed)
                both.append(float(row["pair_status"] == "both"))
                any_view.append(float(row["detected_count"] > 0))
                lags.append(
                    float(row["both_detected_onset_mean"] - row["change_frame"])
                    if row["both_detected_onset_mean"] is not None else float("nan")
                )
                peaks.append(float(row["pair_signal_peak_mean"]))
                fps.append(float(row["n_frames"] / row["intervention_total_s"]))
            base = 20260821 + 100 * delay_index + 10 * scene_index
            result[delay][scene] = {
                "valid_pairs": len(selected),
                "both_detected": int(sum(both)),
                "any_detected": int(sum(any_view)),
                "both_view_detection_fraction": bootstrap_mean_ci(both, seed=base),
                "any_view_detection_fraction": bootstrap_mean_ci(any_view, seed=base + 1),
                "onset_lag_frames": bootstrap_mean_ci(lags, seed=base + 2),
                "flow_divergence_peak": bootstrap_mean_ci(peaks, seed=base + 3),
                "effective_generation_fps": bootstrap_mean_ci(fps, seed=base + 4),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "canonical_stop.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "results" / "canonical_stop" / "manifest.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "canonical_stop")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "canonical_stop_scoring")
    args = parser.parse_args()
    protocol = load_stop_protocol(args.config)
    evidence = validate_canonical_evidence(protocol, verify_artifacts=True)
    if not evidence["allowed"]:
        raise RuntimeError("canonical control/null evidence is invalid")
    manifest = load_json(args.manifest)
    action_hash = protocol.data["action_protocol"]["canonical_action_protocol_sha256"]
    if (
        manifest.get("config_sha256") != protocol.file_sha256
        or manifest.get("protocol_sha256") != protocol.canonical_sha256
        or manifest.get("action_protocol_sha256") != action_hash
        or manifest.get("planned_interventions") != 24
    ):
        raise RuntimeError("canonical STOP manifest identity mismatch")
    null_root = HERE / protocol.data["canonical_evidence"]["null_calibration"]
    null_gate = load_json(null_root / "gate_report.json")
    if (
        null_gate.get("passed") is not True
        or null_gate.get("action_protocol_sha256") != action_hash
        or null_gate.get("manifest_sha256")
        != protocol.data["canonical_evidence"]["source_manifest_sha256"]
    ):
        raise RuntimeError("canonical null locks are incompatible")
    thresholds = {}
    for path in sorted((null_root / "locks").glob("*.json")):
        lock = load_json(path)
        thresholds[(lock["scene"], int(lock["view_index"]))] = float(lock["threshold"])
    if len(thresholds) != 4:
        raise RuntimeError("expected four canonical null locks")
    detector = detector_config(
        type("_Directional", (), {"data": load_json(HERE / protocol.data["canonical_evidence"]["directional_config"])})()
    )
    paired_config = PairedDetectorConfig.from_json(protocol.path)
    interventions = {
        row["rollout_id"]: row
        for row in manifest["rollouts"] if row.get("status") == "complete"
    }
    controls = evidence["controls"]
    control_root = HERE / protocol.data["canonical_evidence"]["source_root"]
    view_rows, pair_rows = [], []
    for spec in stop_specs(protocol):
        record = interventions.get(spec["rollout_id"])
        if record is None:
            raise RuntimeError(f"missing intervention {spec['rollout_id']}")
        control_record = controls[(spec["scene"], int(spec["seed"]))]
        intervention_paths = verify_rollout_artifacts(
            args.root, record,
            required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
        )
        control_paths = verify_rollout_artifacts(
            control_root, control_record,
            required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
        )
        intervention_latent = torch.load(intervention_paths["latent"], map_location="cpu")
        control_latent = torch.load(control_paths["latent"], map_location="cpu")
        admission = int(spec["admission_latent"])
        shape = (
            intervention_latent.shape[0], intervention_latent.shape[1], 2,
            intervention_latent.shape[2] // 2,
            intervention_latent.shape[3], intervention_latent.shape[4],
        )
        latent_prefix_error = float(
            (intervention_latent.reshape(shape)[:, :, :, :admission]
             - control_latent.reshape(shape)[:, :, :, :admission]).abs().max()
        )
        with np.load(intervention_paths["detector_frames"]) as payload:
            intervention_frames = np.asarray(payload["views"])
        with np.load(control_paths["detector_frames"]) as payload:
            control_frames = np.asarray(payload["views"])
        paired_results = []
        for view in (0, 1):
            effect_start = int(record["effect_start_frames"][view])
            start, end = support_aligned_search_range(
                effect_start, paired_config.search_horizon_after_admission
            )
            signals = paired_flow_field_signals(
                list(control_frames[view]), list(intervention_frames[view]), detector
            )
            threshold = thresholds[(spec["scene"], view)]
            result = detect_calibrated_paired_divergence(
                signals["flow_field_l2"], threshold, start, paired_config,
                change_frame=int(protocol.data["model"]["change_frame"]), search_end=end,
            )
            stats = signal_window_statistics(
                signals["flow_field_l2"], start, end, threshold
            )
            valid = (
                latent_prefix_error == 0
                and record.get("raw_prefix_max_errors") == [0.0, 0.0]
                and record.get("action_protocol_sha256") == action_hash
                and float(np.min(signals["finite_fraction"]))
                >= float(protocol.data["validation"]["minimum_finite_flow_fraction"])
            )
            if not valid:
                result.update(
                    status="invalid", onset_signal_index=None, onset_frame=None,
                    lag_from_change=None, offset_from_admission=None,
                )
            result.update({
                "schema_version": "rtwm-v2-canonical-stop-view-1",
                "pair_id": f"{spec['scene']}__seed{spec['seed']}__{spec['delay']}",
                "scene": spec["scene"], "seed": spec["seed"], "delay": spec["delay"],
                "view_index": view, "pair_valid": valid,
                "admission_latent": admission, "effect_start_frame": effect_start,
                "detector_search_start_signal_index": start,
                "latent_prefix_max_error": latent_prefix_error,
                "raw_prefix_max_error": record["raw_prefix_max_errors"][view],
                "minimum_finite_flow_fraction": float(np.min(signals["finite_fraction"])),
                **stats,
            })
            paired_results.append(result)
            view_rows.append(result)
        aggregate = aggregate_pair(paired_results, detector)
        pair_valid = all(row["pair_valid"] for row in paired_results)
        peaks = [float(row["signal_peak"]) for row in paired_results]
        pair_rows.append({
            "schema_version": "rtwm-v2-canonical-stop-pair-1",
            "pair_id": f"{spec['scene']}__seed{spec['seed']}__{spec['delay']}",
            "scene": spec["scene"], "seed": spec["seed"], "delay": spec["delay"],
            "n_frames": int(protocol.data["model"]["n_frames"]),
            "change_frame": int(protocol.data["model"]["change_frame"]),
            "admission_frame": admission * int(protocol.data["model"]["latent_stride_pixels"]),
            "admission_lag_frames": admission * int(protocol.data["model"]["latent_stride_pixels"])
            - int(protocol.data["model"]["change_frame"]),
            "effect_start_frames": record["effect_start_frames"],
            "pair_signal_peak_mean": float(np.mean(peaks)),
            "control_total_s": float(control_record["total_s"]),
            "intervention_total_s": float(record["total_s"]),
            "pair_valid": pair_valid,
            "invalid_reasons": [] if pair_valid else ["prefix_artifact_or_flow_invalid"],
            **aggregate,
        })
    statistics = seed_level_statistics(
        pair_rows, change_frame=int(protocol.data["model"]["change_frame"]),
        n_frames=int(protocol.data["model"]["n_frames"]),
    )
    statistics["schema_version"] = "rtwm-v2-canonical-stop-statistics-1"
    statistics["by_scene"] = scene_statistics(pair_rows)
    statistics["legacy_comparison"] = protocol.data["legacy_comparison"]
    statistics["canonical_minus_legacy"] = {
        delay: {
            metric: statistics["by_delay"][delay]["metrics"][metric]["mean"]
            - float(protocol.data["legacy_comparison"][f"{delay}_{legacy_name}"])
            for metric, legacy_name in (
                ("causal_onset_lag_frames", "onset_lag_frames"),
                ("flow_divergence_peak", "flow_divergence_peak"),
                ("effective_generation_fps", "effective_generation_fps"),
            )
        }
        for delay in ("d0", "d1")
    }
    statistics.update(protocol.identity)
    statistics["statistics_sha256"] = canonical_sha256(statistics)
    valid_pairs = [row for row in pair_rows if row["pair_valid"]]
    analysis = {
        "schema_version": "rtwm-v2-canonical-stop-analysis-gate-1",
        **protocol.identity,
        "expected_pairs": 24, "observed_pairs": len(pair_rows),
        "valid_pairs": len(valid_pairs), "invalid_pairs": 24 - len(valid_pairs),
        "complete": len(valid_pairs) == 24,
        "old_paper_claims_allowed": len(valid_pairs) == 24
        and all(
            statistics["by_delay"][delay]["metrics"]["both_view_detection_fraction"]["mean"] >= 0.75
            for delay in ("d0", "d1")
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "view_results.jsonl", view_rows)
    write_jsonl(args.output / "pair_results.jsonl", pair_rows)
    atomic_write_json(args.output / "statistics.json", statistics)
    analysis["outputs"] = {
        name: file_sha256(args.output / name)
        for name in ("view_results.jsonl", "pair_results.jsonl", "statistics.json")
    }
    analysis["gate_sha256"] = canonical_sha256(analysis)
    atomic_write_json(args.output / "analysis_gate.json", analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
