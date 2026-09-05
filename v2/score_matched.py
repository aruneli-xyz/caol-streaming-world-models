"""Score matched STOP-versus-forward counterfactual rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))

from flow_detector import (  # noqa: E402
    DetectorConfig,
    aggregate_pair,
    detect_onset,
    detect_paired_divergence,
    flow_signals,
    read_side_by_side_video,
)
from artifacts import (  # noqa: E402
    ArtifactError,
    atomic_numpy_savez,
    atomic_write_json,
    commit_temporary,
    record_artifacts,
    temporary_path,
    verify_artifact,
)
from gates import build_stage1_gate  # noqa: E402
from preflight import require_synthetic_suite  # noqa: E402
from protocol import LoadedProtocol, load_protocol  # noqa: E402


DEFAULT_MATCHED = V2 / "results" / "matched"
DEFAULT_OUTPUT = V2 / "results" / "matched_scoring"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, json.loads(json.dumps(payload, default=json_default)))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_path(path) as temporary:
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")
        commit_temporary(temporary, path)


def record_integrity_reasons(
    matched_root: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    protocol: LoadedProtocol,
) -> list[str]:
    reasons: list[str] = []
    if manifest.get("config_sha256") != protocol.file_sha256:
        reasons.append("manifest_config_hash_mismatch")
    if manifest.get("protocol_sha256") not in (None, protocol.canonical_sha256):
        reasons.append("manifest_protocol_hash_mismatch")
    source = manifest.get("gamma_source", {})
    if source.get("commit") != protocol.data["model"]["source_commit"]:
        reasons.append("source_commit_mismatch")
    if source.get("dirty") and not source.get("diff_sha256"):
        reasons.append("source_diff_hash_missing")
    scene_hashes = manifest.get("scene_hashes")
    if not isinstance(scene_hashes, dict):
        reasons.append("manifest_scene_hashes_missing")
    elif scene_hashes.get(record["scene"]) != record.get("scene_sha256"):
        reasons.append("scene_hash_mismatch")

    artifacts = record_artifacts(record)
    verified_paths: dict[str, Path] = {}
    for name in ("latent", "decoded_u8", "detector_frames"):
        metadata = artifacts.get(name)
        if metadata is None:
            reasons.append(f"missing_{name}_artifact")
            continue
        try:
            verified_paths[name] = verify_artifact(
                matched_root,
                metadata,
                label=name,
            )
        except ArtifactError:
            reasons.append(f"invalid_{name}_artifact")
    detector_path = verified_paths.get("detector_frames")
    if detector_path is not None:
        try:
            with np.load(detector_path, allow_pickle=False) as payload:
                detector_views = payload["views"]
                shape = detector_views.shape
                dtype = detector_views.dtype
            expected_shape = (
                2,
                int(protocol.data["validation"]["expected_frames"]),
                int(protocol.data["detector"]["resize_height"]),
                int(protocol.data["detector"]["resize_width"]),
            )
            if shape != expected_shape or dtype != np.uint8:
                reasons.append("invalid_detector_frames_shape")
        except (OSError, KeyError, ValueError):
            reasons.append("invalid_detector_frames_payload")
    decoded_path = verified_paths.get("decoded_u8")
    if decoded_path is not None:
        try:
            decoded = np.load(decoded_path, allow_pickle=False, mmap_mode="r")
            if (
                decoded.ndim != 4
                or decoded.shape[0]
                != int(protocol.data["validation"]["expected_frames"])
                or decoded.shape[2] % 2
                or decoded.shape[3] != 3
                or decoded.dtype != np.uint8
            ):
                reasons.append("invalid_decoded_u8_shape")
        except (OSError, ValueError):
            reasons.append("invalid_decoded_u8_payload")
    if "video" in artifacts:
        try:
            verify_artifact(matched_root, artifacts["video"], label="video")
        except ArtifactError:
            reasons.append("invalid_video_artifact")
    return sorted(set(reasons))


def invalidate_result(result: dict[str, Any], reasons: list[str]) -> None:
    result["diagnostic_status"] = result.get("status")
    result["status"] = "invalid"
    result["invalid_reasons"] = reasons
    for key in (
        "onset_signal_index",
        "onset_frame",
        "lag_from_change",
        "offset_from_admission",
    ):
        result[key] = None


def prefix_max_error(
    control_frames: list[np.ndarray],
    stop_frames: list[np.ndarray],
    admission_frame: int,
) -> float:
    frame_count = min(admission_frame, len(control_frames), len(stop_frames))
    if frame_count <= 0:
        return float("nan")
    control = np.asarray(control_frames[:frame_count], dtype=np.int16)
    stop = np.asarray(stop_frames[:frame_count], dtype=np.int16)
    return float(np.max(np.abs(control - stop)))


def earliest_frame_difference(
    control_frames: list[np.ndarray],
    stop_frames: list[np.ndarray],
) -> int | None:
    frame_count = min(len(control_frames), len(stop_frames))
    if frame_count <= 0:
        return None
    control = np.asarray(control_frames[:frame_count], dtype=np.int16)
    stop = np.asarray(stop_frames[:frame_count], dtype=np.int16)
    per_frame = np.max(np.abs(control - stop), axis=(1, 2))
    indices = np.flatnonzero(per_frame > 0)
    return int(indices[0]) if indices.size else None


def full_max_error(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    frame_count = min(len(first), len(second))
    if frame_count <= 0:
        return float("nan")
    a = np.asarray(first[:frame_count], dtype=np.int16)
    b = np.asarray(second[:frame_count], dtype=np.int16)
    return float(np.max(np.abs(a - b)))


def latent_prefix_metrics(
    matched_root: Path,
    control_record: dict[str, Any],
    stop_record: dict[str, Any],
    admission_frame: int,
    latent_stride_pixels: int,
) -> dict[str, Any] | None:
    control_relative = control_record.get("latent_path")
    stop_relative = stop_record.get("latent_path")
    if not control_relative or not stop_relative:
        return None
    control_path = matched_root / control_relative
    stop_path = matched_root / stop_relative
    if not control_path.exists() or not stop_path.exists():
        return None
    try:
        control = torch.load(control_path, map_location="cpu")
        stop = torch.load(stop_path, map_location="cpu")
    except Exception as error:
        return {"status": "invalid_load", "error": repr(error)}
    if control.shape != stop.shape or control.ndim != 5 or control.shape[2] % 2:
        return {
            "status": "invalid_shape",
            "control_shape": list(control.shape),
            "stop_shape": list(stop.shape),
        }
    batch, channels, combined_time, height, width = control.shape
    temporal = combined_time // 2
    control = control.reshape(batch, channels, 2, temporal, height, width)
    stop = stop.reshape(batch, channels, 2, temporal, height, width)
    difference = (control - stop).abs()
    admission_latent = admission_frame // latent_stride_pixels
    prefix = difference[:, :, :, :admission_latent]
    per_view_earliest: list[int | None] = []
    for view in (0, 1):
        per_time = (
            difference[:, :, view]
            .permute(2, 0, 1, 3, 4)
            .flatten(1)
            .amax(dim=1)
        )
        indices = torch.nonzero(per_time > 0).flatten()
        per_view_earliest.append(int(indices[0]) if len(indices) else None)
    return {
        "status": "ok",
        "shape": list(control.shape),
        "admission_latent": admission_latent,
        "prefix_max_error": float(prefix.max()) if prefix.numel() else None,
        "full_max_error": float(difference.max()),
        "earliest_difference_latent_by_view": per_view_earliest,
    }


def plot_trace(
    path: Path,
    control: np.ndarray,
    stop: np.ndarray,
    difference: np.ndarray,
    paired: dict[str, Any],
    change_frame: int,
    admission_frame: int,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    axes[0].plot(control, label="forward control", lw=1.1)
    axes[0].plot(stop, label="stop intervention", lw=1.1)
    axes[0].set_ylabel("flow magnitude")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)

    axes[1].plot(difference, color="#2d6a4f", label="control - stop")
    if paired["target"] is not None:
        axes[1].axhline(paired["target"], color="#ca6702", ls="--", label="paired target")
    for axis in axes:
        axis.axvline(change_frame, color="#ae2012", ls=":", lw=1)
        axis.axvline(admission_frame, color="#2d6a4f", ls=":", lw=1)
    if paired["onset_signal_index"] is not None:
        axes[1].axvline(paired["onset_signal_index"], color="black", ls="--", lw=1, label="paired onset")
    axes[1].set_ylabel("paired contrast")
    axes[1].set_xlabel("flow signal index t (frames t -> t+1)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=V2 / "config" / "pilot.json")
    parser.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", action="append", default=None)
    args = parser.parse_args()

    loaded_protocol = load_protocol(args.config)
    protocol = loaded_protocol.data
    synthetic_report = require_synthetic_suite(loaded_protocol)
    config = DetectorConfig.from_json(args.config)
    paired_settings = protocol["paired_detector"]
    paired_config = replace(
        config,
        baseline_before=int(paired_settings["baseline_before"]),
        baseline_gap=int(paired_settings["baseline_gap"]),
        search_horizon=int(paired_settings["search_horizon_after_admission"]),
        persistence=int(paired_settings["persistence"]),
        mad_multiplier=float(paired_settings["mad_multiplier"]),
        relative_delta=float(paired_settings["relative_delta"]),
        absolute_delta=float(paired_settings["absolute_delta"]),
    )
    manifest_path = args.matched / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    complete = [record for record in manifest["rollouts"] if record["status"] == "complete"]
    by_key = {
        (record["scene"], int(record["seed"]), record["arm"]): record
        for record in complete
    }
    requested_scenes = set(args.scene or [])

    frame_cache: dict[str, tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]] = {}
    signal_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def load_record(record: dict[str, Any]):
        cache_key = str(record["rollout_id"])
        if cache_key not in frame_cache:
            detector_artifact = record_artifacts(record).get("detector_frames")
            if detector_artifact is not None:
                try:
                    path = verify_artifact(
                        args.matched,
                        detector_artifact,
                        label="detector_frames",
                    )
                    with np.load(path, allow_pickle=False) as payload:
                        views = payload["views"]
                    if (
                        views.ndim != 4
                        or views.shape[0] != 2
                        or views.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"invalid detector frame shape/dtype: {views.shape} {views.dtype}"
                        )
                    frame_cache[cache_key] = (
                        list(views[0]),
                        list(views[1]),
                        {
                            "path": str(path),
                            "frame_count": int(views.shape[1]),
                            "source": "precodec_detector_frames",
                        },
                    )
                except (ArtifactError, KeyError, ValueError):
                    pass
            if cache_key not in frame_cache:
                relative = record["video_path"]
                frame_cache[cache_key] = read_side_by_side_video(
                    args.matched / relative,
                    config,
                )
                frame_cache[cache_key][2]["source"] = "legacy_mp4_fallback"
        return frame_cache[cache_key]

    def signals_for(record: dict[str, Any], view: int) -> dict[str, np.ndarray]:
        key = (record["video_path"], view)
        if key not in signal_cache:
            frames = load_record(record)[view]
            signal_cache[key] = flow_signals(frames, config)
        return signal_cache[key]

    view_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    unique_control_checks: dict[tuple[str, int, int], dict[str, Any]] = {}

    stop_records = [
        record
        for record in complete
        if record["arm"].startswith("stop_d")
        and (not requested_scenes or record["scene"] in requested_scenes)
    ]
    for stop_record in sorted(
        stop_records,
        key=lambda record: (record["scene"], int(record["seed"]), int(record["delay_blocks"])),
    ):
        scene = stop_record["scene"]
        seed = int(stop_record["seed"])
        delay = int(stop_record["delay_blocks"])
        control_record = by_key.get((scene, seed, "control"))
        if control_record is None:
            raise RuntimeError(f"missing control for {scene} seed {seed}")

        change_frame = int(stop_record["change_frame"])
        admission_frame = int(stop_record["admission_frame"])
        pair_id = f"{scene}__seed{seed}__d{delay}"
        pair_invalid_reasons = [
            f"control:{reason}"
            for reason in record_integrity_reasons(
                args.matched,
                manifest,
                control_record,
                loaded_protocol,
            )
        ]
        pair_invalid_reasons.extend(
            f"intervention:{reason}"
            for reason in record_integrity_reasons(
                args.matched,
                manifest,
                stop_record,
                loaded_protocol,
            )
        )
        control_frames = load_record(control_record)
        stop_frames = load_record(stop_record)
        latent_metrics = latent_prefix_metrics(
            args.matched,
            control_record,
            stop_record,
            admission_frame,
            int(protocol["model"]["latent_stride_pixels"]),
        )
        frame_count_ok = all(
            len(view_frames) == int(protocol["validation"]["expected_frames"])
            for view_frames in (*control_frames[:2], *stop_frames[:2])
        )
        if not frame_count_ok:
            pair_invalid_reasons.append("unexpected_frame_count")
        if latent_metrics is None:
            pair_invalid_reasons.append("latent_comparison_unavailable")
        elif latent_metrics.get("status") != "ok":
            pair_invalid_reasons.append(
                f"latent_comparison_{latent_metrics.get('status', 'invalid')}"
            )
        elif latent_metrics.get("prefix_max_error") != 0:
            pair_invalid_reasons.append("latent_prefix_mismatch")

        paired_view_results: list[dict[str, Any]] = []
        prefix_errors: list[float] = []
        earliest_pixel_differences: list[int | None] = []
        finite_fractions: list[dict[str, float]] = []
        for view in (0, 1):
            control_signals = signals_for(control_record, view)
            stop_signals = signals_for(stop_record, view)
            paired, difference = detect_paired_divergence(
                control_signals["magnitude"],
                stop_signals["magnitude"],
                change_frame,
                admission_frame,
                paired_config,
            )
            post_difference = difference[
                admission_frame:min(
                    len(difference),
                    admission_frame + paired_config.search_horizon,
                )
            ]
            paired["max_post_admission_contrast"] = (
                float(np.max(post_difference)) if post_difference.size else None
            )
            paired["max_to_target_ratio"] = (
                float(paired["max_post_admission_contrast"] / paired["target"])
                if paired["target"]
                and paired["max_post_admission_contrast"] is not None
                else None
            )
            stop_single = detect_onset(
                stop_signals,
                "stop",
                change_frame,
                config,
                search_start=change_frame,
                search_end=admission_frame + paired_config.search_horizon,
            )
            control_single = detect_onset(
                control_signals,
                "stop",
                change_frame,
                config,
                search_start=change_frame,
                search_end=admission_frame + paired_config.search_horizon,
            )
            prefix_error = prefix_max_error(
                control_frames[view],
                stop_frames[view],
                admission_frame,
            )
            prefix_errors.append(prefix_error)
            minimum_finite = float(
                protocol["validation"]["minimum_finite_flow_fraction"]
            )
            view_finite = {
                "control": (
                    float(np.min(control_signals["finite_fraction"]))
                    if control_signals["finite_fraction"].size
                    else 0.0
                ),
                "intervention": (
                    float(np.min(stop_signals["finite_fraction"]))
                    if stop_signals["finite_fraction"].size
                    else 0.0
                ),
                "paired": float(np.isfinite(difference).mean()) if difference.size else 0.0,
            }
            finite_fractions.append(view_finite)
            earliest_pixel_difference = earliest_frame_difference(
                control_frames[view],
                stop_frames[view],
            )
            earliest_pixel_differences.append(earliest_pixel_difference)
            qc_flags = list(paired["qc_flags"])
            if prefix_error != 0:
                qc_flags.append("decoded_prefix_mismatch")
                if protocol["validation"]["require_prefix_comparison"]:
                    pair_invalid_reasons.append(f"view{view}:decoded_prefix_mismatch")
            if not frame_count_ok:
                qc_flags.append("unexpected_frame_count")
            if min(view_finite.values()) < minimum_finite:
                qc_flags.append("insufficient_finite_flow")
                pair_invalid_reasons.append(f"view{view}:insufficient_finite_flow")
            view_invalid_reasons = sorted(set(pair_invalid_reasons))
            if view_invalid_reasons:
                qc_flags.append("excluded_invalid_pair")
                invalidate_result(paired, view_invalid_reasons)
            paired["qc_flags"] = qc_flags
            paired["view_index"] = view
            paired_view_results.append(paired)

            row = {
                "schema_version": "rtwm-v2-matched-score-1",
                "pair_id": pair_id,
                "scene": scene,
                "seed": seed,
                "delay_blocks": delay,
                "change_frame": change_frame,
                "admission_frame": admission_frame,
                "view_index": view,
                "paired": paired,
                "stop_single": stop_single,
                "control_single": control_single,
                "prefix_pixel_max_error": prefix_error,
                "earliest_pixel_difference_frame": earliest_pixel_difference,
                "latent_prefix_metrics": latent_metrics,
                "finite_flow_fractions": view_finite,
                "pair_valid": not view_invalid_reasons,
                "excluded_from_analysis": bool(view_invalid_reasons),
                "invalid_reasons": view_invalid_reasons,
                "qc_flags": qc_flags,
            }
            view_rows.append(row)
            unique_control_checks[(scene, seed, view)] = control_single
            plot_trace(
                args.output / "traces" / f"{pair_id}__v{view}.png",
                control_signals["magnitude"],
                stop_signals["magnitude"],
                difference,
                paired,
                change_frame,
                admission_frame,
                f"{scene}, seed {seed}, delay {delay}, view {view}",
            )
            flat_rows.append(
                {
                    "pair_id": pair_id,
                    "scene": scene,
                    "seed": seed,
                    "delay_blocks": delay,
                    "view": view,
                    "paired_status": paired["status"],
                    "paired_onset": paired["onset_frame"],
                    "max_post_admission_contrast": paired["max_post_admission_contrast"],
                    "max_to_target_ratio": paired["max_to_target_ratio"],
                    "lag_from_change": paired["lag_from_change"],
                    "offset_from_admission": paired["offset_from_admission"],
                    "stop_status": stop_single["status"],
                    "control_status": control_single["status"],
                    "prefix_pixel_max_error": prefix_error,
                    "earliest_pixel_difference_frame": earliest_pixel_difference,
                    "latent_prefix_max_error": (
                        latent_metrics.get("prefix_max_error")
                        if latent_metrics and latent_metrics.get("status") == "ok"
                        else None
                    ),
                    "minimum_finite_flow_fraction": min(view_finite.values()),
                    "pair_valid": not view_invalid_reasons,
                    "excluded_from_analysis": bool(view_invalid_reasons),
                    "invalid_reasons": "|".join(view_invalid_reasons),
                    "qc_flags": "|".join(qc_flags),
                }
            )

        pair_invalid_reasons = sorted(set(pair_invalid_reasons))
        pair_valid = not pair_invalid_reasons
        if not pair_valid:
            for result in paired_view_results:
                if result["status"] != "invalid":
                    invalidate_result(result, pair_invalid_reasons)
                result["qc_flags"] = sorted(
                    set(result["qc_flags"]).union({"excluded_invalid_pair"})
                )
            for row in view_rows[-2:]:
                row["pair_valid"] = False
                row["excluded_from_analysis"] = True
                row["invalid_reasons"] = pair_invalid_reasons
                row["paired"] = paired_view_results[int(row["view_index"])]
                row["qc_flags"] = sorted(
                    set(row["qc_flags"]).union({"excluded_invalid_pair"})
                )
            for row in flat_rows[-2:]:
                row["paired_status"] = "invalid"
                row["paired_onset"] = None
                row["lag_from_change"] = None
                row["offset_from_admission"] = None
                row["pair_valid"] = False
                row["excluded_from_analysis"] = True
                row["invalid_reasons"] = "|".join(pair_invalid_reasons)
                row["qc_flags"] = "|".join(
                    sorted(
                        set(filter(None, row["qc_flags"].split("|")))
                        .union({"excluded_invalid_pair"})
                    )
                )

        pair_summary = aggregate_pair(paired_view_results, config)
        if not pair_valid:
            pair_summary.update(
                pair_status="invalid",
                detected_count=0,
                view_statuses=["invalid", "invalid"],
                view_onsets=[None, None],
                view_lags=[None, None],
                both_detected_onset_mean=None,
                onset_disagreement_frames=None,
            )
        pair_qc_flags = sorted(
            set(pair_summary["qc_flags"]).union(
                flag for result in paired_view_results for flag in result["qc_flags"]
            )
        )
        pair_summary["qc_flags"] = pair_qc_flags
        pair_rows.append(
            {
                "pair_id": pair_id,
                "scene": scene,
                "seed": seed,
                "delay_blocks": delay,
                "change_frame": change_frame,
                "admission_frame": admission_frame,
                "prefix_pixel_max_errors": prefix_errors,
                "prefix_pixel_max_error": max(prefix_errors),
                "earliest_pixel_difference_frames": earliest_pixel_differences,
                "latent_prefix_metrics": latent_metrics,
                "finite_flow_fractions": finite_fractions,
                "pair_valid": pair_valid,
                "excluded_from_analysis": not pair_valid,
                "invalid_reasons": pair_invalid_reasons,
                **pair_summary,
            }
        )

    canary_results: list[dict[str, Any]] = []
    canaries = [record for record in complete if record["arm"] == "control_canary"]
    for canary in canaries:
        control = by_key.get((canary["scene"], int(canary["seed"]), "control"))
        if control is None:
            continue
        first = load_record(control)
        second = load_record(canary)
        errors = [full_max_error(first[view], second[view]) for view in (0, 1)]
        canary_invalid_reasons = [
            f"control:{reason}"
            for reason in record_integrity_reasons(
                args.matched,
                manifest,
                control,
                loaded_protocol,
            )
        ]
        canary_invalid_reasons.extend(
            f"canary:{reason}"
            for reason in record_integrity_reasons(
                args.matched,
                manifest,
                canary,
                loaded_protocol,
            )
        )
        canary_results.append(
            {
                "scene": canary["scene"],
                "seed": canary["seed"],
                "view_max_errors": errors,
                "max_error": max(errors),
                "hash_match": control["video_sha256"] == canary["video_sha256"],
                "valid": not canary_invalid_reasons,
                "invalid_reasons": sorted(set(canary_invalid_reasons)),
            }
        )

    false_positive_controls = sum(
        check["status"] == "detected" for check in unique_control_checks.values()
    )
    gate_report = build_stage1_gate(
        loaded_protocol,
        manifest_path,
        pair_rows,
        view_rows,
        canary_results,
        synthetic_report,
        false_positive_controls,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "synthetic_report.json", synthetic_report)
    write_jsonl(args.output / "view_results.jsonl", view_rows)
    write_jsonl(args.output / "pair_results.jsonl", pair_rows)
    write_json(args.output / "gate_report.json", gate_report)
    with temporary_path(args.output / "summary.csv") as temporary:
        with temporary.open("w", newline="") as handle:
            if flat_rows:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(flat_rows[0].keys()),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(flat_rows)
        commit_temporary(temporary, args.output / "summary.csv")

    signal_payload = {
        f"{record['video_path']}__v{view}__{name}": signal
        for record in complete
        if str(record["rollout_id"]) in frame_cache
        for view in (0, 1)
        for name, signal in signals_for(record, view).items()
    }
    atomic_numpy_savez(args.output / "signals.npz", np, **signal_payload)
    print(json.dumps(gate_report, indent=2))
    print(f"saved matched scores -> {args.output}")


if __name__ == "__main__":
    main()
