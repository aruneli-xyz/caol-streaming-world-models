"""Run v1 and transition-aware v2 detector ablations on existing videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

V2 = Path(__file__).resolve().parent
RTWM = V2.parent
sys.path.insert(0, str(RTWM))
sys.path.insert(0, str(V2))

from a2e import flow_signal as v1_flow_signal  # noqa: E402
from a2e import onset_frame as v1_onset_frame  # noqa: E402
from flow_detector import DetectorConfig, aggregate_pair, score_video  # noqa: E402


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_value, sort_keys=True) + "\n")


def transition_name(name: str) -> str:
    return "left" if name == "turn" else name


def v1_view_result(video: Path, view: int, transition: str, change: int) -> dict[str, Any]:
    signal = v1_flow_signal(video, view)
    onset = v1_onset_frame(signal, change, rising=transition == "start")
    return {
        "detector": "v1_future_midpoint_magnitude",
        "view_index": view,
        "signal_name": "magnitude",
        "onset_frame": onset,
        "lag_from_change": onset - change if onset is not None else None,
        "status": "detected" if onset is not None else "undetected",
    }


def enrich(
    result: dict[str, Any],
    metadata: dict[str, Any],
    detector: str,
) -> dict[str, Any]:
    row = dict(metadata)
    row.update(result)
    row["detector"] = detector
    return row


def score_intrinsic(
    config: DetectorConfig,
    signal_store: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = RTWM / "results" / "a2e"
    manifest = json.loads((base / "manifest.json").read_text())
    view_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for item in manifest:
        video = base / "out" / item["sample"] / item["sample"] / "generated.mp4"
        if not video.exists():
            continue
        transition = transition_name(item["transition"])
        metadata = {
            "schema_version": "rtwm-v2-detector-1",
            "dataset": "intrinsic_v1_videos",
            "sample": item["sample"],
            "pair_id": item["sample"],
            "scene": item["frame_key"],
            "seed": None,
            "delay_blocks": 0,
            "arm": "actions_known_upfront",
            "transition": transition,
            "change_frame": item["change_frame"],
            "admission_frame": item["change_frame"],
        }

        v2_results, signals, video_metadata = score_video(
            video,
            transition,
            item["change_frame"],
            config,
            admission_frame=item["change_frame"],
        )
        for view, (result, view_signals) in enumerate(zip(v2_results, signals)):
            row = enrich(result, metadata, "v2_preonly_transition_aware")
            row["video_metadata"] = video_metadata
            view_rows.append(row)
            for signal_name, signal in view_signals.items():
                signal_store[f"{item['sample']}__v{view}__{signal_name}"] = signal

        v2_pair = aggregate_pair(v2_results, config)
        pair_rows.append(
            {
                **metadata,
                **v2_pair,
                "detector": "v2_preonly_transition_aware",
            }
        )

        v1_results = [
            v1_view_result(video, view, item["transition"], item["change_frame"])
            for view in (0, 1)
        ]
        for result in v1_results:
            view_rows.append(enrich(result, metadata, "v1_future_midpoint_magnitude"))
        pair_rows.append(
            {
                **metadata,
                **aggregate_pair(v1_results, config),
                "detector": "v1_future_midpoint_magnitude",
            }
        )
    return view_rows, pair_rows


def score_serving(
    config: DetectorConfig,
    signal_store: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = RTWM / "results" / "serving_a2e"
    manifest = json.loads((base / "manifest.json").read_text())
    view_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for item in manifest:
        video = base / f"seed{item['seed']}" / item["arm"] / "generated.mp4"
        if not video.exists():
            continue
        sample = f"serving__seed{item['seed']}__{item['arm']}"
        metadata = {
            "schema_version": "rtwm-v2-detector-1",
            "dataset": "serving_v1_videos",
            "sample": sample,
            "pair_id": sample,
            "scene": "buildTower_normal",
            "seed": item["seed"],
            "delay_blocks": item["delay_blocks"],
            "arm": item["arm"],
            "transition": "stop",
            "change_frame": item["change_frame"],
            "admission_frame": item["effective_switch_px"],
        }
        results, signals, video_metadata = score_video(
            video,
            "stop",
            item["change_frame"],
            config,
            admission_frame=item["effective_switch_px"],
        )
        for view, (result, view_signals) in enumerate(zip(results, signals)):
            row = enrich(result, metadata, "v2_preonly_transition_aware")
            row["video_metadata"] = video_metadata
            view_rows.append(row)
            for signal_name, signal in view_signals.items():
                signal_store[f"{sample}__v{view}__{signal_name}"] = signal
        pair_rows.append(
            {
                **metadata,
                **aggregate_pair(results, config),
                "detector": "v2_preonly_transition_aware",
            }
        )
    return view_rows, pair_rows


def summarize(view_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in view_rows:
        key = (str(row["dataset"]), str(row["detector"]), str(row["transition"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, detector, transition), rows in sorted(grouped.items()):
        detected = [row for row in rows if row["status"] == "detected"]
        lags = [float(row["lag_from_change"]) for row in detected]
        summaries.append(
            {
                "dataset": dataset,
                "detector": detector,
                "transition": transition,
                "n_views": len(rows),
                "n_detected": len(detected),
                "detection_rate": len(detected) / len(rows),
                "mean_lag": float(np.mean(lags)) if lags else None,
                "median_lag": float(np.median(lags)) if lags else None,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=V2 / "config" / "pilot.json")
    parser.add_argument("--output", type=Path, default=V2 / "results" / "detector_ablation")
    parser.add_argument("--dataset", choices=["intrinsic", "serving", "all"], default="all")
    args = parser.parse_args()

    config = DetectorConfig.from_json(args.config)
    view_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    signals: dict[str, np.ndarray] = {}
    if args.dataset in ("intrinsic", "all"):
        views, pairs = score_intrinsic(config, signals)
        view_rows.extend(views)
        pair_rows.extend(pairs)
    if args.dataset in ("serving", "all"):
        views, pairs = score_serving(config, signals)
        view_rows.extend(views)
        pair_rows.extend(pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "view_results.jsonl", view_rows)
    write_jsonl(args.output / "pair_results.jsonl", pair_rows)
    np.savez_compressed(args.output / "signals.npz", **signals)

    summary = summarize(view_rows)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))
    print(f"saved detector ablation -> {args.output}")


if __name__ == "__main__":
    main()
