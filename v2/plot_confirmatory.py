"""Plot artifact-derived confirmatory reliability, effect, and throughput."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE / "results" / "confirmatory_scoring",
    )
    args = parser.parse_args()
    pairs = read_jsonl(args.results / "pair_results.jsonl")
    views = read_jsonl(args.results / "view_results.jsonl")

    valid_pairs = [row for row in pairs if row.get("pair_valid")]
    scenes = sorted({row["scene"] for row in valid_pairs})
    delays = sorted({row["delay"] for row in valid_pairs})
    if not scenes or not delays:
        raise RuntimeError("no valid confirmatory pairs to plot")
    admission_lags = {
        delay: sorted(
            {
                int(row["admission_lag_frames"])
                for row in valid_pairs
                if row["delay"] == delay
            }
        )
        for delay in delays
    }
    if any(len(values) != 1 for values in admission_lags.values()):
        raise RuntimeError("each delay must map to one admission lag")
    delay_labels = {
        delay: f"{admission_lags[delay][0]}-frame admission lag"
        for delay in delays
    }
    x = np.arange(len(scenes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    ax = axes[0]
    for index, delay in enumerate(delays):
        rates = []
        for scene in scenes:
            selected = [
                row for row in pairs
                if row["scene"] == scene
                and row["delay"] == delay
                and row["pair_valid"]
            ]
            rates.append(
                sum(row["pair_status"] == "both" for row in selected) / len(selected)
                if selected
                else 0
            )
        ax.bar(
            x + (index - 0.5) * width,
            rates,
            width,
            label=delay_labels[delay],
        )
    scene_labels = [
        scene.replace("build", "").replace("_normal", "").replace("_flat", "")
        for scene in scenes
    ]
    ax.set_xticks(x, scene_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("both-view detection fraction")
    ax.set_title("(a) Confirmatory reliability")
    ax.legend(fontsize=8)

    ax = axes[1]
    colors = {"d0": "#2d6a4f", "d1": "#ca6702"}
    seeds = sorted({int(row["seed"]) for row in valid_pairs})
    for delay in delays:
        values = []
        for seed in seeds:
            selected = [
                row
                for row in valid_pairs
                if row["delay"] == delay and int(row["seed"]) == seed
            ]
            if selected:
                values.append(
                    float(np.mean([row["pair_signal_peak_mean"] for row in selected]))
                )
        jitter = np.linspace(-0.08, 0.08, len(values)) if values else []
        ax.scatter(
            np.full(len(values), delays.index(delay)) + jitter,
            values,
            color=colors.get(delay),
            label=delay_labels[delay],
            alpha=0.8,
        )
    ax.set_xticks(
        range(len(delays)),
        [f"{admission_lags[delay][0]} frames" for delay in delays],
    )
    ax.set_ylabel("peak paired flow-field divergence")
    ax.set_title("(b) Continuous causal effect")

    ax = axes[2]
    seed_points: dict[tuple[int, str], tuple[float, float]] = {}
    for delay in delays:
        for seed in seeds:
            selected = [
                row
                for row in valid_pairs
                if row["delay"] == delay and int(row["seed"]) == seed
            ]
            onset_lags = [
                float(row["both_detected_onset_mean"]) - float(row["change_frame"])
                for row in selected
                if row.get("both_detected_onset_mean") is not None
            ]
            if not selected or not onset_lags:
                continue
            fps = float(
                np.mean(
                    [
                        float(row["n_frames"]) / float(row["intervention_total_s"])
                        for row in selected
                    ]
                )
            )
            lag = float(np.mean(onset_lags))
            seed_points[(seed, delay)] = (fps, lag)
            ax.scatter(
                fps,
                lag,
                color=colors.get(delay),
                marker="o" if delay == delays[0] else "s",
                alpha=0.85,
            )
    for seed in seeds:
        points = [
            seed_points[(seed, delay)]
            for delay in delays
            if (seed, delay) in seed_points
        ]
        if len(points) == len(delays):
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color="#6c757d",
                linewidth=0.7,
                alpha=0.55,
            )
    ax.set_xlabel("measured effective generation FPS")
    ax.set_ylabel("paired divergence onset lag (frames)")
    ax.set_title("(c) Similar throughput, different response")

    fig.tight_layout()
    output = args.results / "confirmatory_summary.png"
    fig.savefig(output, dpi=180)

    rows = []
    for scene in scenes:
        for delay in delays:
            selected = [
                row for row in pairs
                if row["scene"] == scene and row["delay"] == delay
            ]
            valid = [row for row in selected if row["pair_valid"]]
            rows.append(
                {
                    "scene": scene,
                    "delay": delay,
                    "planned_pairs": len(selected),
                    "valid_pairs": len(valid),
                    "invalid_pairs": len(selected) - len(valid),
                    "both_views_detected": sum(
                        row["pair_status"] == "both" for row in valid
                    ),
                    "any_view_detected": sum(
                        row["detected_count"] > 0 for row in valid
                    ),
                    "admission_lag_frames": admission_lags[delay][0],
                    "mean_effect_peak": (
                        float(
                            np.mean(
                                [row["pair_signal_peak_mean"] for row in valid]
                            )
                        )
                        if valid
                        else None
                    ),
                    "mean_effective_generation_fps": (
                        float(
                            np.mean(
                                [
                                    float(row["n_frames"])
                                    / float(row["intervention_total_s"])
                                    for row in valid
                                ]
                            )
                        )
                        if valid
                        else None
                    ),
                }
            )
    with (args.results / "reliability.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
