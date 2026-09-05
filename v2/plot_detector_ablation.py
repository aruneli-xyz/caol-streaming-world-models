"""Plot saved detector-ablation summaries and per-view traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


V2 = Path(__file__).resolve().parent
DEFAULT_RESULTS = V2 / "results" / "detector_ablation"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_summary(results: Path) -> None:
    with (results / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    intrinsic = [row for row in rows if row["dataset"] == "intrinsic_v1_videos"]
    transitions = ["start", "stop", "reverse", "left"]
    detectors = ["v1_future_midpoint_magnitude", "v2_preonly_transition_aware"]
    labels = ["v1 future midpoint", "v2 pre-only directional"]
    colors = ["#6c757d", "#2d6a4f"]
    x = np.arange(len(transitions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for offset, (detector, label, color) in enumerate(zip(detectors, labels, colors)):
        rates = []
        for transition in transitions:
            match = next(
                row for row in intrinsic
                if row["detector"] == detector and row["transition"] == transition
            )
            rates.append(float(match["detection_rate"]))
        ax.bar(x + (offset - 0.5) * width, rates, width, label=label, color=color)
    ax.set_xticks(x, transitions)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detected view fraction")
    ax.set_title("Detector ablation on existing Gamma-World videos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "detector_ablation.png", dpi=170)
    plt.close(fig)


def plot_traces(results: Path) -> None:
    rows = [
        row for row in read_jsonl(results / "view_results.jsonl")
        if row["detector"] == "v2_preonly_transition_aware"
    ]
    signals = np.load(results / "signals.npz")
    trace_dir = results / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = f"{row['sample']}__v{row['view_index']}__{row['signal_name']}"
        if key not in signals:
            continue
        signal = signals[key]
        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        ax.plot(signal, lw=1.1, color="#2d6a4f")
        baseline_start, baseline_end = row["baseline_range"]
        search_start, search_end = row["search_range"]
        ax.axvspan(baseline_start, baseline_end, color="#6c757d", alpha=0.12, label="baseline")
        ax.axvspan(search_start, search_end, color="#2d6a4f", alpha=0.07, label="search")
        if row["target"] is not None:
            ax.axhline(row["target"], color="#ca6702", ls="--", label="target")
        if row["onset_signal_index"] is not None:
            ax.axvline(row["onset_signal_index"], color="black", ls=":", label="onset")
        ax.set_xlabel("flow signal index t")
        ax.set_ylabel(row["signal_name"])
        ax.set_title(
            f"{row['sample']} | view {row['view_index']} | "
            f"{row['transition']} | {row['status']}"
        )
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        safe_sample = str(row["sample"]).replace("/", "_")
        fig.savefig(trace_dir / f"{safe_sample}__v{row['view_index']}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    plot_summary(args.results)
    plot_traces(args.results)
    print(f"wrote detector plots -> {args.results}")


if __name__ == "__main__":
    main()
