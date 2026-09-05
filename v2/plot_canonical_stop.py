"""Plot canonical STOP reliability, effects, and throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results" / "canonical_stop_scoring")
    args = parser.parse_args()
    statistics = json.loads((args.results / "statistics.json").read_text())
    delays = ["d0", "d1"]
    colors = {"d0": "#2d6a4f", "d1": "#ca6702"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    metrics = (
        ("both_view_detection_fraction", "both-view detection fraction"),
        ("flow_divergence_peak", "peak paired flow-field L2"),
        ("effective_generation_fps", "effective generation FPS"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        means = [statistics["by_delay"][delay]["metrics"][metric]["mean"] for delay in delays]
        intervals = [statistics["by_delay"][delay]["metrics"][metric]["ci95"] for delay in delays]
        errors = np.array([
            [mean - interval[0] for mean, interval in zip(means, intervals)],
            [interval[1] - mean for mean, interval in zip(means, intervals)],
        ])
        axis.bar(delays, means, color=[colors[delay] for delay in delays], yerr=errors, capsize=4)
        axis.set_ylabel(ylabel)
        axis.set_title(metric.replace("_", " ").title())
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    output = args.results / "canonical_stop_summary.png"
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
