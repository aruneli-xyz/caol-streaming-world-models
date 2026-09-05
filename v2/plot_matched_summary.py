"""Create compact summary plots for the matched-counterfactual pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


V2 = Path(__file__).resolve().parent
DEFAULT_RESULTS = V2 / "results" / "matched_scoring"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = read_jsonl(args.results / "view_results.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = {0: "#2d6a4f", 1: "#ca6702"}

    ax = axes[0]
    for delay in (0, 1):
        selected = [row for row in rows if row["delay_blocks"] == delay]
        x = [row["admission_frame"] for row in selected]
        y = [row["earliest_pixel_difference_frame"] for row in selected]
        ax.scatter(x, y, label=f"delay {delay}", color=colors[delay], alpha=0.8)
    limits = [50, 112]
    ax.plot(limits, limits, "k--", lw=1, label="admission frame")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("nominal admission frame")
    ax.set_ylabel("earliest decoded pixel difference")
    ax.set_title("(a) Decoder look-ahead")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [f"s{row['seed']} d{row['delay_blocks']} v{row['view_index']}" for row in rows]
    ratios = [row["paired"]["max_to_target_ratio"] or 0 for row in rows]
    order = np.argsort(ratios)[::-1]
    ordered_ratios = [ratios[index] for index in order]
    ordered_labels = [labels[index] for index in order]
    ax.bar(np.arange(len(rows)), ordered_ratios, color="#6c757d")
    ax.axhline(1.0, color="#ae2012", ls="--", label="detection threshold")
    ax.set_xticks(np.arange(len(rows)), ordered_labels, rotation=65, ha="right", fontsize=6.5)
    ax.set_ylabel("max paired contrast / target")
    ax.set_title("(b) No paired threshold crossing")
    ax.legend(fontsize=8)

    fig.tight_layout()
    output = args.results / "matched_summary.png"
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
