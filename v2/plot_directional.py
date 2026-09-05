"""Plot held-out causal divergence and gated directional obedience."""

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
    parser.add_argument("--results", type=Path, default=HERE / "results" / "directional_d0_held_out")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    statistics = json.loads((args.results / "statistics.json").read_text())
    transitions = list(statistics["transitions"])
    scenes = list(next(iter(statistics["transitions"].values())))
    labels = [scene.replace("build", "").replace("_normal", "").replace("_flat", "") for scene in scenes]
    x = np.arange(len(scenes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    colors = {"back": "#2d6a4f", "yaw_positive": "#9b2226"}
    for index, transition in enumerate(transitions):
        primary = [
            statistics["transitions"][transition][scene]["primary_detection_fraction"]["mean"]
            for scene in scenes
        ]
        obedience = [
            statistics["transitions"][transition][scene]["directional_obedience_fraction"]["mean"]
            for scene in scenes
        ]
        l2 = [
            statistics["transitions"][transition][scene]["paired_flow_field_l2_peak"]["mean"]
            for scene in scenes
        ]
        offset = (index - 0.5) * width
        axes[0].bar(x + offset, primary, width, color=colors[transition], label=transition)
        axes[1].bar(x + offset, obedience, width, color=colors[transition], label=transition)
        axes[2].bar(x + offset, l2, width, color=colors[transition], label=transition)
    for axis, title, ylabel in zip(
        axes,
        ("Causal divergence", "Directional obedience", "Continuous causal effect"),
        ("seed fraction", "seed fraction", "peak paired flow-field L2"),
    ):
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    output = args.output or args.results / "directional_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
