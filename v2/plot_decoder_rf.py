"""Plot measured raw decoder support and MP4 contamination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    hybrids = read_jsonl(args.run / "hybrid_results.jsonl")
    diagnostic = json.loads((args.run / "mp4_diagnostic.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    for view in (0, 1):
        indices = [row["latent_index"] for row in hybrids]
        observed = [
            next(item for item in row["views"] if item["view_index"] == view)[
                "earliest_changed_frame"
            ]
            for row in hybrids
        ]
        ax.plot(indices, observed, "o-", ms=3, label=f"view {view}")
    expected = [4 * index - 3 for index in indices]
    ax.plot(indices, expected, "k--", label="expected 4t-3")
    ax.set_xlabel("perturbed latent index t")
    ax.set_ylabel("first changed raw frame")
    ax.set_title("(a) Raw decoder support")
    ax.legend(fontsize=8)

    ax = axes[1]
    raw = diagnostic["raw_stop_control"]["views"]
    encoded = diagnostic["mp4_stop_control"]["views"]
    labels = ["view 0", "view 1"]
    x = [0, 1]
    ax.bar(
        [value - 0.18 for value in x],
        [row["earliest_changed_frame"] for row in raw],
        0.36,
        label="raw uint8",
    )
    ax.bar(
        [value + 0.18 for value in x],
        [row["earliest_changed_frame"] for row in encoded],
        0.36,
        label="MP4 decode",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("first pair difference frame")
    ax.set_title("(b) Codec shifts differences earlier")
    ax.legend(fontsize=8)

    fig.tight_layout()
    output = args.run / "decoder_rf_summary.png"
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
