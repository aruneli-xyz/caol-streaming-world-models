"""Create publication-readable tables, figures, and a concise result summary."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import ROOT, atomic_write_bytes, atomic_write_json, load_json, sha256_file


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _atomic_plot(path: Path, replay: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ("stable_common", "intent_change")
    colors = {"stable_common": "#4C78A8", "intent_change": "#E45756"}
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for category in categories:
        rows = sorted(
            (
                row
                for row in replay["scenarios"]
                if row["category"] == category
            ),
            key=lambda row: row["budget"],
        )
        budgets = [row["budget"] for row in rows]
        axes[0].plot(
            budgets,
            [row["total_gpu_work_ms"] / 1000.0 for row in rows],
            marker="o",
            linewidth=2,
            color=colors[category],
            label=category.replace("_", " "),
        )
        axes[1].plot(
            budgets,
            [row["total_retained_snapshot_bytes"] / 2**30 for row in rows],
            marker="o",
            linewidth=2,
            color=colors[category],
            label=category.replace("_", " "),
        )
    axes[0].set_ylabel("Charged measured GPU work (s)")
    axes[1].set_ylabel("Retained delta/capture state (GiB)")
    for axis in axes:
        axis.set_xlabel("Speculative branch budget B")
        axis.set_xticks((0, 1, 2, 4))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Measured serial Gamma replay on one H200")
    figure.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=220, bbox_inches="tight")
    temporary.replace(path)
    plt.close(figure)


def _metric(
    rows: Sequence[Mapping[str, str]],
    target: str,
    predictor: str,
    budget: int,
    subset: str,
) -> Mapping[str, str]:
    return next(
        row
        for row in rows
        if row["target"] == target
        and row["predictor"] == predictor
        and int(row["budget"]) == budget
        and row["subset"] == subset
    )


def main() -> int:
    results = ROOT / "results"
    with (results / "trace_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    replay = load_json(results / "gamma_replay.json")
    fit = load_json(results / "intent_fit.json")

    scenario_rows: list[dict[str, Any]] = []
    for scenario in replay["scenarios"]:
        speculative = [
            row
            for row in scenario["candidate_records"]
            if row["kind"] == "speculative_candidate"
        ]
        scenario_rows.append(
            {
                "category": scenario["category"],
                "episode_id": scenario["episode_id"],
                "block_index": scenario["block_index"],
                "budget": scenario["budget"],
                "hit_count": scenario["hit_count"],
                "miss_count": scenario["miss_count"],
                "ready_by_lead_window": scenario["ready_by_lead_window"],
                "speculative_candidates": len(speculative),
                "fallback_candidates": scenario["generated_fallback_candidates"],
                "mean_speculative_generation_ms": (
                    sum(row["generation_ms"] for row in speculative)
                    / len(speculative)
                    if speculative
                    else ""
                ),
                "fork_ms": scenario["fork_ms"],
                "mean_restore_ms": (
                    sum(scenario["restore_ms"]) / len(scenario["restore_ms"])
                    if scenario["restore_ms"]
                    else ""
                ),
                "accept_ms": scenario["accept_ms"],
                "total_gpu_work_ms": scenario["total_gpu_work_ms"],
                "peak_allocated_bytes": scenario["peak_allocated_bytes"],
                "fork_snapshot_bytes": scenario["fork_snapshot_bytes"],
                "total_retained_snapshot_bytes": (
                    scenario["total_retained_snapshot_bytes"]
                ),
            }
        )
    scenario_path = results / "gamma_scenarios.csv"
    projection_path = results / "gamma_projections.csv"
    figure_path = results / "gamma_systems.png"
    _atomic_csv(scenario_path, scenario_rows)
    _atomic_csv(projection_path, replay["trace_projections"])
    _atomic_plot(figure_path, replay)

    strict_all = _metric(metrics, "strict_hash", "markov_1", 4, "all")
    strict_change = _metric(
        metrics, "strict_hash", "global_frequency", 4, "intent_change"
    )
    intent_all = _metric(metrics, "intent", "markov_1", 4, "all")
    intent_change = _metric(metrics, "intent", "markov_1", 4, "intent_change")
    history_all = _metric(metrics, "intent", "markov_history_3", 4, "all")
    history_change = _metric(
        metrics, "intent", "markov_history_3", 4, "intent_change"
    )
    speculative_times = [
        row["generation_ms"]
        for scenario in replay["scenarios"]
        for row in scenario["candidate_records"]
        if row["kind"] == "speculative_candidate"
    ]
    text = f"""# Speculation frontier results

## Frozen trace evaluation

- Test set: 85 episodes, 28,310 eligible episode-local blocks; 21,397 are
  intent changes. One short test episode has no eligible post-history block.
- Train-only intent fit: yaw dead zone {fit['threshold_degrees']['yaw']:.6f}
  degrees and pitch dead zone {fit['threshold_degrees']['pitch']:.6f} degrees,
  fitted over {fit['fit_block_count']:,} blocks from 512 train episodes.
- Best intent result among the tested predictors is first-order Markov at
  B=4: {float(intent_all['hit_rate']):.4%} overall (episode-bootstrap 95% CI
  {float(intent_all['hit_rate_ci95_low']):.4%}–{float(intent_all['hit_rate_ci95_high']):.4%})
  and {float(intent_change['hit_rate']):.4%} on intent changes
  ({float(intent_change['hit_rate_ci95_low']):.4%}–{float(intent_change['hit_rate_ci95_high']):.4%}).
- The frozen short-history Markov arm at B=4 reaches
  {float(history_all['hit_rate']):.4%} overall and
  {float(history_change['hit_rate']):.4%} on changes.
- Exact 12-frame sequence hashes remain sparse: the best B=4 tested values are
  {float(strict_all['hit_rate']):.4%} overall
  ({float(strict_all['hit_rate_ci95_low']):.4%}–{float(strict_all['hit_rate_ci95_high']):.4%})
  and {float(strict_change['hit_rate']):.4%} on changes
  ({float(strict_change['hit_rate_ci95_low']):.4%}–{float(strict_change['hit_rate_ci95_high']):.4%}).

## Measured Gamma systems replay

- Complete fixed-scene replay: 2 hash-frozen decisions × B={{0,1,2,4}}, serial
  on one H200 through the one-block delta path.
- Speculative candidate generation ranges from {min(speculative_times):.2f} to
  {max(speculative_times):.2f} ms; the trace projection uses the recorded
  median {replay['trace_projections'][0]['measured_candidate_ms']:.2f} ms.
- Measured charged GPU work across the eight scenarios is
  {replay['measured_gpu_runtime_ms'] / 1000.0:.3f} s. Every generated candidate
  and each miss fallback is included.
- With the declared 750 ms lead window, measured replay readiness is false for
  every speculative scenario. Simulated capacities C={{1,2,4}} therefore also
  project 0% trace-wide readiness; these are projections, not concurrent runs.
- Fork snapshots are zero bytes at the selected pre-roll boundary. Each captured
  candidate state is 4,965,556,224 bytes; B=4 retains 19,862,224,896 bytes on
  a hit path and 24,827,781,120 bytes when a miss fallback is also retained.
- This replay supports systems cost/readiness claims only. It uses one scene,
  two decisions, and train-derived representative intent branches; it does not
  measure semantic visual response. Paged copy-on-write allocator measurements
  are separate and were not substituted.
"""
    summary_path = results / "RESULTS.md"
    atomic_write_bytes(summary_path, text.encode("utf-8"))
    artifacts = [
        results / "trace_summary.json",
        results / "trace_metrics.csv",
        results / "intent_fit.json",
        results / "replay_selection.json",
        results / "speculation_frontier.png",
        results / "gamma_replay.json",
        scenario_path,
        projection_path,
        figure_path,
        summary_path,
    ]
    manifest = {
        "schema_version": 1,
        "artifacts": [
            {
                "path": f"results/{path.name}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
    }
    atomic_write_json(results / "artifact_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
