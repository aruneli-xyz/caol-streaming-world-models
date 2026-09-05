"""Build immutable summary, tables, plots, and manifest for the intent arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__:
    from . import gpu_experiment as experiment
    from .pre_roll_confidence import atomic_bytes, atomic_json
    from .validate_artifacts import source_hash_compatible
else:
    import gpu_experiment as experiment
    from pre_roll_confidence import atomic_bytes, atomic_json
    from validate_artifacts import source_hash_compatible

HERE = Path(__file__).resolve().parent


def _atomic_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, bbox_inches="tight")
        if path.suffix.lower() == ".svg":
            temporary_path = Path(temporary)
            normalized = "\n".join(
                line.rstrip() for line in temporary_path.read_text().splitlines()
            ) + "\n"
            temporary_path.write_text(normalized)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
        plt.close(figure)


def _csv_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _validate_inputs(
    *,
    config_path: Path,
    selection_path: Path,
    freeze_path: Path,
    semantic_path: Path,
    confidence_path: Path,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any],
    semantic: Mapping[str, Any],
    confidence: Mapping[str, Any],
) -> None:
    checks = (
        (
            selection["identity"]["config_sha256"],
            experiment.sha256_file(config_path),
            "selection config",
        ),
        (
            selection["identity"]["source_sha256"],
            HERE / "select_intent_heldout.py",
            "selection source",
        ),
        (
            freeze["identity"]["config_sha256"],
            experiment.sha256_file(config_path),
            "confidence freeze config",
        ),
        (
            freeze["identity"]["source_sha256"],
            HERE / "freeze_intent_confidence.py",
            "confidence freeze source",
        ),
        (
            semantic["identity"]["intent_config_sha256"],
            experiment.sha256_file(config_path),
            "semantic config",
        ),
        (
            semantic["identity"]["selection_file_sha256"],
            experiment.sha256_file(selection_path),
            "semantic selection",
        ),
        (
            semantic["identity"]["semantic_benchmark_source_sha256"],
            HERE / "intent_semantic_benchmark.py",
            "semantic source",
        ),
        (
            semantic["identity"]["intent_refine_source_sha256"],
            HERE / "intent_refine.py",
            "intent refine source",
        ),
        (
            confidence["identity"]["config_sha256"],
            experiment.sha256_file(config_path),
            "held-out confidence config",
        ),
        (
            confidence["identity"]["confidence_freeze_sha256"],
            experiment.sha256_file(freeze_path),
            "held-out confidence freeze",
        ),
        (
            confidence["identity"]["semantic_benchmark_sha256"],
            experiment.sha256_file(semantic_path),
            "held-out semantic benchmark",
        ),
        (
            confidence["identity"]["source_sha256"],
            HERE / "evaluate_intent_confidence.py",
            "held-out confidence source",
        ),
    )
    for expected, actual, label in checks:
        if isinstance(actual, Path):
            valid = source_hash_compatible(expected, actual)
        else:
            valid = expected == actual
        if not valid:
            raise RuntimeError(f"{label} self-hash validation failed")
    if not confidence_path.exists():
        raise RuntimeError("held-out confidence artifact is missing")


def build(output: Path) -> dict[str, Any]:
    config_path = HERE / "intent_pre_roll_config.json"
    selection_path = HERE / "manifests" / "intent_heldout_selection.json"
    freeze_path = HERE / "manifests" / "intent_confidence_freeze.json"
    semantic_path = HERE / "results" / "intent_semantic_benchmark.json"
    confidence_path = HERE / "results" / "intent_confidence_heldout.json"
    semantic = json.loads(semantic_path.read_text())
    confidence = json.loads(confidence_path.read_text())
    selection = json.loads(selection_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    _validate_inputs(
        config_path=config_path,
        selection_path=selection_path,
        freeze_path=freeze_path,
        semantic_path=semantic_path,
        confidence_path=confidence_path,
        selection=selection,
        freeze=freeze,
        semantic=semantic,
        confidence=confidence,
    )

    semantic_rows = []
    for row in semantic["samples"]:
        semantic_rows.append(
            {
                "sample_id": row["sample_id"],
                "episode_id": row["episode_id"],
                "block_index": row["block_index"],
                "intent": row["intent"],
                "movement_camera_group": row["movement_camera_group"],
                "final_latent_mse": row["final_latent_mse"],
                "continuation_final_latent_mse": row[
                    "continuation_final_latent_mse"
                ],
                "view_0_psnr_db": row["raw_uint8"]["by_view"][0]["psnr_db"],
                "view_1_psnr_db": row["raw_uint8"]["by_view"][1]["psnr_db"],
                "view_0_ssim": row["raw_uint8"]["by_view"][0]["ssim_mean"],
                "view_1_ssim": row["raw_uint8"]["by_view"][1]["ssim_mean"],
                "before_support_exact": row["raw_uint8"][
                    "before_support_exact"
                ],
                "proposal_ms": row["candidate_timing"][
                    "proposal_pre_action_ms"
                ],
            }
        )
    semantic_csv = HERE / "tables" / "intent_semantic_heldout.csv"
    atomic_bytes(semantic_csv, _csv_bytes(semantic_rows))

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sample_ids = [row["sample_id"] for row in semantic_rows]
    axes[0].plot(
        sample_ids,
        [row["final_latent_mse"] for row in semantic_rows],
        marker="o",
        label="final latent",
    )
    axes[0].axhline(0.024, color="black", linestyle="--", label="gate 0.024")
    axes[0].set_xlabel("held-out sample")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("One-step intent refinement")
    axes[0].legend()
    axes[1].plot(
        sample_ids,
        [row["continuation_final_latent_mse"] for row in semantic_rows],
        marker="o",
        color="#d95f02",
        label="four-block continuation",
    )
    axes[1].axhline(0.005, color="black", linestyle="--", label="gate 0.005")
    axes[1].set_xlabel("held-out sample")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Continuation drift")
    axes[1].legend()
    semantic_plot = HERE / "plots" / "intent_semantic_gate.svg"
    _atomic_figure(semantic_plot, figure)

    heldout = confidence["heldout"]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    names = ["all on-demand", "frozen intent gate"]
    values = [
        heldout["all_on_demand_work_ms"],
        heldout["total_charged_work_ms"],
    ]
    axis.bar(names, values, color=["#7570b3", "#1b9e77"])
    axis.set_ylabel("total charged work (ms)")
    axis.set_title(
        f"Frozen train-only policy: {confidence['frozen_operating_point']['mode']}"
    )
    axis.ticklabel_format(axis="y", style="plain")
    confidence_plot = HERE / "plots" / "intent_confidence_work.svg"
    _atomic_figure(confidence_plot, figure)

    positive = confidence["gates"]["positive_scoped_serving_claim"]
    report = {
        "schema_version": "rtwm-v2-intent-pre-roll-summary-1",
        "status": "complete",
        "scope": {
            "single_h200": True,
            "batch_size": 1,
            "pre_roll_only": True,
            "exact_and_rolling_gates_modified": False,
        },
        "selection": {
            "sample_count": selection["sample_count"],
            "candidate_count": selection["candidate_count"],
            "coverage": selection["coverage"],
            "selection_sha256": selection["identity"]["selection_sha256"],
        },
        "semantic": {
            "aggregate": semantic["aggregate"],
            "gate": semantic["gates"]["semantic_gate"],
        },
        "confidence": {
            "break_even_intent_precision": freeze["measured_cost_model"][
                "break_even_intent_precision"
            ],
            "frozen_operating_point": freeze["frozen_operating_point"],
            "heldout": heldout,
        },
        "gates": confidence["gates"],
        "positive_scoped_serving_claim": positive,
        "gpu_wall_runtime_seconds": semantic["gpu_wall_runtime_seconds"],
        "claims": {
            "approximate_intent_serving_claim": positive,
            "exact_b1_pre_roll_gate_unchanged": True,
            "rolling_cache_readiness": False,
            "directional_obedience": False,
            "perceptual_equivalence": False,
            "papers_modified_by_this_arm": [],
            "committed": False,
        },
    }
    report["identity"] = {
        "config_sha256": experiment.sha256_file(config_path),
        "selection_sha256": experiment.sha256_file(selection_path),
        "confidence_freeze_sha256": experiment.sha256_file(freeze_path),
        "semantic_benchmark_sha256": experiment.sha256_file(semantic_path),
        "confidence_heldout_sha256": experiment.sha256_file(confidence_path),
    }
    report["identity"]["identity_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(report["identity"])
    ).hexdigest()
    atomic_json(output, report)

    artifacts = [
        config_path,
        HERE / "__init__.py",
        HERE / "import_migration.json",
        selection_path,
        freeze_path,
        semantic_path,
        confidence_path,
        output,
        semantic_csv,
        HERE / "tables" / "intent_confidence_heldout.csv",
        semantic_plot,
        confidence_plot,
        HERE / "intent_refine.py",
        HERE / "select_intent_heldout.py",
        HERE / "freeze_intent_confidence.py",
        HERE / "intent_semantic_benchmark.py",
        HERE / "evaluate_intent_confidence.py",
        Path(__file__),
        HERE / "validate_artifacts.py",
        HERE / "test_proposal_refine.py",
        HERE / "README.md",
        HERE.parent / "conftest.py",
    ]
    manifest = {
        "schema_version": "rtwm-v2-intent-pre-roll-artifacts-1",
        "immutable": True,
        "files": [
            {
                "path": Path(os.path.relpath(path, HERE)).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": experiment.sha256_file(path),
            }
            for path in artifacts
        ],
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(manifest)
    ).hexdigest()
    atomic_json(HERE / "manifests" / "intent_pre_roll_artifacts.json", manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "intent_pre_roll_summary.json",
    )
    args = parser.parse_args()
    report = build(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": report["gates"],
                "positive_scoped_serving_claim": report[
                    "positive_scoped_serving_claim"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
