"""Build deterministic proposal/refine summary, tables, plot, and manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3]
RESULTS = HERE / "results"
INPUTS = {
    "exact": RESULTS / "h200_exact.json",
    "unfused": RESULTS / "timing_samples_unfused.json",
    "fused": RESULTS / "timing_samples_fused.json",
    "horizon": RESULTS / "continuation_extension.json",
    "trace": RESULTS / "trace_replay.json",
}
PHASES = ("pre_roll", "first_roll", "steady_roll")
ARMS = ("stock_conditioning", "shallow_conditioning")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_bytes(value))


def _git_patch(repo: Path, paths: list[str]) -> bytes:
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "--", *paths],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _timing_rows(
    exact: dict[str, Any],
    unfused: dict[str, Any],
    fused: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for phase in PHASES:
        for arm in ARMS:
            if phase == "pre_roll":
                record = exact["profiles"][phase]["first_step_proposal_readiness"][
                    arm
                ]
                setting = "NVTE_FUSED_ATTN=0"
                timing = record["synchronized_host"]
                peak = record["max_peak_allocated_bytes"]
                exact_vs_baseline: bool | str = (
                    True if arm == "stock_conditioning" else "gated_by_exact_chain"
                )
            else:
                record = unfused["boundaries"][phase]
                setting = "NVTE_FUSED_ATTN=0"
                timing = record["summaries"][arm]["synchronized_host"]
                peak = max(
                    sample["peak_allocated_bytes"]
                    for sample in record["samples"][arm]
                )
                exact_vs_baseline = (
                    True if arm == "stock_conditioning" else "gated_by_exact_chain"
                )
            rows.append(
                {
                    "setting": setting,
                    "arm": arm,
                    "phase": phase,
                    "count": timing["count"],
                    "p50_ms": timing["p50_ms"],
                    "p90_ms": timing["p90_ms"],
                    "p95_ms": timing["p95_ms"],
                    "p99_ms": timing["p99_ms"],
                    "peak_allocated_bytes": peak,
                    "exact_vs_unfused_stock": exact_vs_baseline,
                    "retained": arm == "shallow_conditioning",
                }
            )
    for phase in ("first_roll", "steady_roll"):
        unfused_exact = unfused["boundaries"][phase]["one_block_exactness"]
        fused_exact = fused["boundaries"][phase]["one_block_exactness"]
        cross_setting_exact = all(
            fused_exact[name]["candidate_sha256"]
            == unfused_exact[name]["candidate_sha256"]
            for name in ("latent", "output", "live_kv")
        )
        for arm in ARMS:
            record = fused["boundaries"][phase]
            timing = record["summaries"][arm]["synchronized_host"]
            rows.append(
                {
                    "setting": "NVTE_FUSED_ATTN=1",
                    "arm": arm,
                    "phase": phase,
                    "count": timing["count"],
                    "p50_ms": timing["p50_ms"],
                    "p90_ms": timing["p90_ms"],
                    "p95_ms": timing["p95_ms"],
                    "p99_ms": timing["p99_ms"],
                    "peak_allocated_bytes": max(
                        sample["peak_allocated_bytes"]
                        for sample in record["samples"][arm]
                    ),
                    "exact_vs_unfused_stock": cross_setting_exact,
                    "retained": False,
                }
            )
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _svg_bytes(rows: list[dict[str, Any]]) -> bytes:
    plotted = [
        row
        for row in rows
        if row["phase"] in {"first_roll", "steady_roll"}
        and row["arm"] == "shallow_conditioning"
    ]
    width, height = 760, 360
    left, top, chart_height = 90, 35, 250
    maximum = 2000.0
    bar_width = 90
    gap = 55
    colors = {
        "NVTE_FUSED_ATTN=0": "#4976ba",
        "NVTE_FUSED_ATTN=1": "#d17b32",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="22" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">Single-H200 first-step proposal p95</text>',
    ]
    gate_y = top + chart_height * (1.0 - 750.0 / maximum)
    elements.append(
        f'<line x1="{left}" y1="{gate_y:.2f}" x2="730" y2="{gate_y:.2f}" '
        'stroke="#b22222" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    elements.append(
        f'<text x="725" y="{gate_y - 5:.2f}" text-anchor="end" '
        'font-family="sans-serif" font-size="12" fill="#b22222">750 ms gate</text>'
    )
    for index, row in enumerate(plotted):
        value = float(row["p95_ms"])
        x = left + 35 + index * (bar_width + gap)
        bar_height = min(chart_height, chart_height * value / maximum)
        y = top + chart_height - bar_height
        elements.extend(
            [
                f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" '
                f'height="{bar_height:.2f}" fill="{colors[row["setting"]]}"/>',
                f'<text x="{x + bar_width / 2}" y="{y - 6:.2f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="12">'
                f'{value:.1f}</text>',
                f'<text x="{x + bar_width / 2}" y="{top + chart_height + 18}" '
                'text-anchor="middle" font-family="sans-serif" font-size="11">'
                f'{row["phase"].replace("_", " ")}</text>',
                f'<text x="{x + bar_width / 2}" y="{top + chart_height + 33}" '
                'text-anchor="middle" font-family="sans-serif" font-size="10">'
                f'{row["setting"].split("=")[1]}</text>',
            ]
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top + chart_height}" x2="730" '
            f'y2="{top + chart_height}" stroke="black"/>',
            '<text x="18" y="170" transform="rotate(-90 18 170)" '
            'font-family="sans-serif" font-size="12">p95 synchronized host ms</text>',
            '<text x="380" y="350" text-anchor="middle" font-family="sans-serif" '
            'font-size="11">Fused measurements are shown but rejected for '
            'cross-setting byte exactness.</text>',
            "</svg>",
        ]
    )
    return ("\n".join(elements) + "\n").encode("utf-8")


def build() -> dict[str, Any]:
    loaded = {name: json.loads(path.read_text()) for name, path in INPUTS.items()}
    exact, unfused, fused = loaded["exact"], loaded["unfused"], loaded["fused"]
    rows = _timing_rows(exact, unfused, fused)
    table_path = RESULTS / "timings_by_arm.csv"
    plot_path = RESULTS / "proposal_p95.svg"
    atomic_bytes(table_path, _csv_bytes(rows))
    atomic_bytes(plot_path, _svg_bytes(rows))

    rtwm_patch_path = RESULTS / "provenance" / "rtwm_tracked.patch"
    gamma_patch_path = RESULTS / "provenance" / "gamma_world_tracked.patch"
    atomic_bytes(
        rtwm_patch_path,
        _git_patch(HERE.parents[1], ["driver_v2.py"]),
    )
    atomic_bytes(
        gamma_patch_path,
        _git_patch(
            CODE / "research" / "safeswm" / "external" / "Gamma-World",
            [
                "gamma_world/_src/gamma_world/inference/inference_i2v.py",
                "scripts/inference.py",
            ],
        ),
    )

    retained = [
        row
        for row in rows
        if row["retained"] and row["phase"] in {"first_roll", "steady_roll"}
    ]
    readiness_p95 = {row["phase"]: row["p95_ms"] for row in retained}
    stock = {
        row["phase"]: row
        for row in rows
        if row["setting"] == "NVTE_FUSED_ATTN=0"
        and row["arm"] == "stock_conditioning"
    }
    shallow = {
        row["phase"]: row
        for row in rows
        if row["setting"] == "NVTE_FUSED_ATTN=0"
        and row["arm"] == "shallow_conditioning"
    }
    exact_records = [
        item
        for gate in exact["exactness_gates"]
        for item in gate["continuation"]
        if not item.get("not_run")
    ]
    summary_report = {
        "schema_version": "rtwm-v2-proposal-refine-summary-1",
        "status": "complete",
        "scope": {
            "gpu": "NVIDIA H200",
            "device_count": 1,
            "batch_size": 1,
            "batch_size_2_tested": False,
            "multi_gpu_claim": False,
        },
        "exactness": {
            "executed_checkpoint_count": len(exact_records),
            "executed_checkpoint_all_exact": all(
                item["exact"] for item in exact_records
            ),
            "maximum_observed_error": 0,
            "steady_roll_eight_block_gate": loaded["horizon"]["status"],
            "all_requested_gates_passed": False,
            "reason": (
                "the stock tokenizer has 50 latent normalization slots; a "
                "steady-roll block-9 parent plus eight successors needs 51"
            ),
        },
        "readiness": {
            "gate_ms": 750.0,
            "retained_arm": "shallow_conditioning, NVTE_FUSED_ATTN=0",
            "p95_ms": readiness_p95,
            "passed": all(value <= 750.0 for value in readiness_p95.values()),
        },
        "measured_latency_delta_stock_minus_shallow_ms": {
            phase: {
                percentile: float(stock[phase][percentile])
                - float(shallow[phase][percentile])
                for percentile in ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
            }
            for phase in PHASES
        },
        "optimization_arms": {
            "shallow_conditioning": {
                "retained": True,
                "exactness": "executed exact chains and closure regression",
            },
            "cross_attention_projection_cache": exact["optimization_probes"][
                "cross_attention_projection_cache"
            ],
            "preallocated_sparse_hub_workspace": exact["optimization_probes"][
                "preallocated_sparse_hub_workspace"
            ],
            "h200_fused_attention": {
                "retained": False,
                "one_path_internal_exactness": True,
                "cross_setting_byte_exactness": False,
                "reason": (
                    "fused latent, output, and live-KV hashes differ from the "
                    "unfused baseline at both rolling boundaries"
                ),
            },
            "torch_compile_fullgraph": exact["optimization_probes"][
                "torch_compile_fullgraph"
            ],
            "cuda_graph": exact["optimization_probes"]["cuda_graph"],
        },
        "trace_replay": {
            "status": loaded["trace"]["status"],
            "proposal_ready": False,
            "final_commit_ready_reported_separately": True,
            "host_frame_ready_reported_separately": True,
            "all_work_charged": True,
            "canonical_stop_only_caol": True,
            "intent_semantic_gate_passed": False,
        },
        "claims": {
            "b1_overall_gate_passed": False,
            "b2_tested": False,
            "paper_edits_allowed": False,
            "best_exact_result_only": True,
        },
        "runtime_seconds": {
            "main_exact_and_profile": exact["gpu_runtime_seconds"],
            "unfused_timing_samples": unfused["runtime_seconds"],
            "fused_isolated_timing_samples": fused["runtime_seconds"],
            "prior_measured_trace_gpu_work": 89.76431649498409,
            "known_total": exact["gpu_runtime_seconds"]
            + unfused["runtime_seconds"]
            + fused["runtime_seconds"]
            + 89.76431649498409,
        },
        "artifacts": {
            "timing_table": str(table_path.relative_to(CODE)),
            "plot": str(plot_path.relative_to(CODE)),
            "trace": str(INPUTS["trace"].relative_to(CODE)),
            "exact_report": str(INPUTS["exact"].relative_to(CODE)),
        },
    }
    summary_path = RESULTS / "summary.json"
    atomic_json(summary_path, summary_report)

    artifact_paths = [
        HERE / "config.json",
        HERE / "__init__.py",
        HERE / "import_migration.json",
        HERE / "core.py",
        HERE / "conditioning.py",
        HERE / "gpu_experiment.py",
        HERE / "timing_samples.py",
        HERE / "continuation_extension.py",
        HERE / "trace_replay.py",
        HERE / "build_artifacts.py",
        HERE / "validate_artifacts.py",
        HERE / "test_proposal_refine.py",
        HERE / "README.md",
        HERE.parent / "conftest.py",
        *INPUTS.values(),
        table_path,
        plot_path,
        summary_path,
        rtwm_patch_path,
        gamma_patch_path,
    ]
    artifacts = [
        {
            "path": str(path.relative_to(CODE)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(artifact_paths))
    ]
    manifest = {
        "schema_version": "rtwm-v2-proposal-refine-manifest-1",
        "immutable_config_sha256": sha256_file(HERE / "config.json"),
        "artifacts": artifacts,
        "paper_files_modified_by_proposal_refine_work": False,
        "preexisting_workspace_paper_modifications_present": True,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(RESULTS / "manifest.json", manifest)
    return summary_report


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness_passed": report["readiness"]["passed"],
                "all_requested_gates_passed": report["exactness"][
                    "all_requested_gates_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
