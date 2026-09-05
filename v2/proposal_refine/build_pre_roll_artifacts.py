"""Build immutable artifacts for the separately scoped pre-roll B=1 gate."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from .validate_artifacts import source_hash_compatible
else:
    from validate_artifacts import source_hash_compatible


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3]
RESULTS = HERE / "results"
CONFIG = HERE / "pre_roll_b1_config.json"
BENCHMARK = RESULTS / "pre_roll_b1_benchmark.json"
CONFIDENCE = RESULTS / "pre_roll_confidence.json"
CONFIDENCE_CSV = RESULTS / "pre_roll_confidence_frontier.csv"
GLOBAL_SUMMARY = RESULTS / "summary.json"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def validate_inputs(
    benchmark: Mapping[str, Any], confidence: Mapping[str, Any]
) -> None:
    checks = (
        (
            benchmark["identity"]["pre_roll_config_sha256"],
            sha256_file(CONFIG),
            "benchmark config",
        ),
        (
            benchmark["identity"]["benchmark_source_sha256"],
            HERE / "pre_roll_b1_benchmark.py",
            "benchmark source",
        ),
        (
            confidence["identity"]["config_sha256"],
            sha256_file(CONFIG),
            "confidence config",
        ),
        (
            confidence["identity"]["benchmark_sha256"],
            sha256_file(BENCHMARK),
            "confidence benchmark",
        ),
        (
            confidence["identity"]["source_sha256"],
            HERE / "pre_roll_confidence.py",
            "confidence source",
        ),
    )
    for expected, actual, label in checks:
        if isinstance(actual, Path):
            valid = source_hash_compatible(expected, actual)
        else:
            valid = expected == actual
        if not valid:
            raise RuntimeError(f"{label} self-hash validation failed")


def _timing_csv(benchmark: Mapping[str, Any]) -> bytes:
    rows = []
    for path, summary in benchmark["summaries"].items():
        for metric, values in summary.items():
            if isinstance(values, Mapping) and "p50_ms" in values:
                rows.append(
                    {
                        "path": path,
                        "metric": metric,
                        "count": values["count"],
                        "p50_ms": values["p50_ms"],
                        "p95_ms": values["p95_ms"],
                        "p99_ms": values["p99_ms"],
                    }
                )
    for metric, values in benchmark["known_hit_savings"].items():
        rows.append(
            {
                "path": "known_hit_savings",
                "metric": metric,
                "count": values["count"],
                "p50_ms": values["p50_ms"],
                "p95_ms": values["p95_ms"],
                "p99_ms": values["p99_ms"],
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _frontier_svg(confidence: Mapping[str, Any]) -> bytes:
    width, height = 760, 430
    left, top, chart_w, chart_h = 75, 45, 620, 300
    colors = {"strict_hash": "#3569a8", "intent": "#d17b32"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">Frozen pre-roll B=1 confidence frontier</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" '
        f'y2="{top + chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" '
        'stroke="black"/>',
    ]
    for tick in range(0, 101, 20):
        x = left + chart_w * tick / 100
        y = top + chart_h * (1 - tick / 100)
        elements.extend(
            [
                f'<text x="{x:.1f}" y="{top + chart_h + 18}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10">{tick}%</text>',
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="10">{tick}%</text>',
            ]
        )
    for target, record in confidence["targets"].items():
        points = []
        for row in record["frontier"]:
            test = row["test_pre_roll"]
            precision = test["exact_hit_precision"]
            if precision is None:
                continue
            x = left + chart_w * float(test["coverage"])
            y = top + chart_h * (1 - float(precision))
            points.append((x, y))
        if points:
            elements.append(
                '<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(
                    colors[target],
                    " ".join(f"{x:.2f},{y:.2f}" for x, y in points),
                )
            )
            for x, y in points:
                elements.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" '
                    f'fill="{colors[target]}"/>'
                )
    elements.extend(
        [
            '<text x="385" y="395" text-anchor="middle" font-family="sans-serif" '
            'font-size="12">Pre-roll test coverage</text>',
            '<text x="18" y="195" transform="rotate(-90 18 195)" '
            'font-family="sans-serif" font-size="12">Exact token precision</text>',
            '<rect x="520" y="365" width="12" height="12" fill="#3569a8"/>',
            '<text x="538" y="375" font-family="sans-serif" font-size="11">'
            'strict exact hash</text>',
            '<rect x="520" y="385" width="12" height="12" fill="#d17b32"/>',
            '<text x="538" y="395" font-family="sans-serif" font-size="11">'
            'intent diagnostic only</text>',
            "</svg>",
        ]
    )
    return ("\n".join(elements) + "\n").encode()


def _artifact_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(CODE)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    ]


def build() -> dict[str, Any]:
    benchmark = json.loads(BENCHMARK.read_text())
    confidence = json.loads(CONFIDENCE.read_text())
    global_summary = json.loads(GLOBAL_SUMMARY.read_text())
    validate_inputs(benchmark, confidence)
    exact = confidence["targets"]["strict_hash"]
    intent = confidence["targets"]["intent"]
    timing_csv = RESULTS / "pre_roll_b1_timings.csv"
    plot = RESULTS / "pre_roll_confidence_frontier.svg"
    atomic_bytes(timing_csv, _timing_csv(benchmark))
    atomic_bytes(plot, _frontier_svg(confidence))

    summary = {
        "schema_version": "rtwm-v2-pre-roll-b1-summary-1",
        "status": "complete",
        "pre_roll_b1_gate": benchmark["gates"]["pre_roll_b1_gate"],
        "measured": {
            "known_hit_savings": benchmark["known_hit_savings"],
            "paths": benchmark["summaries"],
            "gpu_wall_runtime_seconds": benchmark["gpu_wall_runtime_seconds"],
            "exactness": benchmark["exactness"],
        },
        "confidence_gate": {
            "eligibility": confidence["eligibility"],
            "strict_exact_hash": {
                "operating_threshold": exact["operating_threshold"],
                "test_pre_roll": exact["operating_point"]["test_pre_roll"],
                "expected_charged_work": exact["operating_point"][
                    "expected_pre_roll_charged_work"
                ],
            },
            "intent_diagnostic": {
                "operating_threshold": intent["operating_threshold"],
                "test_pre_roll": intent["operating_point"]["test_pre_roll"],
                "exact_acceptance": False,
                "caol": False,
            },
        },
        "global_gate": {
            "rolling_cache_readiness_passed": False,
            "unchanged": True,
            "first_roll_p95_ms": global_summary["readiness"]["p95_ms"][
                "first_roll"
            ],
            "steady_roll_p95_ms": global_summary["readiness"]["p95_ms"][
                "steady_roll"
            ],
            "overall_b1_gate_passed": False,
            "b2_tested": False,
        },
        "claims": {
            "exact_b1_pre_roll_proposal_readiness_works": benchmark["gates"][
                "pre_roll_b1_gate"
            ]["passed"],
            "rolling_cache_readiness_works": False,
            "end_to_end_benefit_depends_on_exact_hit_coverage_and_confidence": True,
            "intent_semantic_fidelity_passed": False,
            "papers_may_be_edited": False,
            "multi_gpu_claim": False,
        },
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "benchmark_sha256": sha256_file(BENCHMARK),
            "confidence_sha256": sha256_file(CONFIDENCE),
            "global_summary_sha256": sha256_file(GLOBAL_SUMMARY),
            "builder_source_sha256": sha256_file(Path(__file__)),
        },
    }
    summary["identity"]["identity_sha256"] = hashlib.sha256(
        canonical_bytes(summary["identity"])
    ).hexdigest()
    summary_path = RESULTS / "pre_roll_b1_summary.json"
    atomic_json(summary_path, summary)
    paths = [
        CONFIG,
        HERE / "__init__.py",
        HERE / "import_migration.json",
        HERE / "pre_roll_b1_benchmark.py",
        HERE / "pre_roll_confidence.py",
        HERE / "build_pre_roll_artifacts.py",
        HERE / "validate_artifacts.py",
        HERE / "test_proposal_refine.py",
        HERE / "README.md",
        HERE.parent / "conftest.py",
        BENCHMARK,
        CONFIDENCE,
        CONFIDENCE_CSV,
        GLOBAL_SUMMARY,
        timing_csv,
        plot,
        summary_path,
    ]
    manifest = {
        "schema_version": "rtwm-v2-pre-roll-b1-manifest-1",
        "immutable_config_sha256": sha256_file(CONFIG),
        "artifacts": _artifact_rows(paths),
        "global_rolling_gate_unchanged": True,
        "b2_ready_claim": False,
        "paper_files_modified_by_this_extension": False,
        "preexisting_workspace_paper_modifications_present": True,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_bytes(manifest)
    ).hexdigest()
    atomic_json(RESULTS / "pre_roll_b1_manifest.json", manifest)
    return summary


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "pre_roll_passed": report["pre_roll_b1_gate"]["passed"],
                "rolling_passed": report["global_gate"][
                    "rolling_cache_readiness_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
