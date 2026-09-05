"""Freeze directional signs and audit-derived thresholds before held-out scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import artifact_metadata, atomic_write_json  # noqa: E402
from directional_study import load_directional_protocol  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--audit-results", type=Path, default=HERE / "results" / "directional_d0_audit" / "view_results.jsonl")
    parser.add_argument("--audit-analysis-gate", type=Path, default=HERE / "results" / "directional_d0_audit" / "analysis_gate.json")
    parser.add_argument("--null-calibration", type=Path, default=HERE / "results" / "directional_d0_calibration" / "gate_report.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "directional_d0_audit_gate" / "gate_report.json")
    args = parser.parse_args()

    protocol = load_directional_protocol(args.config)
    rows = read_jsonl(args.audit_results)
    expected = {
        (scene, seed, transition, view)
        for scene in protocol.data["design"]["scenes"]
        for seed in protocol.data["design"]["audit_seeds"]
        for transition in protocol.data["design"]["transitions"]
        for view in (0, 1)
    }
    observed = {
        (row["scene"], int(row["seed"]), row["transition"], int(row["view_index"]))
        for row in rows
        if row.get("partition") == "audit"
    }
    checks: dict[str, bool] = {
        "audit_partition_exact": observed == expected and len(rows) == len(expected),
        "held_out_seeds_absent": all(
            int(row["seed"]) not in protocol.data["design"]["held_out_seeds"] for row in rows
        ),
        "audit_analysis_complete": False,
        "null_calibration_and_validation_passed": False,
    }
    null_gate = json.loads(args.null_calibration.read_text())
    checks["null_calibration_and_validation_passed"] = (
        null_gate.get("schema_version") == "rtwm-v2-directional-null-gate-1"
        and null_gate.get("config_sha256") == protocol.file_sha256
        and null_gate.get("passed") is True
        and null_gate.get("checks", {}).get("validation_zero_strict_crossings") is True
    )
    analysis = json.loads(args.audit_analysis_gate.read_text())
    checks["audit_analysis_complete"] = (
        analysis.get("config_sha256") == protocol.file_sha256
        and analysis.get("partition") == "audit"
        and analysis.get("complete") is True
        and analysis.get("outputs", {}).get("view_results.jsonl") == file_sha256(args.audit_results)
    )

    thresholds: dict[str, dict[str, float]] = {}
    signs: dict[str, dict[str, int]] = {}
    for transition in protocol.data["design"]["transitions"]:
        endpoint = protocol.data["endpoints"][transition]
        signs[transition] = {
            name: int(sign) for name, sign in endpoint["expected_signs"].items()
        }
        selected = [row for row in rows if row["transition"] == transition]
        thresholds[transition] = {}
        for feature in endpoint["features"]:
            values = [
                float(row["feature_effects_expected_direction"][feature])
                for row in selected
            ]
            finite = [value for value in values if np.isfinite(value)]
            threshold = float(min(finite)) if len(finite) == len(values) and finite else float("nan")
            thresholds[transition][feature] = threshold
            checks[f"{transition}_{feature}_audit_expected_direction"] = bool(
                np.isfinite(threshold) and threshold > 0
            )

    block_reasons = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": "rtwm-v2-directional-audit-gate-1",
        **protocol.identity,
        "allowed": not block_reasons,
        "checks": checks,
        "block_reasons": block_reasons,
        "audit_results_sha256": file_sha256(args.audit_results),
        "audit_analysis_gate_sha256": file_sha256(args.audit_analysis_gate),
        "null_calibration_gate_sha256": file_sha256(args.null_calibration),
        "frozen": {
            "feature_definitions": protocol.data["endpoints"],
            "signs": signs,
            "threshold_rule": protocol.data["audit_gate"]["threshold_rule"],
            "persistence": int(protocol.data["endpoints"]["persistence"]),
            "acceptance_rules": protocol.data["audit_gate"],
        },
        "thresholds": thresholds,
        "forbidden_inputs": {
            "held_out_seeds": protocol.data["design"]["held_out_seeds"],
            "held_out_outcomes_used": False,
        },
        "evidence_artifacts": [
            artifact_metadata(args.audit_results.resolve()),
            artifact_metadata(args.audit_analysis_gate.resolve()),
            artifact_metadata(args.null_calibration.resolve()),
        ],
    }
    report["gate_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if block_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
