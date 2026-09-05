"""Validate every retained canonical artifact and final analysis bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import atomic_write_json, verify_rollout_artifacts  # noqa: E402
from directional_study import action_protocol_sha256, load_directional_protocol  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "directional_d0")
    parser.add_argument("--calibration", type=Path, default=HERE / "results" / "directional_d0_calibration")
    parser.add_argument("--audit-gate", type=Path, default=HERE / "results" / "directional_d0_audit_gate" / "gate_report.json")
    parser.add_argument("--held-out", type=Path, default=HERE / "results" / "directional_d0_held_out")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "directional_d0_validation.json")
    args = parser.parse_args()

    protocol = load_directional_protocol(args.config)
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical_hash = action_protocol_sha256(protocol)
    errors = []
    records = manifest.get("rollouts", [])
    for record in records:
        try:
            paths = verify_rollout_artifacts(
                args.root, record,
                required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
            )
            if record.get("action_protocol_sha256") != canonical_hash:
                errors.append(f"{record['rollout_id']}:action_protocol")
            with np.load(paths["action_tensors"]) as payload:
                expected = record["action_tensor_hashes"]
                for name in (
                    "generation_keyboard", "generation_camera",
                    "post_keyboard", "post_camera",
                ):
                    if array_hash(payload[name]) != expected[f"{name}_sha256"]:
                        errors.append(f"{record['rollout_id']}:{name}_hash")
        except Exception as error:
            errors.append(f"{record.get('rollout_id')}:{error}")
    interventions = [
        row for row in records if row.get("arm") in {"back", "yaw_positive"}
    ]
    calibration = json.loads((args.calibration / "gate_report.json").read_text())
    audit = json.loads(args.audit_gate.read_text())
    held_out = json.loads((args.held_out / "analysis_gate.json").read_text())
    checks = {
        "manifest_identity": (
            manifest.get("config_sha256") == protocol.file_sha256
            and manifest.get("protocol_sha256") == protocol.canonical_sha256
            and manifest.get("action_protocol_sha256") == canonical_hash
        ),
        "exactly_52_complete_records": len(records) == 52
        and all(row.get("status") == "complete" for row in records),
        "all_artifact_and_tensor_hashes_valid": not errors,
        "exactly_24_interventions": len(interventions) == 24,
        "all_intervention_prefixes_exact": all(
            row.get("latent_prefix_max_error") == 0
            and row.get("raw_prefix_max_errors") == [0.0, 0.0]
            for row in interventions
        ),
        "null_calibration_validation_passed": calibration.get("passed") is True,
        "audit_frozen_without_held_out": audit.get("forbidden_inputs", {}).get(
            "held_out_outcomes_used"
        ) is False,
        "held_out_32_views_complete": held_out.get("complete") is True
        and held_out.get("expected_views") == 32
        and held_out.get("observed_views") == 32,
    }
    report = {
        "schema_version": "rtwm-v2-directional-final-validation-1",
        **protocol.identity,
        "checks": checks,
        "passed": all(checks.values()),
        "errors": errors,
        "counts": {
            "records": len(records),
            "interventions": len(interventions),
        },
        "retained_generation_total_s": float(sum(row["total_s"] for row in records)),
        "retained_generation_h200_hours": float(
            sum(row["total_s"] for row in records) / 3600
        ),
        "artifacts": {
            "manifest_sha256": file_sha256(manifest_path),
            "calibration_gate_sha256": file_sha256(args.calibration / "gate_report.json"),
            "audit_gate_sha256": file_sha256(args.audit_gate),
            "held_out_analysis_sha256": file_sha256(args.held_out / "analysis_gate.json"),
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
