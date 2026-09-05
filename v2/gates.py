"""Hash-bound stage gates shared by v2 generation and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifacts import ArtifactError, verify_artifact
from protocol import LoadedProtocol, canonical_sha256, file_sha256


class GateError(RuntimeError):
    """Raised when a requested stage is not authorized by valid evidence."""


def build_stage1_gate(
    protocol: LoadedProtocol,
    manifest_path: str | Path,
    pair_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
    canaries: list[dict[str, Any]],
    synthetic_report: Mapping[str, Any],
    false_positive_controls: int,
) -> dict[str, Any]:
    stage1_scenes = set(protocol.data["pilot"]["stage1_scenes"])
    stage1_pairs = [row for row in pair_rows if row["scene"] in stage1_scenes]
    stage1_views = [row for row in view_rows if row["scene"] in stage1_scenes]
    stage1_canaries = [row for row in canaries if row["scene"] in stage1_scenes]
    valid_pairs = [row for row in stage1_pairs if row.get("pair_valid") is True]
    latent_diagnostics = [
        row["latent_prefix_metrics"]
        for row in stage1_pairs
        if isinstance(row.get("latent_prefix_metrics"), Mapping)
        and row["latent_prefix_metrics"].get("status") == "ok"
    ]

    checks = {
        "has_stage1_pairs": bool(stage1_pairs),
        "all_stage1_pairs_valid": bool(stage1_pairs)
        and len(valid_pairs) == len(stage1_pairs),
        "synthetic_tests_passed": bool(synthetic_report.get("passed")),
        "decoded_prefixes_exact": bool(stage1_pairs)
        and all(row.get("prefix_pixel_max_error") == 0 for row in stage1_pairs),
        "latent_diagnostic_available": bool(latent_diagnostics),
        "latent_prefixes_exact_when_available": bool(latent_diagnostics)
        and all(row.get("prefix_max_error") == 0 for row in latent_diagnostics),
        "control_canary_exact": bool(stage1_canaries)
        and all(
            row.get("valid") is True
            and row.get("max_error") == 0
            and row.get("hash_match")
            for row in stage1_canaries
        ),
        "null_false_positives_within_limit": false_positive_controls
        <= int(protocol.data["validation"]["maximum_stage1_null_false_positives"]),
        "frame_counts_valid": bool(stage1_views)
        and all("unexpected_frame_count" not in row.get("qc_flags", []) for row in stage1_views),
        "finite_flow_valid": bool(stage1_views)
        and all("insufficient_finite_flow" not in row.get("qc_flags", []) for row in stage1_views),
    }
    required_names = (
        "has_stage1_pairs",
        "all_stage1_pairs_valid",
        "synthetic_tests_passed",
        "decoded_prefixes_exact",
        "control_canary_exact",
        "null_false_positives_within_limit",
        "frame_counts_valid",
        "finite_flow_valid",
    )
    block_reasons = [name for name in required_names if not checks[name]]
    allowed = not block_reasons
    report: dict[str, Any] = {
        "schema_version": "rtwm-v2-stage-gate-1",
        **protocol.identity,
        "phase": "stage1_to_stage2",
        "input_manifest_path": str(Path(manifest_path).resolve()),
        "input_manifest_sha256": file_sha256(manifest_path),
        "synthetic_evidence_sha256": synthetic_report.get("evidence_sha256"),
        "synthetic_evidence": dict(synthetic_report),
        "checks": checks,
        "required_checks": list(required_names),
        "stage2_allowed": allowed,
        "allowed": allowed,
        "block_reasons": block_reasons,
        "stage2_block_reason": None if allowed else "; ".join(block_reasons),
        "false_positive_control_views": false_positive_controls,
        "canaries": canaries,
        "n_pairs": len(pair_rows),
        "n_valid_pairs": sum(row.get("pair_valid") is True for row in pair_rows),
        "n_invalid_pairs": sum(row.get("pair_valid") is False for row in pair_rows),
        "n_views": len(view_rows),
    }
    report["gate_sha256"] = canonical_sha256(report)
    return report


def _gate_payload_without_hash(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "gate_sha256"}


def require_stage2_gate(
    gate_path: str | Path,
    protocol: LoadedProtocol,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(gate_path)
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot load stage2 gate {path}: {error}") from error
    if report.get("schema_version") != "rtwm-v2-stage-gate-1":
        raise GateError("stage2 gate uses an unsupported or legacy schema")
    if report.get("config_sha256") != protocol.file_sha256:
        raise GateError("stage2 gate config hash does not match the requested protocol")
    if report.get("protocol_sha256") != protocol.canonical_sha256:
        raise GateError("stage2 gate canonical protocol hash does not match")
    expected_gate_hash = report.get("gate_sha256")
    actual_gate_hash = canonical_sha256(_gate_payload_without_hash(report))
    if expected_gate_hash != actual_gate_hash:
        raise GateError("stage2 gate content hash is invalid")
    synthetic = report.get("synthetic_evidence")
    if not isinstance(synthetic, Mapping):
        raise GateError("stage2 gate has no embedded synthetic evidence")
    synthetic_hash = canonical_sha256(
        {key: value for key, value in synthetic.items() if key != "evidence_sha256"}
    )
    if (
        synthetic.get("evidence_sha256") != synthetic_hash
        or report.get("synthetic_evidence_sha256") != synthetic_hash
        or synthetic.get("config_sha256") != protocol.file_sha256
        or not synthetic.get("passed")
    ):
        raise GateError("stage2 gate synthetic evidence is stale or invalid")
    if not report.get("stage2_allowed") or not report.get("allowed"):
        reasons = report.get("block_reasons") or [report.get("stage2_block_reason")]
        raise GateError(f"stage2 is blocked: {', '.join(str(reason) for reason in reasons if reason)}")

    bound_manifest = Path(manifest_path or report.get("input_manifest_path", ""))
    if not bound_manifest.is_file():
        raise GateError(f"stage2 gate input manifest is missing: {bound_manifest}")
    actual_manifest_hash = file_sha256(bound_manifest)
    if actual_manifest_hash != report.get("input_manifest_sha256"):
        raise GateError("stage2 gate input manifest changed after gate evaluation")

    for metadata in report.get("evidence_artifacts", []):
        try:
            verify_artifact(path.parent, metadata, label="gate evidence")
        except ArtifactError as error:
            raise GateError(str(error)) from error
    return report


def require_confirmatory_gate(
    gate_path: str | Path,
    protocol: LoadedProtocol,
) -> dict[str, Any]:
    """Verify a calibration/RF-bound gate before confirmatory test generation."""
    path = Path(gate_path)
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot load confirmatory gate {path}: {error}") from error
    if report.get("schema_version") != "rtwm-v2-confirmatory-generation-gate-1":
        raise GateError("confirmatory gate uses an unsupported schema")
    if report.get("config_sha256") != protocol.file_sha256:
        raise GateError("confirmatory gate config file hash does not match")
    if report.get("protocol_sha256") != protocol.canonical_sha256:
        raise GateError("confirmatory gate protocol hash does not match")
    expected_hash = report.get("gate_sha256")
    actual_hash = canonical_sha256(_gate_payload_without_hash(report))
    if expected_hash != actual_hash:
        raise GateError("confirmatory gate content hash is invalid")
    if not report.get("allowed"):
        raise GateError(
            "confirmatory generation blocked: "
            + "; ".join(str(reason) for reason in report.get("block_reasons", []))
        )
    for metadata in report.get("evidence_artifacts", []):
        try:
            verify_artifact(path.parent, metadata, label="confirmatory gate evidence")
        except ArtifactError as error:
            raise GateError(str(error)) from error
    required = {
        "synthetic_tests_passed",
        "calibration_locked",
        "validation_zero_crossings",
        "receptive_field_passed",
        "raw_artifacts_required",
    }
    checks = report.get("checks", {})
    failed = sorted(name for name in required if not checks.get(name))
    if failed:
        raise GateError(f"confirmatory gate missing required checks: {failed}")
    return report


def invalid_reasons_from_checks(checks: Iterable[tuple[str, bool]]) -> list[str]:
    return [name for name, passed in checks if not passed]
