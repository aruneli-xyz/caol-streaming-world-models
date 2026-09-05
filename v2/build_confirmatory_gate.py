"""Build the hash-bound gate for confirmatory intervention generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from artifacts import artifact_metadata, atomic_write_json, verify_artifact
from protocol import canonical_sha256, load_protocol


HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def payload_hash_valid(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    return expected == canonical_sha256(body)


def verify_calibration(
    bundle: Path,
    config_sha256: str,
) -> tuple[bool, list[Path], dict[str, Any]]:
    gate_path = bundle / "gate_report.json"
    gate = load_json(gate_path)
    locks = sorted((bundle / "locks").glob("*.json"))
    passed = (
        gate.get("passed") is True
        and gate.get("config_sha256") == config_sha256
        and payload_hash_valid(gate, "payload_sha256")
        and len(locks) == 4
    )
    expected_locks = gate.get("lock_payload_sha256", {})
    for lock_path in locks:
        payload = load_json(lock_path)
        passed = (
            passed
            and payload_hash_valid(payload, "payload_sha256")
            and expected_locks.get(lock_path.name) == payload.get("payload_sha256")
        )
    return passed, [gate_path, *locks], gate


def verify_rf(
    manifest_path: Path,
    config_sha256: str,
    required_indices: list[int],
) -> tuple[dict[str, bool], list[Path], dict[str, Any]]:
    manifest = load_json(manifest_path)
    run_dir = manifest_path.parent
    artifacts = manifest.get("artifacts", {})

    def artifact_path(name: str) -> Path:
        metadata = artifacts[name]
        return verify_artifact(run_dir, metadata, label=f"RF {name}")

    full_path = artifact_path("full_stop_control")
    determinism_path = artifact_path("determinism")
    hybrid_path = artifact_path("hybrid_results")
    mp4_path = artifact_path("mp4_diagnostic")
    full = load_json(full_path)
    determinism = load_json(determinism_path)
    hybrids = [
        json.loads(line)
        for line in hybrid_path.read_text().splitlines()
        if line.strip()
    ]
    mp4 = load_json(mp4_path)

    by_index = {int(row["latent_index"]): row for row in hybrids}
    required_present = all(index in by_index for index in required_indices)
    selected = [by_index[index] for index in required_indices if index in by_index]
    no_early = required_present and all(
        view.get("early_changed_frame_count") == 0
        for row in selected
        for view in row["views"]
    )
    first_frames = [
        min(
            view["earliest_changed_frame"]
            for view in by_index[index]["views"]
            if view["earliest_changed_frame"] is not None
        )
        for index in required_indices
        if index in by_index
    ]
    monotone = required_present and first_frames == sorted(first_frames)
    full_no_early = all(
        view.get("early_changed_frame_count") == 0
        for view in full["views"]
    )
    checks = {
        "rf_manifest_complete": manifest.get("status") == "complete",
        "rf_config_matches": manifest.get("source", {}).get("config", {}).get("sha256")
        == config_sha256,
        "decoder_deterministic": determinism.get("exact") is True,
        "required_indices_present": required_present,
        "no_raw_effect_before_expected_support": no_early,
        "full_pair_no_raw_effect_before_support": full_no_early,
        "support_monotone": monotone,
        "mp4_diagnostic_complete": mp4.get("status") == "ok",
    }
    return checks, [manifest_path, full_path, determinism_path, hybrid_path, mp4_path], manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "confirmatory.json")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rf-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "confirmatory_gate" / "gate_report.json",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    preflight = load_json(args.preflight)
    preflight_passed = (
        preflight.get("passed") is True
        and preflight.get("config_sha256") == protocol.file_sha256
        and preflight.get("evidence_sha256")
        == canonical_sha256(
            {key: value for key, value in preflight.items() if key != "evidence_sha256"}
        )
    )
    calibration_passed, calibration_paths, calibration_gate = verify_calibration(
        args.calibration,
        protocol.file_sha256,
    )
    required_indices = [
        int(index) for index in protocol.data["receptive_field"]["probe_latent_indices"]
    ]
    rf_checks, rf_paths, _ = verify_rf(
        args.rf_manifest,
        protocol.file_sha256,
        required_indices,
    )
    checks = {
        "synthetic_tests_passed": preflight_passed,
        "calibration_locked": calibration_passed,
        "validation_zero_crossings": calibration_gate.get("checks", {}).get(
            "seed104_has_zero_strict_crossings"
        )
        is True,
        "receptive_field_passed": all(rf_checks.values()),
        "raw_artifacts_required": set(protocol.data["validation"]["required_artifacts"])
        == {"latent", "decoded_u8", "detector_frames"},
        **rf_checks,
    }
    block_reasons = sorted(name for name, passed in checks.items() if not passed)
    evidence_paths = [args.preflight, *calibration_paths, *rf_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = [
        artifact_metadata(path.resolve())
        for path in evidence_paths
    ]
    report: dict[str, Any] = {
        "schema_version": "rtwm-v2-confirmatory-generation-gate-1",
        **protocol.identity,
        "allowed": not block_reasons,
        "checks": checks,
        "block_reasons": block_reasons,
        "evidence_artifacts": evidence,
        "calibration_payload_sha256": calibration_gate.get("payload_sha256"),
        "required_rf_indices": required_indices,
    }
    report["gate_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if block_reasons:
        raise RuntimeError(f"confirmatory gate blocked: {block_reasons}")


if __name__ == "__main__":
    main()
