"""Hash-bound generation gate for fresh canonical STOP interventions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import atomic_write_json  # noqa: E402
from canonical_stop import load_stop_protocol, stop_specs, validate_canonical_evidence  # noqa: E402
from preflight_directional import gamma_source_identity  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "canonical_stop.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "canonical_stop")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "canonical_stop_preflight.json")
    args = parser.parse_args()
    protocol = load_stop_protocol(args.config)
    source = gamma_source_identity()
    evidence = validate_canonical_evidence(protocol, verify_artifacts=True)
    model = protocol.data["model"]
    manifest_path = args.root / "manifest.json"
    resume_valid = True
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        resume_valid = (
            manifest.get("schema_version") == "rtwm-v2-canonical-stop-rollouts-1"
            and manifest.get("config_sha256") == protocol.file_sha256
            and manifest.get("protocol_sha256") == protocol.canonical_sha256
            and manifest.get("planned_interventions") == 24
        )
    elif args.root.exists():
        resume_valid = not any(args.root.iterdir())
    checks = {
        "gamma_source_commit": source["commit"] == model["source_commit"],
        "gamma_source_diff": source["diff_sha256"] == model["source_diff_sha256"],
        "exact_24_interventions": len(stop_specs(protocol)) == 24,
        "fresh_output_root": resume_valid,
        **evidence["checks"],
    }
    blockers = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": "rtwm-v2-canonical-stop-preflight-1",
        **protocol.identity,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "output_root": str(args.root.resolve()),
        "checks": checks,
        "artifact_errors": evidence["artifact_errors"],
        "allowed": not blockers,
        "blockers": blockers,
        "evidence_hashes": {
            name: file_sha256(HERE / protocol.data["canonical_evidence"][name])
            for name in (
                "directional_config", "source_manifest", "null_gate",
                "source_validation", "decoder_rf_manifest",
            )
        },
    }
    report["gate_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
