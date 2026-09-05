"""CPU-only, hash-bound preflight for the bounded directional study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GAMMA = HERE.parents[1] / "safeswm" / "external" / "Gamma-World"
SOLARIS_ACTIONS = HERE.parents[1] / "safeswm" / "external" / "solaris" / "src" / "data" / "minecraft.py"

sys.path.insert(0, str(HERE))

from artifacts import atomic_write_json  # noqa: E402
from directional_study import (  # noqa: E402
    action_protocol_sha256,
    action_sequence,
    load_directional_protocol,
    rollout_specs,
)
from protocol import canonical_sha256, file_sha256  # noqa: E402


def gamma_source_identity() -> dict[str, Any]:
    source_paths = ("gamma_world", "packages", "scripts", "pyproject.toml", "uv.lock")

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", *arguments], cwd=GAMMA, stderr=subprocess.STDOUT
        ).strip()

    commit = git("rev-parse", "HEAD").decode()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *source_paths], cwd=GAMMA
    )
    untracked = git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", *source_paths
    ).decode().splitlines()
    digest = hashlib.sha256()
    digest.update(diff)
    for entry in sorted(line for line in untracked if line.startswith("?? ")):
        relative = entry[3:]
        digest.update(relative.encode())
        path = GAMMA / relative
        if path.is_file():
            digest.update(path.read_bytes())
    return {"commit": commit, "dirty": bool(diff or untracked), "diff_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "directional_d0_preflight.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "directional_d0")
    args = parser.parse_args()

    protocol = load_directional_protocol(args.config)
    source = gamma_source_identity()
    model = protocol.data["model"]
    checks = {
        "gamma_source_commit": source["commit"] == model["source_commit"],
        "gamma_source_diff": source["diff_sha256"] == model["source_diff_sha256"],
        "solaris_action_source": file_sha256(SOLARIS_ACTIONS)
        == protocol.data["action_protocol"]["solaris_source_sha256"],
        "exactly_52_fresh_rollouts": len(rollout_specs(protocol)) == 52,
        "exactly_24_interventions": sum(
            row["arm"] in {"back", "yaw_positive"} for row in rollout_specs(protocol)
        ) == 24,
    }
    back = action_sequence(protocol, "back")
    yaw = action_sequence(protocol, "yaw_positive")
    change = int(model["change_frame"])
    checks.update({
        "canonical_forward_index_11": back["keyboard"][0][11] == 1.0
        and sum(back["keyboard"][0]) == 1.0,
        "canonical_back_index_12": back["keyboard"][change][12] == 1.0
        and sum(back["keyboard"][change]) == 1.0,
        "yaw_uses_nonzero_camera": yaw["camera"][change] == [6.0, 0.0],
        "yaw_is_not_keyboard_strafe": yaw["keyboard"][change][13:15] == [0.0, 0.0],
    })

    evidence_checks: dict[str, bool] = {}
    for name, key in (
        ("confirmatory_gate_hash", "confirmatory_gate"),
        ("decoder_rf_manifest_hash", "decoder_rf_manifest"),
    ):
        path = HERE / protocol.data["evidence"][key]
        evidence_checks[name] = path.is_file() and file_sha256(path) == protocol.data["evidence"][f"{key}_sha256"]
    checks.update(evidence_checks)

    manifest_path = args.root / "manifest.json"
    root_state: dict[str, Any] = {
        "path": str(args.root.resolve()),
        "manifest_exists": manifest_path.exists(),
        "legacy_inputs_allowed": False,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        root_state["manifest_sha256"] = file_sha256(manifest_path)
        checks["fresh_root_resume_identity"] = (
            manifest.get("schema_version") == "rtwm-v2-directional-rollouts-2"
            and manifest.get("config_sha256") == protocol.file_sha256
            and manifest.get("protocol_sha256") == protocol.canonical_sha256
            and manifest.get("action_protocol_sha256") == action_protocol_sha256(protocol)
            and manifest.get("planned_rollouts") == 52
            and not manifest.get("legacy_control_manifest")
        )
    else:
        checks["fresh_root_resume_identity"] = (
            not args.root.exists() or not any(args.root.iterdir())
        )
    blockers = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": "rtwm-v2-directional-preflight-1",
        **protocol.identity,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "checks": checks,
        "fresh_root": root_state,
        "allowed": not blockers,
        "blockers": blockers,
        "note": (
            "Only the independent 52-rollout canonical root is authorized; "
            "legacy action records are forbidden."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
