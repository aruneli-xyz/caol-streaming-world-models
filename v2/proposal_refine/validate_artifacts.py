"""Read-only strict resume and artifact-manifest validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3]
RTWM = HERE.parents[1]
GAMMA = CODE / "research" / "safeswm" / "external" / "Gamma-World"
MODELS = CODE / "research" / "safeswm" / "models"
IMPORT_MIGRATION = HERE / "import_migration.json"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash_compatible(recorded: str, path: Path) -> bool:
    current = sha256_file(path)
    if current == recorded:
        return True
    migration = json.loads(IMPORT_MIGRATION.read_text())
    return any(
        entry["path"] == path.relative_to(HERE).as_posix()
        and entry["recorded_sha256"] == recorded
        and entry["current_sha256"] == current
        for entry in migration["migrations"]
    )


def _identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(identity)
    payload.pop("identity_sha256", None)
    return payload


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_identity(path: Path) -> dict[str, Any]:
    commit = _git_commit(path)
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def strict_resume_check(
    measured_path: Path = HERE / "results" / "h200_exact.json",
) -> dict[str, Any]:
    """Validate the completed run against its actual execution dependencies.

    The RTWM commit is checked, while its whole-worktree diff hash is not:
    every RTWM file used by this run is already checked byte-for-byte below.
    This prevents unrelated tracked edits from invalidating a completed run.
    """

    measured_sha = sha256_file(measured_path)
    report = json.loads(measured_path.read_text())
    if report.get("status") != "complete":
        raise RuntimeError("strict no-op resume requires a complete artifact")
    identity = report["identity"]
    recorded_identity_sha = identity["identity_sha256"]
    computed_identity_sha = hashlib.sha256(
        canonical_bytes(_identity_payload(identity))
    ).hexdigest()
    if computed_identity_sha != recorded_identity_sha:
        raise RuntimeError("recorded execution identity self-hash mismatch")
    if report["identity_sha256"] != recorded_identity_sha:
        raise RuntimeError("top-level and embedded execution identities differ")

    source_paths = {
        "conditioning.py": HERE / "conditioning.py",
        "config.json": HERE / "config.json",
        "core.py": HERE / "core.py",
        "driver_v2.py": RTWM / "driver_v2.py",
        "gpu_experiment.py": HERE / "gpu_experiment.py",
    }
    current_source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    if any(
        not source_hash_compatible(identity["sources"].get(name, ""), path)
        for name, path in source_paths.items()
    ):
        changed = sorted(
            name
            for name, path in source_paths.items()
            if not source_hash_compatible(identity["sources"].get(name, ""), path)
        )
        raise RuntimeError(f"strict resume source mismatch: {changed}")

    model_paths = {
        "checkpoint": (
            MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"
        ),
        "tokenizer": MODELS / "gamma-world" / "tokenizer.pth",
    }
    for name, path in model_paths.items():
        recorded = identity["models"][name]
        if (
            sha256_file(path) != recorded["sha256"]
            or path.stat().st_size != recorded["bytes"]
        ):
            raise RuntimeError(f"strict resume {name} mismatch")

    environment = identity["environment"]
    current_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "nvte_fused_attn": os.environ.get("NVTE_FUSED_ATTN"),
    }
    if current_environment != environment:
        changed = sorted(
            key
            for key in current_environment
            if current_environment[key] != environment.get(key)
        )
        raise RuntimeError(f"strict resume environment mismatch: {changed}")
    current_rtwm_commit = _git_commit(RTWM)
    if _git_identity(GAMMA) != identity["gamma_git"]:
        raise RuntimeError("strict resume Gamma source-state mismatch")

    if sha256_file(measured_path) != measured_sha:
        raise RuntimeError("strict resume check mutated the measured artifact")
    return {
        "status": "complete_noop",
        "measured_artifact": measured_path.relative_to(HERE).as_posix(),
        "measured_artifact_sha256": measured_sha,
        "recorded_identity_sha256": recorded_identity_sha,
        "recorded_source_hashes": identity["sources"],
        "current_source_hashes": current_source_hashes,
        "import_migration_sha256": sha256_file(IMPORT_MIGRATION),
        "recorded_rtwm_commit": identity["rtwm_git"]["commit"],
        "current_rtwm_commit": current_rtwm_commit,
        "rtwm_head_equality_required": False,
        "execution_sources_validated_by_hash": True,
        "unrelated_rtwm_worktree_diff_excluded": True,
        "measured_payload_mutated": False,
    }


def _validate_identity(path: Path) -> str:
    value = json.loads(path.read_text())
    identity = value["identity"]
    recorded = identity["identity_sha256"]
    computed = hashlib.sha256(
        canonical_bytes(_identity_payload(identity))
    ).hexdigest()
    if recorded != computed:
        raise RuntimeError(f"{path.name} identity self-hash mismatch")
    return recorded


def _validate_manifest(
    path: Path,
    *,
    entries_key: str,
    self_key: str,
    root: Path,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    payload = dict(manifest)
    recorded = payload.pop(self_key)
    computed = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if recorded != computed:
        raise RuntimeError(f"{path.name} self-hash mismatch")
    entries = manifest[entries_key]
    for entry in entries:
        artifact = root / entry["path"]
        if not artifact.is_file():
            raise RuntimeError(f"missing artifact: {entry['path']}")
        if (
            artifact.stat().st_size != entry["bytes"]
            or sha256_file(artifact) != entry["sha256"]
        ):
            raise RuntimeError(f"artifact hash mismatch: {entry['path']}")
    return {
        "path": path.relative_to(HERE).as_posix(),
        "file_sha256": sha256_file(path),
        "self_hash": recorded,
        "artifact_count": len(entries),
    }


def validate_manifests() -> dict[str, Any]:
    manifests = [
        _validate_manifest(
            HERE / "results" / "manifest.json",
            entries_key="artifacts",
            self_key="manifest_sha256",
            root=CODE,
        ),
        _validate_manifest(
            HERE / "results" / "pre_roll_b1_manifest.json",
            entries_key="artifacts",
            self_key="manifest_sha256",
            root=CODE,
        ),
        _validate_manifest(
            HERE / "manifests" / "intent_pre_roll_artifacts.json",
            entries_key="files",
            self_key="manifest_payload_sha256",
            root=HERE,
        ),
    ]
    summaries = {
        "pre_roll_b1_summary": _validate_identity(
            HERE / "results" / "pre_roll_b1_summary.json"
        ),
        "intent_pre_roll_summary": _validate_identity(
            HERE / "results" / "intent_pre_roll_summary.json"
        ),
    }
    return {
        "status": "valid",
        "manifests": manifests,
        "summary_identity_sha256": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "check",
        choices=("resume", "manifests", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args()
    output: dict[str, Any] = {}
    if args.check in {"resume", "all"}:
        output["strict_resume"] = strict_resume_check()
    if args.check in {"manifests", "all"}:
        output["artifacts"] = validate_manifests()
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
