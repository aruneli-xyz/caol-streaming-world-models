"""Immutable protocol helpers for fresh Solaris-canonical STOP validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from artifacts import verify_rollout_artifacts
from directional_study import CANONICAL_KEYS
from protocol import canonical_sha256, file_sha256

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StopProtocol:
    path: Path
    data: dict[str, Any]
    file_sha256: str
    canonical_sha256: str

    @property
    def identity(self) -> dict[str, str]:
        return {"config_sha256": self.file_sha256, "protocol_sha256": self.canonical_sha256}


def load_stop_protocol(path: str | Path = HERE / "config" / "canonical_stop.json") -> StopProtocol:
    target = Path(path).resolve()
    data = json.loads(target.read_text())
    if data.get("schema_version") != "rtwm-v2-canonical-stop-1" or not data.get("immutable"):
        raise ValueError("unsupported or mutable canonical STOP protocol")
    design = data["design"]
    if design["scenes"] != ["buildTower_normal", "buildHouse_flat"]:
        raise ValueError("STOP scenes changed")
    if design["seeds"] != [201, 202, 203, 204, 205, 206]:
        raise ValueError("STOP seeds changed")
    if design["delays"] != [
        {"name": "d0", "admission_latent": 27},
        {"name": "d1", "admission_latent": 30},
    ]:
        raise ValueError("STOP delays changed")
    if tuple(data["action_protocol"]["ordering"]) != CANONICAL_KEYS:
        raise ValueError("noncanonical keyboard ordering")
    if design["pre"]["keys"] != ["forward"] or design["post"]["keys"] != []:
        raise ValueError("STOP must be canonical forward to all-zero stay")
    return StopProtocol(target, data, file_sha256(target), canonical_sha256(data))


def action_sequence(protocol: StopProtocol, *, stop: bool) -> dict[str, np.ndarray]:
    n_frames = int(protocol.data["model"]["n_frames"])
    change = int(protocol.data["model"]["change_frame"])
    keyboard = np.zeros((1, n_frames, 23), dtype=np.float32)
    keyboard[:, :, CANONICAL_KEYS.index("forward")] = 1.0
    if stop:
        keyboard[:, change:, :] = 0.0
    camera = np.zeros((1, n_frames, 2), dtype=np.float32)
    return {"keyboard": keyboard, "camera": camera}


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stop_specs(protocol: StopProtocol) -> list[dict[str, Any]]:
    rows = []
    for scene in protocol.data["design"]["scenes"]:
        for seed in protocol.data["design"]["seeds"]:
            for delay in protocol.data["design"]["delays"]:
                rows.append({
                    "scene": scene,
                    "seed": int(seed),
                    "delay": delay["name"],
                    "admission_latent": int(delay["admission_latent"]),
                    "arm": f"stop_{delay['name']}",
                    "rollout_id": f"{scene}__seed{seed}__stop_{delay['name']}",
                })
    if len(rows) != 24 or len({row["rollout_id"] for row in rows}) != 24:
        raise AssertionError("canonical STOP grid must contain 24 unique interventions")
    return rows


def validate_canonical_evidence(
    protocol: StopProtocol, *, verify_artifacts: bool
) -> dict[str, Any]:
    evidence = protocol.data["canonical_evidence"]
    checks = {}
    for name in (
        "directional_config", "source_manifest", "null_gate",
        "source_validation", "decoder_rf_manifest",
    ):
        path = HERE / evidence[name]
        checks[f"{name}_hash"] = path.is_file() and file_sha256(path) == evidence[f"{name}_sha256"]
    manifest_path = HERE / evidence["source_manifest"]
    manifest = json.loads(manifest_path.read_text())
    expected_action = protocol.data["action_protocol"]["canonical_action_protocol_sha256"]
    checks["source_manifest_identity"] = (
        manifest.get("protocol_sha256") == evidence["source_protocol_sha256"]
        and manifest.get("action_protocol_sha256") == expected_action
        and manifest.get("planned_rollouts") == 52
    )
    null_gate = json.loads((HERE / evidence["null_gate"]).read_text())
    checks["null_gate_compatible"] = (
        null_gate.get("passed") is True
        and null_gate.get("action_protocol_sha256") == expected_action
        and null_gate.get("manifest_sha256") == evidence["source_manifest_sha256"]
    )
    controls = {
        (row["scene"], int(row["seed"])): row
        for row in manifest["rollouts"]
        if row.get("stage") == "test" and row.get("arm") == "control"
        and row.get("status") == "complete"
    }
    wanted = {
        (scene, seed)
        for scene in protocol.data["design"]["scenes"]
        for seed in protocol.data["design"]["seeds"]
    }
    checks["exact_12_controls"] = set(controls) == wanted
    checks["control_action_identity"] = all(
        row.get("action_protocol_sha256") == expected_action
        and row.get("action_tensor_hashes", {}).get("generation_keyboard_sha256")
        == array_sha256(action_sequence(protocol, stop=False)["keyboard"])
        and row.get("action_tensor_hashes", {}).get("generation_camera_sha256")
        == array_sha256(action_sequence(protocol, stop=False)["camera"])
        for row in controls.values()
    )
    artifact_errors = []
    if verify_artifacts and all(checks.values()):
        for key, record in sorted(controls.items()):
            try:
                verify_rollout_artifacts(
                    HERE / evidence["source_root"], record,
                    required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
                )
            except Exception as error:
                artifact_errors.append(f"{key}:{error}")
    checks["control_artifacts"] = not artifact_errors if verify_artifacts else True
    return {
        "allowed": all(checks.values()),
        "checks": checks,
        "artifact_errors": artifact_errors,
        "controls": controls,
        "manifest": manifest,
    }
