"""Frozen protocol, action construction, and endpoints for the d0 directional study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from artifacts import ArtifactError, verify_rollout_artifacts
from flow_detector import DetectorConfig, farneback_flow, summarize_paired_flow_field
from protocol import canonical_sha256, file_sha256

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config" / "directional_d0.json"

CANONICAL_KEYS = (
    "inventory", "ESC",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "swapHands", "attack", "use", "pickItem", "drop",
)


@dataclass(frozen=True)
class DirectionalProtocol:
    path: Path
    data: dict[str, Any]
    file_sha256: str
    canonical_sha256: str

    @property
    def identity(self) -> dict[str, str]:
        return {
            "config_sha256": self.file_sha256,
            "protocol_sha256": self.canonical_sha256,
        }


def load_directional_protocol(path: str | Path = DEFAULT_CONFIG) -> DirectionalProtocol:
    target = Path(path).resolve()
    data = json.loads(target.read_text())
    if data.get("schema_version") != "rtwm-v2-directional-d0-1":
        raise ValueError("unsupported directional protocol")
    design = data["design"]
    if design["delay"] != "d0":
        raise ValueError("directional study is d0-only")
    if design["scenes"] != ["buildTower_normal", "buildHouse_flat"]:
        raise ValueError("directional scenes changed")
    if design["audit_seeds"] != [201, 202] or design["held_out_seeds"] != [203, 204, 205, 206]:
        raise ValueError("directional audit/held-out partition changed")
    if set(design["transitions"]) != {"back", "yaw_positive"}:
        raise ValueError("directional transitions changed")
    expected_grid = design["fresh_grid"]["expected_rollouts"]
    if expected_grid != {
        "calibration": 12,
        "validation": 4,
        "test_controls": 12,
        "directional_interventions": 24,
        "total": 52,
    }:
        raise ValueError("fresh canonical rollout grid changed")
    yaw = design["transitions"]["yaw_positive"]["post"]
    if yaw["keys"] != ["forward"] or float(yaw["camera_yaw_degrees"]) == 0:
        raise ValueError("yaw must be nonzero camera conditioning while moving forward")
    if any(key in yaw["keys"] for key in ("left", "right")):
        raise ValueError("keyboard strafe is not a camera turn")
    if tuple(data["action_protocol"]["ordering"]) != CANONICAL_KEYS:
        raise ValueError("action ordering is not Solaris canonical ordering")
    if int(data["model"]["admission_latent"]) != 27:
        raise ValueError("d0 admission latent must remain 27")
    if int(data["endpoints"]["persistence"]) < 1:
        raise ValueError("persistence must be positive")
    if not data.get("immutable"):
        raise ValueError("directional protocol must be marked immutable")
    return DirectionalProtocol(target, data, file_sha256(target), canonical_sha256(data))


def action_protocol_sha256(protocol: DirectionalProtocol) -> str:
    return canonical_sha256(protocol.data["action_protocol"])


def action_sequence(
    protocol: DirectionalProtocol,
    transition: str,
) -> dict[str, list[list[float]]]:
    """Build exact canonical keyboard and [yaw, pitch] camera conditioning."""
    model = protocol.data["model"]
    n_frames = int(model["n_frames"])
    change = int(model["change_frame"])
    design = protocol.data["design"]
    spec = (
        design["canonical_forward"]
        if transition == "canonical_forward"
        else design["transitions"][transition]
    )
    keyboard: list[list[float]] = []
    camera: list[list[float]] = []
    key_index = {name: index for index, name in enumerate(CANONICAL_KEYS)}
    for frame in range(n_frames):
        state = spec["pre"] if frame < change else spec["post"]
        row = [0.0] * len(CANONICAL_KEYS)
        for key in state["keys"]:
            if key not in key_index:
                raise ValueError(f"unknown canonical key {key!r}")
            row[key_index[key]] = 1.0
        keyboard.append(row)
        camera.append([
            float(state["camera_yaw_degrees"]),
            float(state["camera_pitch_degrees"]),
        ])
    return {"keyboard": keyboard, "camera": camera}


def intervention_specs(protocol: DirectionalProtocol) -> list[dict[str, Any]]:
    specs = []
    design = protocol.data["design"]
    partitions = (("audit", design["audit_seeds"]), ("held_out", design["held_out_seeds"]))
    for partition, seeds in partitions:
        for scene in design["scenes"]:
            for seed in seeds:
                for transition in design["transitions"]:
                    specs.append({
                        "partition": partition,
                        "scene": scene,
                        "seed": int(seed),
                        "transition": transition,
                        "arm": transition,
                        "rollout_id": f"{scene}__seed{seed}__{transition}_d0",
                    })
    if len(specs) != 24 or len({row["rollout_id"] for row in specs}) != 24:
        raise AssertionError("directional design must contain exactly 24 interventions")
    return specs


def rollout_specs(protocol: DirectionalProtocol) -> list[dict[str, Any]]:
    """Return the frozen 52-rollout order: nulls, controls, interventions."""
    design = protocol.data["design"]
    grid = design["fresh_grid"]
    specs: list[dict[str, Any]] = []
    for stage in ("calibration", "validation"):
        for scene in design["scenes"]:
            for seed in grid[stage]["seeds"]:
                for arm in grid[stage]["arms"]:
                    specs.append({
                        "stage": stage,
                        "partition": stage,
                        "scene": scene,
                        "seed": int(seed),
                        "arm": arm,
                        "transition": "canonical_forward",
                        "rollout_id": f"{scene}__seed{seed}__{arm}",
                    })
    for scene in design["scenes"]:
        for seed in grid["test"]["seeds"]:
            specs.append({
                "stage": "test",
                "partition": "audit" if seed in design["audit_seeds"] else "held_out",
                "scene": scene,
                "seed": int(seed),
                "arm": "control",
                "transition": "canonical_forward",
                "rollout_id": f"{scene}__seed{seed}__control",
            })
    for row in intervention_specs(protocol):
        specs.append({**row, "stage": "test"})
    counts = {
        "calibration": sum(row["stage"] == "calibration" for row in specs),
        "validation": sum(row["stage"] == "validation" for row in specs),
        "test_controls": sum(row["arm"] == "control" for row in specs),
        "directional_interventions": sum(row["arm"] in {"back", "yaw_positive"} for row in specs),
        "total": len(specs),
    }
    if counts != grid["expected_rollouts"]:
        raise AssertionError(f"fresh rollout grid mismatch: {counts}")
    return specs


def resolve_protocol_path(protocol: DirectionalProtocol, relative: str) -> Path:
    return HERE / relative


def validate_control_reuse(
    protocol: DirectionalProtocol,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Require exact provenance and all 12 forward control bundles.

    Controls generated by an action encoder without a recorded canonical action
    protocol hash are rejected: a command label such as ``forward`` cannot prove
    which of the 23 conditioning coordinates was activated.
    """
    cfg = protocol.data["control_reuse"]
    manifest_path = resolve_protocol_path(protocol, cfg["manifest"])
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    checks["manifest_hash"] = (
        manifest_path.is_file()
        and file_sha256(manifest_path) == cfg["manifest_sha256"]
    )
    if not checks["manifest_hash"]:
        blockers.append("control_manifest_hash_mismatch")
        return {"allowed": False, "checks": checks, "blockers": blockers, "controls": []}
    manifest = json.loads(manifest_path.read_text())
    expected_model = protocol.data["model"]
    expected = {
        "config_sha256": cfg["protocol_config_sha256"],
        "protocol_sha256": cfg["protocol_sha256"],
        "checkpoint": expected_model["checkpoint"],
        "n_frames": expected_model["n_frames"],
        "fps": expected_model["fps"],
        "latent_stride_pixels": expected_model["latent_stride_pixels"],
        "nfpb": expected_model["latent_frames_per_block"],
    }
    checks["manifest_protocol_model"] = all(manifest.get(key) == value for key, value in expected.items())
    if not checks["manifest_protocol_model"]:
        blockers.append("control_protocol_or_model_mismatch")
    source = manifest.get("gamma_source", {})
    checks["source_identity"] = (
        source.get("commit") == expected_model["source_commit"]
        and source.get("diff_sha256") == expected_model["source_diff_sha256"]
    )
    if not checks["source_identity"]:
        blockers.append("control_source_identity_mismatch")
    artifacts = manifest.get("model_artifacts", {})
    checks["model_artifacts"] = (
        artifacts.get("checkpoint_sha256") == expected_model["checkpoint_sha256"]
        and artifacts.get("tokenizer_sha256") == expected_model["tokenizer_sha256"]
    )
    if not checks["model_artifacts"]:
        blockers.append("control_model_artifact_mismatch")

    expected_action_hash = action_protocol_sha256(protocol)
    recorded_action_hash = manifest.get("action_protocol_sha256")
    checks["recorded_action_protocol"] = recorded_action_hash == expected_action_hash
    if not checks["recorded_action_protocol"]:
        blockers.append("control_action_protocol_unrecorded_or_incompatible")

    wanted = {
        (scene, int(seed))
        for scene in protocol.data["design"]["scenes"]
        for seed in protocol.data["design"]["audit_seeds"] + protocol.data["design"]["held_out_seeds"]
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in manifest.get("rollouts", []):
        key = (str(record.get("scene")), int(record.get("seed", -1)))
        if key in wanted and record.get("arm") == "control" and record.get("status") == "complete":
            if key in indexed:
                blockers.append(f"duplicate_control:{key[0]}:{key[1]}")
            indexed[key] = record
    checks["control_coverage"] = set(indexed) == wanted
    if not checks["control_coverage"]:
        blockers.append("incomplete_control_coverage")

    scene_hashes = manifest.get("scene_hashes", {})
    checks["scene_hashes"] = all(
        record.get("scene_sha256") == scene_hashes.get(scene)
        for (scene, _), record in indexed.items()
    )
    if not checks["scene_hashes"]:
        blockers.append("control_scene_hash_mismatch")
    checks["commands_and_seeds"] = all(
        record.get("pre_command") == "forward"
        and record.get("post_command") == "forward"
        and int(record["seed"]) == seed
        for (scene, seed), record in indexed.items()
    )
    if not checks["commands_and_seeds"]:
        blockers.append("control_command_or_seed_mismatch")

    artifact_errors = []
    if verify_artifacts and not blockers:
        for key, record in sorted(indexed.items()):
            try:
                verify_rollout_artifacts(
                    manifest_path.parent,
                    record,
                    required=tuple(cfg["required_artifacts"]),
                )
            except ArtifactError as error:
                artifact_errors.append(f"{key[0]}:{key[1]}:{error}")
    checks["artifact_hashes"] = not artifact_errors if verify_artifacts else False
    blockers.extend(f"control_artifact:{error}" for error in artifact_errors)
    return {
        "allowed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "manifest_path": str(manifest_path.relative_to(HERE)),
        "manifest_sha256": file_sha256(manifest_path),
        "expected_action_protocol_sha256": expected_action_hash,
        "recorded_action_protocol_sha256": recorded_action_hash,
        "controls": [
            {"scene": scene, "seed": seed, "rollout_id": record["rollout_id"]}
            for (scene, seed), record in sorted(indexed.items())
        ],
    }


def detector_config(protocol: DirectionalProtocol) -> DetectorConfig:
    data = protocol.data["detector"]
    roi, center = data["roi"], data["radial_center"]
    return DetectorConfig(
        resize_width=int(data["resize_width"]),
        resize_height=int(data["resize_height"]),
        roi_x0=int(roi["x0"]), roi_x1=int(roi["x1"]),
        roi_y0=int(roi["y0"]), roi_y1=int(roi["y1"]),
        trim_fraction=float(data["trim_fraction"]),
        radial_center_x=float(center[0]), radial_center_y=float(center[1]),
        radial_min_radius=float(data["radial_min_radius"]),
        farneback=dict(data["farneback"]),
        baseline_before=24, baseline_gap=2, search_horizon=40, persistence=3,
        mad_multiplier=4.0, relative_delta=0.15, absolute_delta=0.03,
        pair_disagreement_flag_frames=8, transitions={},
    )


def trimmed(values: np.ndarray, fraction: float) -> float:
    finite = np.sort(np.asarray(values, dtype=float).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    count = int(math.floor(fraction * finite.size))
    if count and 2 * count < finite.size:
        finite = finite[count:-count]
    return float(np.mean(finite))


def radial_component(flow: np.ndarray, config: DetectorConfig) -> float:
    y0, y1, x0, x1 = config.roi_y0, config.roi_y1, config.roi_x0, config.roi_x1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - config.radial_center_x, yy - config.radial_center_y
    radius = np.hypot(dx, dy)
    roi = flow[y0:y1, x0:x1]
    mask = radius >= config.radial_min_radius
    values = (roi[..., 0] * dx + roi[..., 1] * dy) / np.maximum(radius, 1e-12)
    return trimmed(values[mask], config.trim_fraction)


def projection_features(
    control_flow: np.ndarray,
    intervention_flow: np.ndarray,
    config: DetectorConfig,
) -> dict[str, float]:
    y0, y1, x0, x1 = config.roi_y0, config.roi_y1, config.roi_x0, config.roi_x1
    control = control_flow[y0:y1, x0:x1]
    intervention = intervention_flow[y0:y1, x0:x1]
    norm = np.linalg.norm(control, axis=-1)
    valid = np.isfinite(control).all(axis=-1) & np.isfinite(intervention).all(axis=-1) & (norm > 1e-4)
    unit = control / np.maximum(norm[..., None], 1e-12)
    signed = np.sum((intervention - control) * unit, axis=-1)
    cosine = np.sum(intervention * unit, axis=-1) / np.maximum(
        np.linalg.norm(intervention, axis=-1), 1e-12
    )
    return {
        "signed_projection": trimmed(signed[valid], config.trim_fraction),
        "cosine_projection": trimmed(cosine[valid], config.trim_fraction),
        "radial_reversal": radial_component(intervention_flow, config),
    }


def robust_global_motion(flow: np.ndarray, config: DetectorConfig) -> dict[str, float]:
    """Fit a robust affine global field and separate residual object motion."""
    y0, y1, x0, x1 = config.roi_y0, config.roi_y1, config.roi_x0, config.roi_x1
    yy, xx = np.mgrid[y0:y1:4, x0:x1:4]
    source = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    sampled = flow[y0:y1:4, x0:x1:4].reshape(-1, 2).astype(np.float32)
    finite = np.isfinite(sampled).all(axis=1)
    source, sampled = source[finite], sampled[finite]
    if len(source) < 12:
        return {"global_horizontal": float("nan"), "residual_flow_l2": float("nan"), "inlier_fraction": 0.0}
    destination = source + sampled
    matrix, inliers = cv2.estimateAffine2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=1.5,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None:
        return {"global_horizontal": float("nan"), "residual_flow_l2": float("nan"), "inlier_fraction": 0.0}
    predicted = source @ matrix[:, :2].T + matrix[:, 2] - source
    residual = sampled - predicted
    return {
        "global_horizontal": trimmed(predicted[:, 0], config.trim_fraction),
        # Keep localized object motion in this diagnostic; trimming it as an
        # outlier would incorrectly turn a global-motion residual into zero.
        "residual_flow_l2": float(np.mean(np.linalg.norm(residual, axis=1))),
        "inlier_fraction": float(np.mean(inliers)) if inliers is not None else 0.0,
    }


def flow_endpoint_sample(
    control_previous: np.ndarray,
    control_current: np.ndarray,
    intervention_previous: np.ndarray,
    intervention_current: np.ndarray,
    config: DetectorConfig,
    transition: str,
) -> dict[str, float]:
    control = farneback_flow(control_previous, control_current, config)
    intervention = farneback_flow(intervention_previous, intervention_current, config)
    primary = summarize_paired_flow_field(control, intervention, config)["flow_field_l2"]
    result = {"paired_flow_field_l2": float(primary)}
    if transition == "back":
        result.update(projection_features(control, intervention, config))
    elif transition == "yaw_positive":
        result.update(robust_global_motion(intervention, config))
    else:
        raise ValueError(f"unsupported transition {transition}")
    return result


def persistent_effect(values: Iterable[float], sign: int, persistence: int) -> float:
    array = sign * np.asarray(list(values), dtype=float)
    finite_windows = [
        float(np.min(array[index:index + persistence]))
        for index in range(max(0, len(array) - persistence + 1))
        if np.isfinite(array[index:index + persistence]).all()
    ]
    return max(finite_windows) if finite_windows else float("nan")
