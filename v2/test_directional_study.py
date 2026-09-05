from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from directional_study import (
    CANONICAL_KEYS,
    action_protocol_sha256,
    action_sequence,
    detector_config,
    intervention_specs,
    load_directional_protocol,
    persistent_effect,
    projection_features,
    radial_component,
    robust_global_motion,
    rollout_specs,
)
from flow_detector import (
    farneback_flow,
    first_sustained_strict_crossing,
    paired_flow_field_divergence,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = load_directional_protocol(HERE / "config" / "directional_d0.json")
CONFIG = detector_config(PROTOCOL)


def textured(seed: int = 20260820) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((CONFIG.resize_height, CONFIG.resize_width), dtype=np.uint8)
    for x, y, value in zip(
        rng.integers(8, CONFIG.resize_width - 8, 700),
        rng.integers(8, CONFIG.resize_height - 8, 700),
        rng.integers(60, 256, 700),
    ):
        cv2.circle(frame, (int(x), int(y)), 2, int(value), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def translate(frame: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def scale(frame: np.ndarray, factor: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        (CONFIG.radial_center_x, CONFIG.radial_center_y), 0, factor
    )
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def test_frozen_partition_and_exact_intervention_count():
    specs = intervention_specs(PROTOCOL)
    assert len(specs) == 24
    assert len({row["rollout_id"] for row in specs}) == 24
    assert {row["partition"] for row in specs if row["seed"] <= 202} == {"audit"}
    assert {row["partition"] for row in specs if row["seed"] >= 203} == {"held_out"}
    fresh = rollout_specs(PROTOCOL)
    assert len(fresh) == 52
    assert sum(row["stage"] == "calibration" for row in fresh) == 12
    assert sum(row["stage"] == "validation" for row in fresh) == 4
    assert sum(row["arm"] == "control" for row in fresh) == 12
    assert sum(row["arm"] in {"back", "yaw_positive"} for row in fresh) == 24


def test_canonical_action_order_and_true_yaw():
    assert len(CANONICAL_KEYS) == 23
    assert CANONICAL_KEYS.index("forward") == 11
    assert CANONICAL_KEYS.index("back") == 12
    change = int(PROTOCOL.data["model"]["change_frame"])
    back = action_sequence(PROTOCOL, "back")
    yaw = action_sequence(PROTOCOL, "yaw_positive")
    assert np.flatnonzero(back["keyboard"][0]).tolist() == [11]
    assert np.flatnonzero(back["keyboard"][change]).tolist() == [12]
    assert np.flatnonzero(yaw["keyboard"][change]).tolist() == [11]
    assert yaw["camera"][change] == [6.0, 0.0]
    assert yaw["keyboard"][change][13:15] == [0.0, 0.0]


def test_translation_and_opposite_equal_speed_flow():
    frame = textured()
    positive = [frame, translate(frame, 2)]
    negative = [frame, translate(frame, -2)]
    direct = paired_flow_field_divergence(positive, negative, CONFIG)[0]
    positive_flow = farneback_flow(*positive, CONFIG)
    negative_flow = farneback_flow(*negative, CONFIG)
    projection = projection_features(positive_flow, negative_flow, CONFIG)
    assert direct > 1.0
    assert projection["signed_projection"] < -1.0
    assert projection["cosine_projection"] < -0.5


def test_expansion_and_contraction_radial_signs():
    frame = textured()
    expansion = radial_component(farneback_flow(frame, scale(frame, 1.04), CONFIG), CONFIG)
    contraction = radial_component(farneback_flow(frame, scale(frame, 0.96), CONFIG), CONFIG)
    assert expansion > 0.1
    assert contraction < -0.1


def test_true_global_yaw_affine_horizontal_sign():
    flow = np.zeros((CONFIG.resize_height, CONFIG.resize_width, 2), dtype=np.float32)
    flow[..., 0] = -3.25
    result = robust_global_motion(flow, CONFIG)
    assert result["global_horizontal"] < -3.0
    assert result["residual_flow_l2"] < 1e-4
    assert result["inlier_fraction"] > 0.99


def test_residual_object_motion_is_separate_from_global_yaw():
    flow = np.zeros((CONFIG.resize_height, CONFIG.resize_width, 2), dtype=np.float32)
    flow[..., 0] = -3.0
    flow[55:95, 90:145, 0] = 4.0
    flow[55:95, 90:145, 1] = 1.5
    result = robust_global_motion(flow, CONFIG)
    assert result["global_horizontal"] < -2.8
    assert result["residual_flow_l2"] > 0.1


def test_endpoint_indexing_reports_rendered_destination_frame():
    signal = np.zeros(30)
    signal[8:11] = 2.0
    crossing = first_sustained_strict_crossing(signal, 1.0, "rise", 6, 20, 3)
    assert crossing == 8
    assert crossing + 1 == 9
    assert persistent_effect(signal[6:20], +1, 3) == 2.0


def test_audit_and_held_out_inputs_cannot_overlap():
    design = PROTOCOL.data["design"]
    assert set(design["audit_seeds"]).isdisjoint(design["held_out_seeds"])
    audit = {
        (row["scene"], row["seed"], row["transition"])
        for row in intervention_specs(PROTOCOL)
        if row["partition"] == "audit"
    }
    held_out = {
        (row["scene"], row["seed"], row["transition"])
        for row in intervention_specs(PROTOCOL)
        if row["partition"] == "held_out"
    }
    assert audit.isdisjoint(held_out)
    assert PROTOCOL.data["audit_gate"]["forbid_held_out_threshold_inputs"] is True


def test_fresh_grid_forbids_legacy_action_records():
    assert "control_reuse" not in PROTOCOL.data
    assert "legacy_forward_encoder_source" not in PROTOCOL.data["action_protocol"]
    assert all(
        row["rollout_id"] != "buildTower_normal__seed201__stop_d0"
        for row in rollout_specs(PROTOCOL)
    )
    assert len(action_protocol_sha256(PROTOCOL)) == 64
