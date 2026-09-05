from pathlib import Path

import cv2
import numpy as np

from flow_detector import (
    DetectorConfig,
    detect_onset,
    flow_signals,
    split_views,
)


HERE = Path(__file__).resolve().parent
CONFIG = DetectorConfig.from_json(HERE / "config" / "pilot.json")


def textured_frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((CONFIG.resize_height, CONFIG.resize_width), dtype=np.uint8)
    for x, y, value in zip(
        rng.integers(10, CONFIG.resize_width - 10, 350),
        rng.integers(10, CONFIG.resize_height - 10, 350),
        rng.integers(80, 256, 350),
    ):
        cv2.circle(frame, (int(x), int(y)), 2, int(value), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def translated(frame: np.ndarray, dx: float, dy: float = 0) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def scaled(frame: np.ndarray, scale: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        (CONFIG.radial_center_x, CONFIG.radial_center_y),
        0,
        scale,
    )
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def test_static_frames_have_near_zero_signals():
    frame = textured_frame()
    signals = flow_signals([frame, frame.copy()], CONFIG)
    assert abs(signals["horizontal"][0]) < 1e-3
    assert abs(signals["radial"][0]) < 1e-3
    assert signals["magnitude"][0] < 1e-3


def test_positive_horizontal_translation_has_positive_sign():
    frame = textured_frame()
    signals = flow_signals([frame, translated(frame, 2)], CONFIG)
    assert signals["horizontal"][0] > 0.5


def test_zoom_and_shrink_have_opposite_radial_signs():
    frame = textured_frame()
    expansion = flow_signals([frame, scaled(frame, 1.04)], CONFIG)["radial"][0]
    contraction = flow_signals([frame, scaled(frame, 0.96)], CONFIG)["radial"][0]
    assert expansion > 0.1
    assert contraction < -0.1


def test_crossing_index_reports_rendered_frame_endpoint():
    change = 24
    magnitude = np.ones(80, dtype=np.float64)
    magnitude[change:change + 3] = 0.1
    signals = {
        "magnitude": magnitude,
        "horizontal": np.zeros_like(magnitude),
        "radial": np.zeros_like(magnitude),
        "finite_fraction": np.ones_like(magnitude),
    }
    result = detect_onset(signals, "stop", change, CONFIG)
    assert result["onset_signal_index"] == change
    assert result["onset_frame"] == change + 1
    assert result["lag_from_change"] == 1


def test_split_views_preserves_left_right_order():
    left = np.full((20, 30, 3), 25, dtype=np.uint8)
    right = np.full((20, 30, 3), 225, dtype=np.uint8)
    split_left, split_right = split_views(np.concatenate([left, right], axis=1))
    assert float(split_left.mean()) == 25
    assert float(split_right.mean()) == 225
