from pathlib import Path

import cv2
import numpy as np

from calibrate_detector import fit_null_threshold, null_window_statistics, validate_protocol
from flow_detector import (
    DetectorConfig,
    PairedDetectorConfig,
    detect_calibrated_paired_divergence,
    first_sustained_strict_crossing,
    flow_signals,
    paired_flow_field_divergence,
)
from score_confirmatory import (
    seed_level_statistics,
    signal_window_statistics,
    support_aligned_search_range,
)


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "confirmatory.json"
DETECTOR = DetectorConfig.from_json(CONFIG_PATH)
PAIRED = PairedDetectorConfig.from_json(CONFIG_PATH)


def textured_frame(seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros(
        (DETECTOR.resize_height, DETECTOR.resize_width), dtype=np.uint8
    )
    for x, y, value in zip(
        rng.integers(10, DETECTOR.resize_width - 10, 500),
        rng.integers(10, DETECTOR.resize_height - 10, 500),
        rng.integers(80, 256, 500),
    ):
        cv2.circle(frame, (int(x), int(y)), 2, int(value), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def translated(frame: np.ndarray, dx: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def test_paired_config_is_read_from_dedicated_section():
    assert PAIRED.signal == "flow_field_l2"
    assert PAIRED.direction == "rise"
    assert PAIRED.persistence == 3
    assert PAIRED.threshold_statistic == "maximum"
    assert PAIRED.crossing_operator == "strict_greater_than"


def test_identical_frame_streams_have_zero_direct_divergence():
    frame = textured_frame()
    frames = [frame, translated(frame, 1), translated(frame, 2)]
    divergence = paired_flow_field_divergence(frames, frames, DETECTOR)
    assert divergence.shape == (2,)
    assert np.max(np.abs(divergence)) == 0


def test_direct_field_divergence_detects_opposite_equal_speed_motion():
    frame = textured_frame()
    positive = [frame, translated(frame, 2)]
    negative = [frame, translated(frame, -2)]
    direct = paired_flow_field_divergence(positive, negative, DETECTOR)[0]
    positive_magnitude = flow_signals(positive, DETECTOR)["magnitude"][0]
    negative_magnitude = flow_signals(negative, DETECTOR)["magnitude"][0]
    indirect = abs(positive_magnitude - negative_magnitude)
    assert direct > 1.0
    assert direct > 10 * indirect


def test_strict_crossing_does_not_count_equality():
    signal = np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
    crossing = first_sustained_strict_crossing(
        signal, 0.5, "rise", 0, len(signal), persistence=3
    )
    assert crossing == 3


def test_calibrated_detection_respects_admission_and_endpoint_indexing():
    signal = np.zeros(20)
    signal[3:6] = 2.0
    signal[8:11] = 2.0
    result = detect_calibrated_paired_divergence(
        signal,
        threshold=1.0,
        admission_frame=6,
        change_frame=4,
        config=PAIRED,
        search_end=15,
    )
    assert result["onset_signal_index"] == 8
    assert result["onset_frame"] == 9
    assert result["lag_from_change"] == 5
    assert result["offset_from_admission"] == 3
    assert result["comparison"] == "strict_greater_than"


def test_null_threshold_fit_uses_configured_maximum():
    threshold = fit_null_threshold(
        [np.array([0.1, 0.2]), np.array([0.3, 0.25])],
        PAIRED,
    )
    assert threshold == 0.3
    assert first_sustained_strict_crossing(
        np.array([threshold, threshold, threshold]),
        threshold,
        "rise",
        0,
        3,
        3,
    ) is None


def test_null_window_statistic_uses_persistent_minimum():
    signal = np.zeros(30)
    signal[10:13] = [0.2, 0.4, 0.3]
    rows = null_window_statistics(signal, [8], 10, persistence=3)
    assert rows == [
        {
            "anchor": 8,
            "search_end": 18,
            "maximum_sustained_min": 0.2,
        }
    ]


def test_confirmatory_seed_partition_is_frozen():
    import json

    protocol = json.loads(CONFIG_PATH.read_text())
    scenes, calibration, validation, test, arms = validate_protocol(protocol)
    assert scenes == ["buildTower_normal", "buildHouse_flat"]
    assert calibration == [101, 102, 103]
    assert validation == 104
    assert test == [201, 202, 203, 204, 205, 206]
    assert arms == ["null_control_a", "null_control_b"]


def test_signal_window_statistics_reports_effect_magnitude():
    result = signal_window_statistics(
        np.array([0.0, 0.5, 1.5, 2.0, np.nan]),
        start=1,
        end=5,
        threshold=0.25,
    )
    assert result == {
        "signal_window_count": 3,
        "signal_peak": 2.0,
        "signal_mean": 4.0 / 3.0,
        "signal_auc": 4.0,
        "signal_peak_above_threshold": 1.75,
    }


def test_raw_support_search_includes_flow_entering_first_affected_frame():
    effect_start = 105
    search_start, search_end = support_aligned_search_range(effect_start, 40)
    assert (search_start, search_end) == (104, 145)
    signal = np.zeros(160)
    signal[104:107] = 1.0
    result = detect_calibrated_paired_divergence(
        signal,
        threshold=0.0,
        admission_frame=search_start,
        change_frame=96,
        config=PAIRED,
        search_end=search_end,
    )
    assert result["onset_signal_index"] == 104
    assert result["onset_frame"] == effect_start
    assert result["lag_from_change"] == 9


def test_seed_statistics_cluster_scenes_within_seed():
    rows = []
    for delay, total_s, onset in (("d0", 260.0, 110.0), ("d1", 262.0, 122.0)):
        for seed in (201, 202):
            for scene in ("tower", "house"):
                rows.append(
                    {
                        "pair_valid": True,
                        "delay": delay,
                        "seed": seed,
                        "scene": scene,
                        "pair_status": "both",
                        "detected_count": 2,
                        "both_detected_onset_mean": onset,
                        "pair_signal_peak_mean": 1.0 + (delay == "d1"),
                        "intervention_total_s": total_s,
                    }
                )
    result = seed_level_statistics(rows, change_frame=96, n_frames=189)
    assert result["by_delay"]["d0"]["seed_count"] == 2
    assert result["by_delay"]["d0"]["scene_pair_count"] == 4
    assert result["by_delay"]["d0"]["metrics"]["causal_onset_lag_frames"]["mean"] == 14.0
    assert result["paired_delay_comparison"]["similar_throughput"]
    assert (
        result["paired_delay_comparison"]["d1_minus_d0_causal_onset_lag_frames"][
            "mean"
        ]
        == 12.0
    )
