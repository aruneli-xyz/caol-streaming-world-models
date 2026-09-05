"""Transition-aware optical-flow detectors for RTWM v2.

The primary detector uses only pre-change samples to define its threshold.
Signal index ``t`` always denotes flow from rendered frame ``t`` to
rendered frame ``t + 1``; reported onset frames are therefore ``t + 1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    resize_width: int
    resize_height: int
    roi_x0: int
    roi_x1: int
    roi_y0: int
    roi_y1: int
    trim_fraction: float
    radial_center_x: float
    radial_center_y: float
    radial_min_radius: float
    farneback: dict[str, Any]
    baseline_before: int
    baseline_gap: int
    search_horizon: int
    persistence: int
    mad_multiplier: float
    relative_delta: float
    absolute_delta: float
    pair_disagreement_flag_frames: int
    transitions: dict[str, dict[str, Any]]

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectorConfig":
        data = json.loads(Path(path).read_text())["detector"]
        roi = data["roi"]
        center = data["radial_center"]
        return cls(
            resize_width=int(data["resize_width"]),
            resize_height=int(data["resize_height"]),
            roi_x0=int(roi["x0"]),
            roi_x1=int(roi["x1"]),
            roi_y0=int(roi["y0"]),
            roi_y1=int(roi["y1"]),
            trim_fraction=float(data["trim_fraction"]),
            radial_center_x=float(center[0]),
            radial_center_y=float(center[1]),
            radial_min_radius=float(data["radial_min_radius"]),
            farneback=dict(data["farneback"]),
            baseline_before=int(data["baseline_before"]),
            baseline_gap=int(data["baseline_gap"]),
            search_horizon=int(data["search_horizon"]),
            persistence=int(data["persistence"]),
            mad_multiplier=float(data["mad_multiplier"]),
            relative_delta=float(data["relative_delta"]),
            absolute_delta=float(data["absolute_delta"]),
            pair_disagreement_flag_frames=int(data["pair_disagreement_flag_frames"]),
            transitions=dict(data["transitions"]),
        )


@dataclass(frozen=True)
class PairedDetectorConfig:
    """Configuration for the separately calibrated paired detector."""

    signal: str
    direction: str
    search_horizon_after_admission: int
    persistence: int
    threshold_statistic: str
    threshold_quantile: float
    crossing_operator: str

    @classmethod
    def from_json(cls, path: str | Path) -> "PairedDetectorConfig":
        data = json.loads(Path(path).read_text())["paired_detector"]
        fit = data.get("threshold_fit", {})
        config = cls(
            signal=str(data["signal"]),
            direction=str(data["direction"]),
            search_horizon_after_admission=int(
                data["search_horizon_after_admission"]
            ),
            persistence=int(data["persistence"]),
            threshold_statistic=str(fit.get("statistic", "maximum")),
            threshold_quantile=float(fit.get("quantile", 1.0)),
            crossing_operator=str(
                data.get("crossing_operator", "strict_greater_than")
            ),
        )
        if not config.signal:
            raise ValueError("paired signal must be nonempty")
        if config.direction != "rise":
            raise ValueError("direct paired divergence direction must be rise")
        if config.search_horizon_after_admission <= 0:
            raise ValueError("paired search horizon must be positive")
        if config.persistence <= 0:
            raise ValueError("paired persistence must be positive")
        if config.threshold_statistic not in {"maximum", "quantile"}:
            raise ValueError(
                f"unsupported threshold statistic: {config.threshold_statistic}"
            )
        if not 0 < config.threshold_quantile <= 1:
            raise ValueError("paired threshold quantile must be in (0, 1]")
        if config.crossing_operator != "strict_greater_than":
            raise ValueError("paired crossing operator must be strict_greater_than")
        return config


def trimmed_mean(values: np.ndarray | Iterable[float], fraction: float) -> float:
    """Finite-only symmetric trimmed mean."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    array.sort()
    trim = int(np.floor(fraction * array.size))
    if trim and 2 * trim < array.size:
        array = array[trim:-trim]
    return float(array.mean())


def split_views(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side BGR frame into equal player views."""
    if frame.ndim != 3 or frame.shape[1] % 2:
        raise ValueError(f"expected even-width BGR frame, got {frame.shape}")
    width = frame.shape[1] // 2
    return frame[:, :width], frame[:, width:]


def preprocess_view(view: np.ndarray, config: DetectorConfig) -> np.ndarray:
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    return cv2.resize(
        gray,
        (config.resize_width, config.resize_height),
        interpolation=cv2.INTER_AREA,
    )


def read_side_by_side_video(
    path: str | Path,
    config: DetectorConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Read and preprocess both views from one side-by-side MP4."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    views: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    source_width = source_height = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_height, source_width = frame.shape[:2]
        left, right = split_views(frame)
        views[0].append(preprocess_view(left, config))
        views[1].append(preprocess_view(right, config))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    metadata = {
        "path": str(path),
        "frame_count": len(views[0]),
        "fps": fps,
        "source_width": source_width,
        "source_height": source_height,
    }
    return views[0], views[1], metadata


def farneback_flow(
    previous: np.ndarray,
    current: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    p = config.farneback
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        float(p["pyr_scale"]),
        int(p["levels"]),
        int(p["winsize"]),
        int(p["iterations"]),
        int(p["poly_n"]),
        float(p["poly_sigma"]),
        int(p["flags"]),
    )


def summarize_flow(flow: np.ndarray, config: DetectorConfig) -> dict[str, float]:
    """Compute robust magnitude, horizontal, and radial summaries."""
    y0, y1 = config.roi_y0, config.roi_y1
    x0, x1 = config.roi_x0, config.roi_x1
    roi = flow[y0:y1, x0:x1]
    horizontal = roi[..., 0]
    vertical = roi[..., 1]
    magnitude = np.hypot(horizontal, vertical)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - config.radial_center_x
    dy = yy - config.radial_center_y
    radius = np.hypot(dx, dy)
    radial_mask = radius >= config.radial_min_radius
    radial = np.full_like(radius, np.nan, dtype=np.float64)
    radial[radial_mask] = (
        horizontal[radial_mask] * dx[radial_mask]
        + vertical[radial_mask] * dy[radial_mask]
    ) / radius[radial_mask]

    return {
        "magnitude": trimmed_mean(magnitude, config.trim_fraction),
        "horizontal": trimmed_mean(horizontal, config.trim_fraction),
        "radial": trimmed_mean(radial, config.trim_fraction),
        "finite_fraction": float(np.isfinite(roi).all(axis=-1).mean()),
    }


def flow_signals(
    frames: list[np.ndarray],
    config: DetectorConfig,
) -> dict[str, np.ndarray]:
    """Return one scalar sample per inter-frame flow interval."""
    samples: dict[str, list[float]] = {
        "magnitude": [],
        "horizontal": [],
        "radial": [],
        "finite_fraction": [],
    }
    for previous, current in zip(frames, frames[1:]):
        summary = summarize_flow(farneback_flow(previous, current, config), config)
        for key in samples:
            samples[key].append(summary[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples.items()}


def summarize_paired_flow_field(
    first_flow: np.ndarray,
    second_flow: np.ndarray,
    config: DetectorConfig,
) -> dict[str, float]:
    """Summarize direct vector-field disagreement inside the detector ROI."""
    if (
        first_flow.shape != second_flow.shape
        or first_flow.ndim != 3
        or first_flow.shape[-1] != 2
    ):
        raise ValueError(
            "paired flow fields must have identical (height, width, 2) shapes"
        )
    y0, y1 = config.roi_y0, config.roi_y1
    x0, x1 = config.roi_x0, config.roi_x1
    difference = first_flow[y0:y1, x0:x1] - second_flow[y0:y1, x0:x1]
    finite = np.isfinite(difference).all(axis=-1)
    divergence = np.linalg.norm(difference, axis=-1)
    return {
        "flow_field_l2": trimmed_mean(divergence, config.trim_fraction),
        "finite_fraction": float(finite.mean()) if finite.size else 0.0,
    }


def paired_flow_field_signals(
    first_frames: list[np.ndarray],
    second_frames: list[np.ndarray],
    config: DetectorConfig,
) -> dict[str, np.ndarray]:
    """Compute direct paired flow-field divergence for aligned frame streams."""
    if len(first_frames) != len(second_frames):
        raise ValueError(
            "paired frame streams must have the same length "
            f"({len(first_frames)} != {len(second_frames)})"
        )
    samples: dict[str, list[float]] = {
        "flow_field_l2": [],
        "finite_fraction": [],
    }
    for first_previous, first_current, second_previous, second_current in zip(
        first_frames,
        first_frames[1:],
        second_frames,
        second_frames[1:],
    ):
        first_flow = farneback_flow(first_previous, first_current, config)
        second_flow = farneback_flow(second_previous, second_current, config)
        summary = summarize_paired_flow_field(first_flow, second_flow, config)
        for key in samples:
            samples[key].append(summary[key])
    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in samples.items()
    }


def paired_flow_field_divergence(
    first_frames: list[np.ndarray],
    second_frames: list[np.ndarray],
    config: DetectorConfig,
) -> np.ndarray:
    """Return the direct L2 flow-field divergence signal."""
    return paired_flow_field_signals(first_frames, second_frames, config)[
        "flow_field_l2"
    ]


def robust_baseline(
    signal: np.ndarray,
    change_frame: int,
    config: DetectorConfig,
) -> dict[str, Any]:
    start = max(0, change_frame - config.baseline_before)
    end = max(start, change_frame - config.baseline_gap)
    baseline_values = np.asarray(signal[start:end], dtype=np.float64)
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size < 4:
        return {
            "status": "insufficient_baseline",
            "range": [start, end],
            "baseline": None,
            "mad_sigma": None,
            "delta": None,
        }
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline)))
    sigma = 1.4826 * mad
    delta = max(
        config.mad_multiplier * sigma,
        config.relative_delta * abs(baseline),
        config.absolute_delta,
    )
    return {
        "status": "ok",
        "range": [start, end],
        "baseline": baseline,
        "mad_sigma": sigma,
        "delta": float(delta),
    }


def transition_target(
    transition: str,
    baseline: float,
    delta: float,
    config: DetectorConfig,
) -> dict[str, Any]:
    if transition not in config.transitions:
        raise KeyError(f"unknown transition: {transition}")
    spec = config.transitions[transition]
    direction = str(spec["direction"])
    signal_name = str(spec["signal"])
    qc_flags: list[str] = []

    if transition == "stop" and baseline <= delta:
        return {
            "status": "indeterminate_baseline",
            "signal_name": signal_name,
            "direction": direction,
            "target": None,
            "qc_flags": ["stop_baseline_too_small"],
        }
    if transition == "reverse" and baseline <= 0:
        qc_flags.append("reverse_baseline_nonpositive")

    if direction == "rise":
        target = baseline + delta
        if "absolute_target" in spec:
            target = max(target, float(spec["absolute_target"]))
    elif direction == "fall":
        target = baseline - delta
        if "absolute_target" in spec:
            target = min(target, float(spec["absolute_target"]))
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return {
        "status": "ok",
        "signal_name": signal_name,
        "direction": direction,
        "target": float(target),
        "qc_flags": qc_flags,
    }


def first_sustained_crossing(
    signal: np.ndarray,
    target: float,
    direction: str,
    search_start: int,
    search_end: int,
    persistence: int,
) -> int | None:
    """Return the first crossing signal index; ``search_end`` is exclusive."""
    upper = min(search_end, len(signal) - persistence + 1)
    for index in range(max(0, search_start), max(0, upper)):
        window = signal[index:index + persistence]
        if window.size != persistence or not np.isfinite(window).all():
            continue
        crossed = bool(np.all(window >= target)) if direction == "rise" else bool(np.all(window <= target))
        if crossed:
            return index
    return None


def first_sustained_strict_crossing(
    signal: np.ndarray,
    target: float,
    direction: str,
    search_start: int,
    search_end: int,
    persistence: int,
) -> int | None:
    """Return the first strict crossing; equality never counts as a crossing."""
    if direction not in {"rise", "fall"}:
        raise ValueError(f"unsupported direction: {direction}")
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    upper = min(search_end, len(signal) - persistence + 1)
    for index in range(max(0, search_start), max(0, upper)):
        window = signal[index:index + persistence]
        if window.size != persistence or not np.isfinite(window).all():
            continue
        crossed = (
            bool(np.all(window > target))
            if direction == "rise"
            else bool(np.all(window < target))
        )
        if crossed:
            return index
    return None


def detect_calibrated_paired_divergence(
    divergence: np.ndarray,
    threshold: float,
    admission_frame: int,
    config: PairedDetectorConfig,
    *,
    change_frame: int | None = None,
    search_end: int | None = None,
) -> dict[str, Any]:
    """Apply a locked threshold to a direct paired flow-field signal."""
    signal = np.asarray(divergence, dtype=np.float64).reshape(-1)
    if not np.isfinite(threshold):
        raise ValueError("paired threshold must be finite")
    end = (
        admission_frame + config.search_horizon_after_admission
        if search_end is None
        else search_end
    )
    result: dict[str, Any] = {
        "signal_name": config.signal,
        "direction": config.direction,
        "threshold": float(threshold),
        "comparison": config.crossing_operator,
        "search_range": [int(admission_frame), int(end)],
        "onset_signal_index": None,
        "onset_frame": None,
        "lag_from_change": None,
        "offset_from_admission": None,
        "status": "undetected",
        "qc_flags": [],
    }
    onset_index = first_sustained_strict_crossing(
        signal,
        float(threshold),
        config.direction,
        admission_frame,
        end,
        config.persistence,
    )
    if onset_index is None:
        return result
    onset_frame = onset_index + 1
    result.update(
        status="detected",
        onset_signal_index=int(onset_index),
        onset_frame=int(onset_frame),
        lag_from_change=(
            int(onset_frame - change_frame) if change_frame is not None else None
        ),
        offset_from_admission=int(onset_frame - admission_frame),
    )
    return result


def detect_onset(
    signals: dict[str, np.ndarray],
    transition: str,
    change_frame: int,
    config: DetectorConfig,
    *,
    search_start: int | None = None,
    search_end: int | None = None,
) -> dict[str, Any]:
    """Detect one transition-aware onset from pre-change calibration only."""
    spec = config.transitions[transition]
    signal_name = str(spec["signal"])
    signal = signals[signal_name]
    baseline = robust_baseline(signal, change_frame, config)
    result: dict[str, Any] = {
        "transition": transition,
        "signal_name": signal_name,
        "expected_direction": spec["direction"],
        "baseline_range": baseline["range"],
        "search_range": [
            change_frame if search_start is None else search_start,
            change_frame + config.search_horizon if search_end is None else search_end,
        ],
        "baseline": baseline["baseline"],
        "mad_sigma": baseline["mad_sigma"],
        "delta": baseline["delta"],
        "target": None,
        "onset_signal_index": None,
        "onset_frame": None,
        "lag_from_change": None,
        "status": baseline["status"],
        "qc_flags": [],
    }
    if baseline["status"] != "ok":
        return result
    target = transition_target(
        transition,
        float(baseline["baseline"]),
        float(baseline["delta"]),
        config,
    )
    result["target"] = target["target"]
    result["qc_flags"] = target["qc_flags"]
    if target["status"] != "ok":
        result["status"] = target["status"]
        return result

    start, end = result["search_range"]
    onset_index = first_sustained_crossing(
        signal,
        float(target["target"]),
        str(target["direction"]),
        int(start),
        int(end),
        config.persistence,
    )
    if onset_index is None:
        result["status"] = "undetected"
        return result
    onset_frame = onset_index + 1
    result.update(
        status="detected",
        onset_signal_index=int(onset_index),
        onset_frame=int(onset_frame),
        lag_from_change=int(onset_frame - change_frame),
    )
    return result


def score_video(
    video_path: str | Path,
    transition: str,
    change_frame: int,
    config: DetectorConfig,
    *,
    admission_frame: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]], dict[str, Any]]:
    """Score both views and return view results, signals, and video metadata."""
    left, right, metadata = read_side_by_side_video(video_path, config)
    view_results: list[dict[str, Any]] = []
    all_signals: list[dict[str, np.ndarray]] = []
    for view_index, frames in enumerate((left, right)):
        signals = flow_signals(frames, config)
        result = detect_onset(signals, transition, change_frame, config)
        finite_fraction = float(np.nanmin(signals["finite_fraction"])) if len(signals["finite_fraction"]) else 0.0
        if finite_fraction < 0.999:
            result["qc_flags"].append("low_finite_flow_fraction")
        result["view_index"] = view_index
        result["finite_flow_fraction"] = finite_fraction
        if admission_frame is not None and result["onset_frame"] is not None:
            result["offset_from_admission"] = int(result["onset_frame"] - admission_frame)
        else:
            result["offset_from_admission"] = None
        view_results.append(result)
        all_signals.append(signals)
    return view_results, all_signals, metadata


def aggregate_pair(
    view_results: list[dict[str, Any]],
    config: DetectorConfig,
) -> dict[str, Any]:
    """Aggregate without treating views as independent seeds."""
    if len(view_results) != 2:
        raise ValueError("expected exactly two view results")
    detected = [result["status"] == "detected" for result in view_results]
    if detected == [True, True]:
        pair_status = "both"
    elif detected == [True, False]:
        pair_status = "left_only"
    elif detected == [False, True]:
        pair_status = "right_only"
    elif any(result["status"].startswith("indeterminate") for result in view_results):
        pair_status = "invalid"
    else:
        pair_status = "neither"

    onsets = [result["onset_frame"] for result in view_results]
    both_mean = None
    disagreement = None
    qc_flags: list[str] = []
    if pair_status == "both":
        both_mean = float(np.mean(onsets))
        disagreement = int(abs(int(onsets[0]) - int(onsets[1])))
        if disagreement > config.pair_disagreement_flag_frames:
            qc_flags.append("view_onset_disagreement")
    return {
        "view_statuses": [result["status"] for result in view_results],
        "view_onsets": onsets,
        "view_lags": [result["lag_from_change"] for result in view_results],
        "detected_count": int(sum(detected)),
        "pair_status": pair_status,
        "both_detected_onset_mean": both_mean,
        "onset_disagreement_frames": disagreement,
        "qc_flags": qc_flags,
    }


def detect_paired_divergence(
    control_magnitude: np.ndarray,
    intervention_magnitude: np.ndarray,
    change_frame: int,
    admission_frame: int,
    config: DetectorConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    """Detect when control and intervention motion magnitudes diverge."""
    length = min(len(control_magnitude), len(intervention_magnitude))
    difference = np.asarray(control_magnitude[:length] - intervention_magnitude[:length], dtype=np.float64)
    baseline = robust_baseline(difference, change_frame, config)
    result: dict[str, Any] = {
        "signal_name": "control_magnitude_minus_stop_magnitude",
        "baseline_range": baseline["range"],
        "search_range": [admission_frame, admission_frame + config.search_horizon],
        "baseline": baseline["baseline"],
        "mad_sigma": baseline["mad_sigma"],
        "delta": baseline["delta"],
        "target": None,
        "onset_signal_index": None,
        "onset_frame": None,
        "lag_from_change": None,
        "offset_from_admission": None,
        "status": baseline["status"],
        "qc_flags": [],
    }
    if baseline["status"] != "ok":
        return result, difference
    target = float(baseline["baseline"]) + float(baseline["delta"])
    result["target"] = target
    onset_index = first_sustained_crossing(
        difference,
        target,
        "rise",
        admission_frame,
        admission_frame + config.search_horizon,
        config.persistence,
    )
    if onset_index is None:
        result["status"] = "undetected"
        return result, difference
    onset_frame = onset_index + 1
    result.update(
        status="detected",
        onset_signal_index=int(onset_index),
        onset_frame=int(onset_frame),
        lag_from_change=int(onset_frame - change_frame),
        offset_from_admission=int(onset_frame - admission_frame),
    )
    return result, difference
