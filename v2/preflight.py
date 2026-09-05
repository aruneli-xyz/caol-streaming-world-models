"""Lightweight integrity checks that run before expensive model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))

from artifacts import atomic_write_json  # noqa: E402
from flow_detector import (  # noqa: E402
    DetectorConfig,
    PairedDetectorConfig,
    detect_onset,
    first_sustained_strict_crossing,
    flow_signals,
    paired_flow_field_divergence,
    split_views,
)
from protocol import LoadedProtocol, canonical_sha256, load_protocol  # noqa: E402


class PreflightError(RuntimeError):
    """Raised when required preflight evidence does not pass."""


def git_source_identity(repo: str | Path) -> dict[str, Any]:
    repository = Path(repo)
    source_paths = (
        "gamma_world",
        "packages",
        "scripts",
        "pyproject.toml",
        "uv.lock",
    )

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repository,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    commit = run("rev-parse", "HEAD")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *source_paths],
        cwd=repository,
    )
    untracked = [
        line
        for line in run(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *source_paths,
        ).splitlines()
        if line.startswith("?? ")
    ]
    digest = hashlib.sha256()
    digest.update(diff)
    for entry in sorted(untracked):
        relative = entry[3:]
        path = repository / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(diff or untracked),
        "diff_sha256": digest.hexdigest(),
    }


def require_source(protocol: LoadedProtocol, source: dict[str, Any]) -> None:
    expected = str(protocol.data["model"]["source_commit"])
    if source["commit"] != expected:
        raise PreflightError(
            f"Gamma-World source commit mismatch: expected {expected}, got {source['commit']}"
        )


def _textured_frame(config: DetectorConfig, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((config.resize_height, config.resize_width), dtype=np.uint8)
    for x, y, value in zip(
        rng.integers(10, config.resize_width - 10, 350),
        rng.integers(10, config.resize_height - 10, 350),
        rng.integers(80, 256, 350),
    ):
        cv2.circle(frame, (int(x), int(y)), 2, int(value), -1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _translated(frame: np.ndarray, dx: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def _scaled(frame: np.ndarray, scale: float, config: DetectorConfig) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        (config.radial_center_x, config.radial_center_y),
        0,
        scale,
    )
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def run_synthetic_suite(protocol: LoadedProtocol) -> dict[str, Any]:
    """Run the frozen sign, indexing, view-order, and finite-flow checks."""
    config = DetectorConfig.from_json(protocol.path)
    paired_config = (
        PairedDetectorConfig.from_json(protocol.path)
        if "paired_detector" in protocol.data
        and protocol.data["paired_detector"].get("signal") == "flow_field_l2"
        else None
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
        except Exception as error:  # report every independent check
            checks.append({"name": name, "passed": False, "error": repr(error)})
        else:
            checks.append({"name": name, "passed": True, "error": None})

    frame = _textured_frame(config)

    def static_check() -> None:
        signals = flow_signals([frame, frame.copy()], config)
        assert signals["magnitude"][0] < 1e-3
        assert abs(signals["horizontal"][0]) < 1e-3
        assert abs(signals["radial"][0]) < 1e-3

    def horizontal_check() -> None:
        assert flow_signals([frame, _translated(frame, 2)], config)["horizontal"][0] > 0.5

    def radial_check() -> None:
        expansion = flow_signals([frame, _scaled(frame, 1.04, config)], config)["radial"][0]
        contraction = flow_signals([frame, _scaled(frame, 0.96, config)], config)["radial"][0]
        assert expansion > 0.1 and contraction < -0.1

    def indexing_check() -> None:
        change = 24
        magnitude = np.ones(80, dtype=np.float64)
        magnitude[change:change + config.persistence] = 0.1
        zeros = np.zeros_like(magnitude)
        result = detect_onset(
            {
                "magnitude": magnitude,
                "horizontal": zeros,
                "radial": zeros,
                "finite_fraction": np.ones_like(magnitude),
            },
            "stop",
            change,
            config,
        )
        assert result["onset_signal_index"] == change
        assert result["onset_frame"] == change + 1

    def view_order_check() -> None:
        left = np.full((20, 30, 3), 25, dtype=np.uint8)
        right = np.full((20, 30, 3), 225, dtype=np.uint8)
        actual_left, actual_right = split_views(np.concatenate([left, right], axis=1))
        assert float(actual_left.mean()) == 25
        assert float(actual_right.mean()) == 225

    def finite_rejection_check() -> None:
        signal = np.array([1.0, np.nan, np.inf])
        assert float(np.isfinite(signal).mean()) < float(
            protocol.data["validation"]["minimum_finite_flow_fraction"]
        )

    check("static_near_zero", static_check)
    check("positive_horizontal_sign", horizontal_check)
    check("radial_opposite_signs", radial_check)
    check("rendered_endpoint_indexing", indexing_check)
    check("view_order", view_order_check)
    check("nonfinite_rejected", finite_rejection_check)
    if paired_config is not None:
        def paired_duplicate_check() -> None:
            divergence = paired_flow_field_divergence(
                [frame, frame],
                [frame, frame],
                config,
            )
            assert np.max(divergence) == 0

        check("paired_exact_duplicate_zero", paired_duplicate_check)

        def strict_equality_check() -> None:
            threshold = 0.5
            assert first_sustained_strict_crossing(
                np.full(6, threshold),
                threshold,
                "rise",
                0,
                6,
                paired_config.persistence,
            ) is None

        check("paired_strict_equality_not_crossing", strict_equality_check)

        def paired_translation_check() -> None:
            control = [frame] + [_translated(frame, float(index)) for index in range(1, 7)]
            intervention = [frame] * 7
            signal = paired_flow_field_divergence(control, intervention, config)
            assert np.max(signal) > 0.25

        check("paired_translation_detectable", paired_translation_check)
    passed = all(item["passed"] for item in checks)
    report: dict[str, Any] = {
        "schema_version": "rtwm-v2-synthetic-preflight-1",
        **protocol.identity,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
        "passed": passed,
    }
    report["evidence_sha256"] = canonical_sha256(report)
    return report


def require_synthetic_suite(protocol: LoadedProtocol) -> dict[str, Any]:
    report = run_synthetic_suite(protocol)
    if protocol.data["validation"]["require_synthetic_tests"] and not report["passed"]:
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        raise PreflightError(f"required synthetic checks failed: {', '.join(failed)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=V2 / "config" / "pilot.json")
    parser.add_argument("--output", type=Path, default=V2 / "results" / "preflight.json")
    parser.add_argument("--source-repo", type=Path, default=None)
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    report = require_synthetic_suite(protocol)
    if args.source_repo is not None:
        source = git_source_identity(args.source_repo)
        require_source(protocol, source)
        report["source"] = source
        report["evidence_sha256"] = canonical_sha256(
            {key: value for key, value in report.items() if key != "evidence_sha256"}
        )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
