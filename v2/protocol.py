"""Protocol loading, validation, and stable identity helpers for RTWM v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProtocolError(ValueError):
    """Raised when a v2 protocol is incomplete or internally inconsistent."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ProtocolError(f"missing {context}.{key}")
    return mapping[key]


def validate_protocol(data: Mapping[str, Any]) -> None:
    allowed_root = {
        "schema_version",
        "experiment",
        "model",
        "pilot",
        "confirmatory",
        "detector",
        "paired_detector",
        "validation",
        "receptive_field",
        "view_result_fields",
        "pair_result_fields",
    }
    unknown = sorted(set(data) - allowed_root)
    if unknown:
        raise ProtocolError(f"unknown protocol keys: {unknown}")
    for key in ("schema_version", "experiment", "model", "detector", "validation"):
        _require(data, key, "protocol")
    if ("pilot" in data) == ("confirmatory" in data):
        raise ProtocolError("protocol must contain exactly one of pilot or confirmatory")

    model = data["model"]
    detector = data["detector"]
    validation = data["validation"]
    experiment = data.get("pilot", data.get("confirmatory"))
    if not all(isinstance(section, Mapping) for section in (model, experiment, detector, validation)):
        raise ProtocolError("model, experiment, detector, and validation must be objects")

    n_frames = int(_require(model, "n_frames", "model"))
    expected_frames = int(_require(validation, "expected_frames", "validation"))
    if n_frames != expected_frames:
        raise ProtocolError(
            f"model.n_frames ({n_frames}) must equal validation.expected_frames ({expected_frames})"
        )
    context = "pilot" if "pilot" in data else "confirmatory"
    if int(_require(experiment, "change_frame", context)) >= n_frames:
        raise ProtocolError(f"{context}.change_frame must be before the final frame")

    minimum_finite = float(
        _require(validation, "minimum_finite_flow_fraction", "validation")
    )
    if not 0.0 < minimum_finite <= 1.0:
        raise ProtocolError("validation.minimum_finite_flow_fraction must be in (0, 1]")
    require_synthetic = _require(validation, "require_synthetic_tests", "validation")
    if not isinstance(require_synthetic, bool):
        raise ProtocolError("validation.require_synthetic_tests must be boolean")
    if "pilot" in data:
        require_prefix = _require(validation, "require_prefix_comparison", "validation")
        if not isinstance(require_prefix, bool):
            raise ProtocolError("validation.require_prefix_comparison must be boolean")
    else:
        required_artifacts = list(_require(validation, "required_artifacts", "validation"))
        if set(required_artifacts) != {"latent", "decoded_u8", "detector_frames"}:
            raise ProtocolError("confirmatory required_artifacts must be latent/raw/detector frames")

    persistence = int(_require(detector, "persistence", "detector"))
    if persistence < 1:
        raise ProtocolError("detector.persistence must be positive")
    if "paired_detector" in data:
        paired = data["paired_detector"]
        if not isinstance(paired, Mapping):
            raise ProtocolError("paired_detector must be an object")
        paired_keys = (
            (
                "baseline_before",
                "baseline_gap",
                "search_horizon_after_admission",
                "persistence",
                "mad_multiplier",
                "relative_delta",
                "absolute_delta",
            )
            if "pilot" in data
            else ("signal", "direction", "search_horizon_after_admission", "persistence", "threshold_fit", "crossing_operator")
        )
        for key in paired_keys:
            _require(paired, key, "paired_detector")

    if "pilot" in data:
        pilot = data["pilot"]
        stage1 = list(_require(pilot, "stage1_scenes", "pilot"))
        stage2 = list(_require(pilot, "stage2_scenes", "pilot"))
        if not stage1 or not stage2:
            raise ProtocolError("both pilot stage scene lists must be non-empty")
        if set(stage1).intersection(stage2):
            raise ProtocolError("stage1 and stage2 scenes must be disjoint")
    else:
        confirmatory = data["confirmatory"]
        scenes = list(_require(confirmatory, "scenes", "confirmatory"))
        if len(scenes) != 2 or len(set(scenes)) != 2:
            raise ProtocolError("confirmatory protocol requires two distinct scenes")
        for split in ("calibration", "validation", "test"):
            _require(confirmatory, split, "confirmatory")
        calibration = set(confirmatory["calibration"]["seeds"])
        validation_seeds = set(confirmatory["validation"]["seeds"])
        test = set(confirmatory["test"]["seeds"])
        if calibration & validation_seeds or calibration & test or validation_seeds & test:
            raise ProtocolError("confirmatory seed splits must be disjoint")
        delays = list(_require(confirmatory["test"], "delays", "confirmatory.test"))
        names = [str(delay["name"]) for delay in delays]
        admission_latents = [int(delay["admission_latent"]) for delay in delays]
        if names != ["d0", "d1"] or admission_latents != [27, 30]:
            raise ProtocolError("confirmatory delays must be d0/d1 at latent 27/30")


@dataclass(frozen=True)
class LoadedProtocol:
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


def load_protocol(path: str | Path) -> LoadedProtocol:
    protocol_path = Path(path).resolve()
    try:
        data = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load protocol {protocol_path}: {error}") from error
    if not isinstance(data, dict):
        raise ProtocolError("protocol root must be an object")
    validate_protocol(data)
    return LoadedProtocol(
        path=protocol_path,
        data=data,
        file_sha256=file_sha256(protocol_path),
        canonical_sha256=canonical_sha256(data),
    )
