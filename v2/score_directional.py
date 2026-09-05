"""Score audit or held-out directional interventions from measured raw support."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifacts import atomic_write, atomic_write_json, verify_rollout_artifacts  # noqa: E402
from directional_study import (  # noqa: E402
    action_protocol_sha256,
    detector_config,
    flow_endpoint_sample,
    intervention_specs,
    load_directional_protocol,
    persistent_effect,
)
from flow_detector import first_sustained_strict_crossing  # noqa: E402
from protocol import canonical_sha256, file_sha256  # noqa: E402
from score_confirmatory import bootstrap_mean_ci  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        lambda temporary: temporary.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        ),
    )


def load_primary_thresholds(
    calibration: Path, protocol: Any, manifest: Path
) -> dict[tuple[str, int], float]:
    gate = load_json(calibration / "gate_report.json")
    if (
        gate.get("schema_version") != "rtwm-v2-directional-null-gate-1"
        or gate.get("config_sha256") != protocol.file_sha256
        or gate.get("action_protocol_sha256") != action_protocol_sha256(protocol)
        or gate.get("manifest_sha256") != file_sha256(manifest)
        or gate.get("passed") is not True
    ):
        raise RuntimeError("fresh canonical null calibration is stale or failed")
    root = calibration / "locks"
    thresholds = {}
    for path in sorted(root.glob("*.json")):
        payload = load_json(path)
        thresholds[(payload["scene"], int(payload["view_index"]))] = float(payload["threshold"])
    if set(thresholds) != {
        (scene, view)
        for scene in protocol.data["design"]["scenes"]
        for view in (0, 1)
    }:
        raise RuntimeError("paired null calibration does not cover all scene/views")
    return thresholds


def require_audit_gate(path: Path, protocol: Any, audit_results: Path) -> dict[str, Any]:
    gate = load_json(path)
    actual = canonical_sha256({key: value for key, value in gate.items() if key != "gate_sha256"})
    if (
        gate.get("schema_version") != "rtwm-v2-directional-audit-gate-1"
        or gate.get("config_sha256") != protocol.file_sha256
        or gate.get("protocol_sha256") != protocol.canonical_sha256
        or gate.get("gate_sha256") != actual
        or gate.get("audit_results_sha256") != file_sha256(audit_results)
    ):
        raise RuntimeError("held-out scoring blocked by stale or invalid audit freeze")
    return gate


def summarize_seed_bootstrap(rows: list[dict[str, Any]], protocol: Any) -> dict[str, Any]:
    samples = int(protocol.data["bootstrap"]["samples"])
    base_seed = int(protocol.data["bootstrap"]["seed"])
    output: dict[str, Any] = {}
    for transition_index, transition in enumerate(protocol.data["design"]["transitions"]):
        transition_rows = [row for row in rows if row["transition"] == transition]
        by_scene = {}
        for scene_index, scene in enumerate(protocol.data["design"]["scenes"]):
            selected = [row for row in transition_rows if row["scene"] == scene]
            seeds = sorted({int(row["seed"]) for row in selected})
            seed_primary = []
            seed_obedience = []
            seed_l2 = []
            feature_names = sorted({
                feature
                for row in selected
                for feature in row["feature_effects_expected_direction"]
            })
            seed_features: dict[str, list[float]] = {
                feature: [] for feature in feature_names
            }
            seed_residual: list[float] = []
            for seed in seeds:
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                seed_primary.append(float(all(row["primary_detected"] for row in seed_rows)))
                seed_obedience.append(float(all(row["directional_pass"] for row in seed_rows)))
                seed_l2.append(float(np.mean([row["primary_peak"] for row in seed_rows])))
                for feature in feature_names:
                    seed_features[feature].append(float(np.mean([
                        row["feature_effects_expected_direction"][feature]
                        for row in seed_rows
                    ])))
                residuals = [
                    row["residual_flow_l2_mean"] for row in seed_rows
                    if row.get("residual_flow_l2_mean") is not None
                ]
                if residuals:
                    seed_residual.append(float(np.mean(residuals)))
            by_scene[scene] = {
                "seed_count": len(seeds),
                "primary_detection_fraction": bootstrap_mean_ci(
                    seed_primary, seed=base_seed + 100 * transition_index + 10 * scene_index,
                    samples=samples,
                ),
                "directional_obedience_fraction": bootstrap_mean_ci(
                    seed_obedience, seed=base_seed + 100 * transition_index + 10 * scene_index + 1,
                    samples=samples,
                ),
                "paired_flow_field_l2_peak": bootstrap_mean_ci(
                    seed_l2, seed=base_seed + 100 * transition_index + 10 * scene_index + 2,
                    samples=samples,
                ),
                "directional_features_expected_direction": {
                    feature: bootstrap_mean_ci(
                        values,
                        seed=base_seed + 1000 + 100 * transition_index
                        + 10 * scene_index + feature_index,
                        samples=samples,
                    )
                    for feature_index, (feature, values) in enumerate(
                        seed_features.items()
                    )
                },
                "residual_flow_l2": (
                    bootstrap_mean_ci(
                        seed_residual,
                        seed=base_seed + 2000 + 100 * transition_index + scene_index,
                        samples=samples,
                    )
                    if seed_residual else None
                ),
            }
        output[transition] = by_scene
    return {
        "schema_version": "rtwm-v2-directional-bootstrap-1",
        "unit": "seed; views are repeated observations",
        "samples": samples,
        "transitions": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config" / "directional_d0.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "results" / "directional_d0" / "manifest.json")
    parser.add_argument("--root", type=Path, default=HERE / "results" / "directional_d0")
    parser.add_argument("--calibration", type=Path, default=HERE / "results" / "directional_d0_calibration")
    parser.add_argument("--partition", choices=["audit", "held_out"], required=True)
    parser.add_argument("--audit-gate", type=Path)
    parser.add_argument("--audit-results", type=Path, default=HERE / "results" / "directional_d0_audit" / "view_results.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_directional_protocol(args.config)
    thresholds = load_primary_thresholds(args.calibration, protocol, args.manifest)
    gate = None
    if args.partition == "held_out":
        if args.audit_gate is None:
            raise RuntimeError("--audit-gate is required for held-out scoring")
        gate = require_audit_gate(args.audit_gate, protocol, args.audit_results)

    generated_manifest = load_json(args.manifest)
    if (
        generated_manifest.get("config_sha256") != protocol.file_sha256
        or generated_manifest.get("protocol_sha256") != protocol.canonical_sha256
        or generated_manifest.get("action_protocol_sha256") != action_protocol_sha256(protocol)
        or generated_manifest.get("planned_rollouts") != 52
    ):
        raise RuntimeError("directional generation manifest protocol mismatch")
    generated = {
        row["rollout_id"]: row
        for row in generated_manifest["rollouts"]
        if row.get("status") == "complete"
    }
    controls = {
        (row["scene"], int(row["seed"])): row
        for row in generated_manifest["rollouts"]
        if row.get("stage") == "test"
        and row.get("arm") == "control"
        and row.get("status") == "complete"
        and row.get("action_protocol_sha256") == action_protocol_sha256(protocol)
    }
    config = detector_config(protocol)
    persistence = int(protocol.data["endpoints"]["persistence"])
    horizon = int(protocol.data["endpoints"]["window_frames"])
    expected_specs = [
        row for row in intervention_specs(protocol) if row["partition"] == args.partition
    ]
    rows: list[dict[str, Any]] = []
    for spec in expected_specs:
        record = generated.get(spec["rollout_id"])
        if record is None:
            raise RuntimeError(f"missing complete intervention {spec['rollout_id']}")
        control_record = controls[(spec["scene"], int(spec["seed"]))]
        intervention_paths = verify_rollout_artifacts(
            args.root, record,
            required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
        )
        control_paths = verify_rollout_artifacts(
            args.root, control_record,
            required=("latent", "decoded_u8", "detector_frames", "action_tensors"),
        )
        with np.load(intervention_paths["detector_frames"]) as payload:
            intervention = np.asarray(payload["views"])
        with np.load(control_paths["detector_frames"]) as payload:
            control = np.asarray(payload["views"])
        if intervention.shape != control.shape:
            raise RuntimeError(f"detector frame shape mismatch for {spec['rollout_id']}")
        starts = {int(key): int(value) for key, value in record["raw_support_starts"].items()}
        transition = spec["transition"]
        expected_signs = protocol.data["endpoints"][transition]["expected_signs"]
        for view in (0, 1):
            start = max(0, starts[view] - 1)
            end = min(intervention.shape[1] - 1, starts[view] + horizon)
            samples = [
                flow_endpoint_sample(
                    control[view, index], control[view, index + 1],
                    intervention[view, index], intervention[view, index + 1],
                    config, transition,
                )
                for index in range(start, end)
            ]
            primary = np.asarray([sample["paired_flow_field_l2"] for sample in samples])
            threshold = thresholds[(spec["scene"], view)]
            crossing = first_sustained_strict_crossing(
                primary, threshold, "rise", 0, len(primary), persistence
            )
            feature_effects = {
                feature: persistent_effect(
                    [sample[feature] for sample in samples],
                    int(sign), persistence,
                )
                for feature, sign in expected_signs.items()
            }
            feature_thresholds = (
                {feature: 0.0 for feature in feature_effects}
                if gate is None
                else gate["thresholds"][transition]
            )
            feature_pass = {
                feature: (
                    math.isfinite(effect)
                    and effect > 0
                    and effect >= float(feature_thresholds[feature])
                )
                for feature, effect in feature_effects.items()
            }
            row = {
                "schema_version": "rtwm-v2-directional-view-1",
                "partition": args.partition,
                "scene": spec["scene"],
                "seed": int(spec["seed"]),
                "transition": transition,
                "view_index": view,
                "effect_start_frame": starts[view],
                "search_start_signal_index": start,
                "primary_threshold": threshold,
                "primary_detected": crossing is not None,
                "primary_onset_signal_index": None if crossing is None else start + crossing,
                "primary_onset_frame": None if crossing is None else start + crossing + 1,
                "primary_peak": float(np.max(primary)),
                "feature_effects_expected_direction": feature_effects,
                "feature_thresholds": feature_thresholds,
                "feature_pass": feature_pass,
                "directional_pass": all(feature_pass.values()),
                "residual_flow_l2_mean": (
                    float(np.mean([sample["residual_flow_l2"] for sample in samples]))
                    if transition == "yaw_positive" else None
                ),
            }
            rows.append(row)

    summaries = summarize_seed_bootstrap(rows, protocol)
    minimum_primary = float(protocol.data["audit_gate"]["minimum_primary_detection_seed_fraction"])
    minimum_directional = float(protocol.data["audit_gate"]["minimum_held_out_seed_fraction"])
    claims: dict[str, Any] = {}
    for transition in protocol.data["design"]["transitions"]:
        scenes = {}
        for scene in protocol.data["design"]["scenes"]:
            stats = summaries["transitions"][transition][scene]
            primary_fraction = stats["primary_detection_fraction"]["mean"]
            directional_fraction = stats["directional_obedience_fraction"]["mean"]
            causal = primary_fraction is not None and primary_fraction >= minimum_primary
            obedience = (
                args.partition == "held_out"
                and gate is not None
                and gate.get("allowed") is True
                and causal
                and directional_fraction is not None
                and directional_fraction >= minimum_directional
            )
            scenes[scene] = {
                "causal_divergence_pass": causal,
                "directional_obedience_pass": obedience,
                "claim": (
                    "directional_obedience"
                    if obedience else "causal_divergence_only" if causal else "no_supported_claim"
                ),
            }
        claims[transition] = {
            "scenes": scenes,
            "causal_divergence_pass": all(item["causal_divergence_pass"] for item in scenes.values()),
            "directional_obedience_pass": all(item["directional_obedience_pass"] for item in scenes.values()),
        }
        if claims[transition]["causal_divergence_pass"] and not claims[transition]["directional_obedience_pass"]:
            claims[transition]["failed_claim"] = "directional obedience explicitly failed"

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "view_results.jsonl", rows)
    summaries.update(protocol.identity)
    summaries["partition"] = args.partition
    summaries["audit_gate_allowed"] = None if gate is None else bool(gate.get("allowed"))
    summaries["audit_gate_block_reasons"] = [] if gate is None else gate.get("block_reasons", [])
    summaries["claims"] = claims
    summaries["summary_sha256"] = canonical_sha256(summaries)
    atomic_write_json(args.output / "statistics.json", summaries)
    report = {
        "schema_version": "rtwm-v2-directional-analysis-gate-1",
        **protocol.identity,
        "partition": args.partition,
        "expected_views": len(expected_specs) * 2,
        "observed_views": len(rows),
        "complete": len(rows) == len(expected_specs) * 2,
        "audit_gate_allowed": None if gate is None else bool(gate.get("allowed")),
        "audit_gate_block_reasons": [] if gate is None else gate.get("block_reasons", []),
        "outputs": {
            "view_results.jsonl": file_sha256(args.output / "view_results.jsonl"),
            "statistics.json": file_sha256(args.output / "statistics.json"),
        },
        "claims": claims,
    }
    report["gate_sha256"] = canonical_sha256(report)
    atomic_write_json(args.output / "analysis_gate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
