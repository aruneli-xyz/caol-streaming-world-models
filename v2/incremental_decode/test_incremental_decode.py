import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import (
    ProbeError,
    array_sha256,
    artifact_metadata,
    atomic_save_npy,
    atomic_write_json,
    compare_exact,
    global_normalization_slice,
    seal_manifest,
    split_gamma_views,
    strict_resume,
    validate_block_plan,
    verify_sealed_manifest,
)
from stop_wallclock import record_event, run_identity, validate_config


HERE = Path(__file__).resolve().parent


def test_gamma_two_view_split_preserves_view_local_time():
    latent = torch.arange(1 * 2 * 2 * 6 * 1 * 1).reshape(1, 2, 12, 1, 1)
    flat = split_gamma_views(latent, n_views=2)
    assert tuple(flat.shape) == (2, 2, 6, 1, 1)
    assert torch.equal(flat[0], latent[0, :, :6])
    assert torch.equal(flat[1], latent[0, :, 6:])


def test_block_plan_is_three_latents_and_complete_only():
    assert validate_block_plan(8, max_blocks=None) == [(0, 3), (3, 6)]
    assert validate_block_plan(12, max_blocks=3) == [(0, 3), (3, 6), (6, 9)]
    with pytest.raises(ValueError, match="3-latent"):
        validate_block_plan(9, block_latents=2)
    with pytest.raises(ValueError, match="fewer"):
        validate_block_plan(2)


def test_global_normalization_does_not_restart_at_zero():
    latent = torch.ones((2, 1, 3, 1, 1))
    mean = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1, 1)
    std = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 12, 1, 1)
    result = global_normalization_slice(
        latent, mean, std, global_start=3
    )
    expected = torch.tensor([7.0, 9.0, 11.0]).reshape(1, 1, 3, 1, 1)
    assert torch.equal(result[:1], expected)
    assert torch.equal(result[1:], expected)


def test_exact_comparison_reports_each_view_and_hash():
    first = np.zeros((2, 5, 2, 2, 3), dtype=np.uint8)
    second = first.copy()
    exact = compare_exact(first, second)
    assert exact["exact"] is True
    assert exact["max_error"] == 0
    assert exact["first_sha256"] == exact["second_sha256"] == array_sha256(first)
    assert [row["frames"] for row in exact["views"]] == [5, 5]

    second[1, 3, 0, 0, 2] = 7
    mismatch = compare_exact(first, second)
    assert mismatch["exact"] is False
    assert mismatch["max_error"] == 7
    assert mismatch["views"][0]["exact"] is True
    assert mismatch["views"][1]["changed_frames"] == [3]


def test_manifest_seal_and_strict_resume_are_hash_bound(tmp_path):
    output = tmp_path / "run"
    array_path = output / "raw.npy"
    atomic_save_npy(array_path, np.arange(8, dtype=np.uint8))
    identity = {"source": "canonical", "blocks": 2}
    body = {
        "schema_version": "rtwm-v2-incremental-decode-1",
        "status": "complete",
        "identity": identity,
        "artifacts": {"raw": artifact_metadata(array_path, root=output)},
    }
    manifest = seal_manifest(body)
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    resumed = strict_resume(manifest_path, expected_identity=identity)
    assert resumed == manifest
    verify_sealed_manifest(json.loads(manifest_path.read_text()))

    with pytest.raises(ProbeError, match="identity differs"):
        strict_resume(manifest_path, expected_identity={"source": "other"})

    array_path.write_bytes(b"corrupt")
    with pytest.raises(ProbeError, match="size mismatch"):
        strict_resume(manifest_path, expected_identity=identity)


def test_manifest_tampering_is_refused():
    manifest = seal_manifest({"status": "complete", "identity": {}})
    manifest["status"] = "failed"
    with pytest.raises(ProbeError, match="seal mismatch"):
        verify_sealed_manifest(manifest)


def test_stop_wallclock_config_freezes_counterbalance_and_receipt():
    config = json.loads((HERE / "stop_wallclock_config.json").read_text())
    validate_config(config)
    assert [
        (row["seed"], row["delay"], row["admission_latent"])
        for row in config["counterbalanced_run_order"]
    ] == [
        (301, "d0", 27),
        (301, "d1", 30),
        (302, "d1", 30),
        (302, "d0", 27),
    ]
    assert config["synthetic_action_receipt_latent"] == 24
    assert config["uncertainty"]["method"] == "sample_standard_deviation"


def test_stop_wallclock_event_and_run_identity_are_explicit():
    events = []
    record_event(
        events,
        "synthetic_action_received",
        timestamp_ns=1_250_000,
        origin_ns=1_000_000,
        receipt_latent=24,
    )
    assert events == [
        {
            "event": "synthetic_action_received",
            "monotonic_ns": 1_250_000,
            "relative_ms": 0.25,
            "receipt_latent": 24,
        }
    ]
    identity = run_identity(
        {"schema": "test"},
        {"run_index": 3, "seed": 302, "delay": "d1", "admission_latent": 30},
    )
    assert identity["run"] == {
        "run_index": 3,
        "seed": 302,
        "delay": "d1",
        "admission_latent": 30,
    }
