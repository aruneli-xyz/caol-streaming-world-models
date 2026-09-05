from pathlib import Path

import numpy as np

from canonical_stop import (
    action_sequence, array_sha256, load_stop_protocol, stop_specs,
    validate_canonical_evidence,
)
from flow_detector import first_sustained_strict_crossing
HERE = Path(__file__).resolve().parent
PROTOCOL = load_stop_protocol(HERE / "config" / "canonical_stop.json")


def test_stop_grid_is_exactly_24_d0_d1_pairs():
    rows = stop_specs(PROTOCOL)
    assert len(rows) == 24
    assert {row["delay"] for row in rows} == {"d0", "d1"}
    assert {row["admission_latent"] for row in rows} == {27, 30}


def test_stop_action_is_canonical_forward_to_all_zero_stay():
    forward = action_sequence(PROTOCOL, stop=False)
    stop = action_sequence(PROTOCOL, stop=True)
    change = PROTOCOL.data["model"]["change_frame"]
    assert np.flatnonzero(forward["keyboard"][0, 0]).tolist() == [11]
    assert np.flatnonzero(stop["keyboard"][0, change - 1]).tolist() == [11]
    assert np.count_nonzero(stop["keyboard"][0, change:]) == 0
    assert np.count_nonzero(stop["camera"]) == 0


def test_action_tensor_hashes_are_stable_and_distinct():
    forward = action_sequence(PROTOCOL, stop=False)
    stop = action_sequence(PROTOCOL, stop=True)
    assert len(array_sha256(forward["keyboard"])) == 64
    assert array_sha256(forward["keyboard"]) != array_sha256(stop["keyboard"])
    assert array_sha256(forward["camera"]) == array_sha256(stop["camera"])


def test_canonical_evidence_hashes_and_identities_pass():
    result = validate_canonical_evidence(PROTOCOL, verify_artifacts=False)
    assert result["allowed"]
    assert len(result["controls"]) == 12


def test_strict_endpoint_indexing_uses_destination_frame():
    signal = np.zeros(30)
    signal[9:12] = 1.0
    crossing = first_sustained_strict_crossing(signal, 0.0, "rise", 8, 20, 3)
    assert crossing == 9
    assert crossing + 1 == 10
