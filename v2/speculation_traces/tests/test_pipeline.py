from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from actions import ACTION_KEYS, KEYBOARD_KEYS, pack_keyboard, unpack_keyboard
from build_protocol import build_protocol
from common import ROOT, canonical_json_bytes, load_json, sha256_file
from convert import (
    convert_records,
    pack_keyboard_array,
    quantize_camera,
    read_jsonl,
    resample_20_to_16,
)
from download import _assert_action_url, _count_valid_jsonl
from validate import validate_all

SOLARIS = Path("/home/arunkumareli/code/research/safeswm/external/solaris")
VPT_FIXTURES = SOLARIS / "src/tests/fixtures/vpt_action_slices"
GAMMA_DATA = Path(
    "/home/arunkumareli/code/research/safeswm/external/Gamma-World/data"
)

SLICE_RANGES = {
    "all_off.jsonl": (0, 1),
    "each_binary_on.jsonl": (0, 33),
    "camera_ranges.jsonl": (0, 29),
    "attack_stuck.jsonl": (0, 4),
    "attack_stuck_offset_unstuck_before.jsonl": (2, 4),
    "attack_stuck_offset_unstuck_during.jsonl": (1, 5),
    "hotbar_offset_set_before_slice.jsonl": (2, 4),
}


class SolarisSemanticsTests(unittest.TestCase):
    def test_all_upstream_solaris_fixtures_exactly(self) -> None:
        for filename, (start, stop) in SLICE_RANGES.items():
            with self.subTest(filename=filename):
                records = read_jsonl(VPT_FIXTURES / filename)
                keyboard, camera = convert_records(records)
                actual = np.concatenate(
                    (keyboard.astype(np.float32), camera), axis=1
                )[start:stop]
                expected = np.asarray(
                    json.loads(
                        (
                            VPT_FIXTURES
                            / filename.replace(".jsonl", ".expected.json")
                        ).read_text(encoding="utf-8")
                    ),
                    dtype=np.float32,
                )
                if filename == "camera_ranges.jsonl":
                    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
                else:
                    np.testing.assert_array_equal(actual, expected)

    def test_order_is_authoritative_solaris_order(self) -> None:
        self.assertEqual(len(KEYBOARD_KEYS), 23)
        self.assertEqual(ACTION_KEYS[-2:], ("cameraX", "cameraY"))
        self.assertEqual(
            KEYBOARD_KEYS,
            (
                "inventory",
                "ESC",
                "hotbar.1",
                "hotbar.2",
                "hotbar.3",
                "hotbar.4",
                "hotbar.5",
                "hotbar.6",
                "hotbar.7",
                "hotbar.8",
                "hotbar.9",
                "forward",
                "back",
                "left",
                "right",
                "jump",
                "sneak",
                "sprint",
                "swapHands",
                "attack",
                "use",
                "pickItem",
                "drop",
            ),
        )

    def test_lossless_keyboard_token_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        keyboard = rng.integers(0, 2, size=(100, 23), dtype=np.uint8)
        tokens = pack_keyboard_array(keyboard)
        for row, token in zip(keyboard.tolist(), tokens.tolist(), strict=True):
            self.assertEqual(pack_keyboard(row), token)
            self.assertEqual(unpack_keyboard(token), row)


class GammaCompatibilityTests(unittest.TestCase):
    def test_bundled_gamma_actions_use_solaris_positions(self) -> None:
        expected_active_names = {
            "buildHouse_flat/action_left.json": {"back", "jump", "sneak", "use"},
            "buildHouse_flat/action_right.json": {
                "hotbar.8",
                "forward",
                "jump",
                "use",
            },
            "buildTower_normal/action_left.json": {"hotbar.5", "jump", "use"},
            "buildTower_normal/action_right.json": {"hotbar.5", "jump", "use"},
        }
        for relative, expected in expected_active_names.items():
            with self.subTest(relative=relative):
                payload = json.loads((GAMMA_DATA / relative).read_text())
                keyboard = np.asarray(payload["keyboard"])
                camera = np.asarray(payload["camera"])
                self.assertEqual(keyboard.shape[1], len(KEYBOARD_KEYS))
                self.assertEqual(camera.shape, (keyboard.shape[0], 2))
                active = {
                    KEYBOARD_KEYS[index]
                    for index in np.flatnonzero(np.any(keyboard != 0, axis=0))
                }
                self.assertEqual(active, expected)


class ResamplingAndQuantizationTests(unittest.TestCase):
    def test_rational_resample_is_deterministic_and_conservative(self) -> None:
        keyboard = np.zeros((7, 23), dtype=np.uint8)
        keyboard[:, 11] = np.arange(7) % 2
        camera = np.arange(14, dtype=np.float32).reshape(7, 2)
        out_keyboard, out_camera, source_indices = resample_20_to_16(
            keyboard, camera
        )
        np.testing.assert_array_equal(source_indices, [0, 1, 2, 3, 5, 6])
        np.testing.assert_array_equal(out_keyboard, keyboard[source_indices])
        np.testing.assert_allclose(out_camera.sum(axis=0), camera.sum(axis=0))
        np.testing.assert_array_equal(out_camera[0], camera[0] + camera[1])

    def test_quantizer_edges_produce_bounded_tokens(self) -> None:
        quantizer = {
            "edges": {
                "yaw": [-1.0, 0.0, 1.0],
                "pitch": [-2.0, 0.0, 2.0],
            }
        }
        camera = np.asarray([[-3, -3], [0, 0], [3, 3]], dtype=np.float32)
        tokens = quantize_camera(camera, quantizer)
        np.testing.assert_array_equal(tokens, [[0, 0], [2, 2], [3, 3]])


class FrozenProtocolTests(unittest.TestCase):
    def test_frozen_manifest_regenerates_byte_identically(self) -> None:
        expected = (ROOT / "manifests/protocol.json").read_bytes()
        actual = canonical_json_bytes(build_protocol(ROOT / "config.json"))
        self.assertEqual(actual, expected)

    def test_protocol_is_episode_disjoint_and_action_only(self) -> None:
        protocol = load_json(ROOT / "manifests/protocol.json")
        train = [entry for entry in protocol["episodes"] if entry["split"] == "train"]
        test = [entry for entry in protocol["episodes"] if entry["split"] == "test"]
        self.assertEqual(len(train), 512)
        self.assertEqual(len(test), 85)
        self.assertEqual(len({entry["episode_id"] for entry in train}), 512)
        self.assertEqual(len({entry["episode_id"] for entry in test}), 85)
        self.assertTrue(all(entry["source_url"].endswith(".jsonl") for entry in train + test))
        self.assertTrue(all(".mp4" not in entry["source_url"] for entry in train + test))
        for source in protocol["source_metadata"]["episode_files"]:
            self.assertEqual(sha256_file(Path(source["path"])), source["sha256"])

    def test_compact_manifests_form_a_complete_hash_chain(self) -> None:
        report = validate_all(ROOT, ROOT / "manifests/protocol.json", deep=False)
        self.assertEqual(report["downloaded_files"], 597)
        self.assertEqual(report["converted_files"], 597)
        self.assertEqual(report["quantizer_fit_episodes"], 512)
        self.assertEqual(report["block_files"], 597)


class DownloaderGuardTests(unittest.TestCase):
    def test_only_public_https_jsonl_is_accepted(self) -> None:
        _assert_action_url(
            "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/x.jsonl"
        )
        for url in (
            "http://openaipublic.blob.core.windows.net/minecraft-rl/x.jsonl",
            "https://example.com/x.jsonl",
            "https://openaipublic.blob.core.windows.net/minecraft-rl/x.mp4",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _assert_action_url(url)

    def test_fixture_jsonl_validation(self) -> None:
        self.assertEqual(_count_valid_jsonl(VPT_FIXTURES / "attack_stuck.jsonl"), 4)
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.jsonl"
            broken.write_text("{}\nnot-json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _count_valid_jsonl(broken)

    def test_windows_1252_source_fallback_matches_solaris(self) -> None:
        record = json.loads((VPT_FIXTURES / "all_off.jsonl").read_text().splitlines()[0])
        record["stats"] = {"label": "caf\u00e9"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp1252.jsonl"
            path.write_bytes(
                (json.dumps(record, ensure_ascii=False) + "\n").encode("windows-1252")
            )
            self.assertEqual(_count_valid_jsonl(path), 1)
            self.assertEqual(read_jsonl(path)[0]["stats"]["label"], "caf\u00e9")


if __name__ == "__main__":
    unittest.main()

