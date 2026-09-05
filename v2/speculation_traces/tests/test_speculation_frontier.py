from __future__ import annotations

import unittest

import numpy as np

from speculation_frontier import (
    BLOCK_FRAMES,
    Episode,
    PredictorSet,
    TargetEpisode,
    assert_no_action_latency_schema,
    canonical_strict_hash,
    charge_serial_work,
    construct_targets,
    dominant_keyboard_token,
    encode_intent,
    episode_bootstrap_interval,
    evaluate_models,
    fit_intent_thresholds,
    rank_counts,
    simulate_readiness,
)


def make_episode(
    split: str,
    episode_id: int,
    block_tokens: list[list[int]],
    camera_totals: list[tuple[float, float]] | None = None,
) -> Episode:
    block_count = len(block_tokens)
    keyboard_token = np.asarray(block_tokens, dtype=np.uint32)
    keyboard = (
        (
            keyboard_token[..., None]
            >> np.arange(23, dtype=np.uint32)[None, None, :]
        )
        & 1
    ).astype(np.uint8)
    camera = np.zeros((block_count, BLOCK_FRAMES, 2), dtype=np.float32)
    for index, totals in enumerate(camera_totals or [(0.0, 0.0)] * block_count):
        camera[index] = np.asarray(totals, dtype=np.float32) / BLOCK_FRAMES
    return Episode(
        split=split,
        episode_id=episode_id,
        source_sha256=f"{episode_id:064x}",
        keyboard=keyboard,
        keyboard_token=keyboard_token,
        camera_degrees=camera,
    )


class SplitAndTargetTests(unittest.TestCase):
    def test_fit_rejects_test_episode(self) -> None:
        episode = make_episode("test", 1, [[0] * BLOCK_FRAMES])
        with self.assertRaises(ValueError):
            fit_intent_thresholds([episode])

    def test_strict_hash_is_exact_and_episode_local(self) -> None:
        keyboard = np.zeros(BLOCK_FRAMES, dtype=np.uint32)
        camera = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)
        original = canonical_strict_hash(keyboard, camera)
        camera[11, 1] = np.float32(1e-6)
        self.assertNotEqual(original, canonical_strict_hash(keyboard, camera))
        with self.assertRaises(ValueError):
            canonical_strict_hash(
                np.zeros(BLOCK_FRAMES + 1, dtype=np.uint32), camera
            )

    def test_dominant_keyboard_tie_uses_smallest_token(self) -> None:
        tokens = np.asarray([7, 9] * 6, dtype=np.uint32)
        self.assertEqual(dominant_keyboard_token(tokens), 7)

    def test_intent_thresholds_are_train_fit_and_applied(self) -> None:
        train = make_episode(
            "train",
            1,
            [[3] * BLOCK_FRAMES, [5] * BLOCK_FRAMES],
            [(2.0, 0.0), (4.0, -6.0)],
        )
        fit = fit_intent_thresholds([train])
        self.assertEqual(fit["fit_split"], "train")
        self.assertAlmostEqual(fit["threshold_degrees"]["yaw"], 3.0, places=6)
        test = make_episode(
            "test", 2, [[5] * BLOCK_FRAMES], [(4.0, -7.0)]
        )
        target = construct_targets([test], fit)[0]
        self.assertEqual(target.intent, (encode_intent(5, 1, -1),))


class RankingAndEvaluationTests(unittest.TestCase):
    def test_count_ties_and_duplicate_backoff_are_deterministic(self) -> None:
        self.assertEqual(rank_counts({"b": 2, "a": 2, "c": 1}), ["a", "b", "c"])
        train = [
            TargetEpisode(
                "train",
                1,
                ("x", "a", "b", "a"),
                ("x", "a", "b", "a"),
            )
        ]
        model = PredictorSet(train, "intent")
        candidates = model.predict("markov_history_3", ["x", "a", "b"], 4)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(candidates[0], "a")

    def test_predictor_fit_rejects_nontrain(self) -> None:
        episode = TargetEpisode("test", 1, ("x",), ("x",))
        with self.assertRaises(ValueError):
            PredictorSet([episode], "intent")

    def test_change_only_denominator_is_current_vs_previous_intent(self) -> None:
        train = [
            TargetEpisode(
                "train",
                1,
                ("s0", "s1", "s1"),
                ("i0", "i1", "i1"),
            )
        ]
        tests = [
            TargetEpisode(
                "test",
                index,
                ("s0", "s1", "s1"),
                ("i0", "i1", "i1"),
            )
            for index in range(85)
        ]
        rows, _ = evaluate_models(train, tests)
        row = next(
            item
            for item in rows
            if item["target"] == "strict_hash"
            and item["predictor"] == "repeat_last"
            and item["budget"] == 1
            and item["subset"] == "intent_change"
        )
        self.assertEqual(row["eligible_episode_count"], 85)
        self.assertEqual(row["eligible_block_count"], 85)


class StatisticsAndTimingTests(unittest.TestCase):
    def test_episode_bootstrap_is_deterministic(self) -> None:
        successes = np.asarray([0, 2, 3, 1])
        denominators = np.asarray([1, 2, 4, 3])
        first = episode_bootstrap_interval(
            successes, denominators, draws=500, seed=17
        )
        second = episode_bootstrap_interval(
            successes, denominators, draws=500, seed=17
        )
        self.assertEqual(first, second)

    def test_worker_projection_and_serial_charging(self) -> None:
        self.assertEqual(
            simulate_readiness([400.0, 400.0, 400.0, 400.0], capacity=2),
            [400.0, 400.0, 800.0, 800.0],
        )
        # A miss does not erase generated or abandoned branch work.
        charged = charge_serial_work(
            [400.0, 410.0, 420.0],
            fork_ms=5.0,
            restore_ms=[2.0, 2.0],
            capture_ms=[1.0, 1.0, 1.0],
            accept_ms=3.0,
        )
        self.assertEqual(charged, 1245.0)

    def test_output_schema_rejects_action_latency_claims(self) -> None:
        assert_no_action_latency_schema(
            {
                "hit_rate": 0.5,
                "miss_rate": 0.5,
                "readiness_rate": 0.0,
                "compute_ms": 10.0,
                "extra_state_bytes": 4,
            }
        )
        with self.assertRaises(ValueError):
            assert_no_action_latency_schema({"caol_ms": 1.0})


if __name__ == "__main__":
    unittest.main()
