"""Frozen trace evaluation for action-block speculation.

The evaluator fits every statistic on whole train episodes, scores the 85
frozen test episodes, and writes only hash-bound, path-relative artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from common import (
    ROOT,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
)

BLOCK_FRAMES = 12
BUDGETS = (1, 2, 4)
PREDICTORS = ("repeat_last", "global_frequency", "markov_1", "markov_history_3")
TARGETS = ("strict_hash", "intent")
SUBSETS = ("all", "intent_change")
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_DRAWS = 10_000
HISTORY_ORDER = 3
INTENT_QUANTILE = 0.50
RESULT_SCHEMA_KEYS = {
    "hit_rate",
    "miss_rate",
    "readiness_rate",
    "compute_candidates",
    "compute_ms",
    "extra_state_bytes",
}


@dataclass(frozen=True)
class Episode:
    split: str
    episode_id: int
    source_sha256: str
    keyboard: np.ndarray
    keyboard_token: np.ndarray
    camera_degrees: np.ndarray


@dataclass(frozen=True)
class TargetEpisode:
    split: str
    episode_id: int
    strict_hash: tuple[str, ...]
    intent: tuple[str, ...]


def _relative_label(path: Path) -> str:
    """Return a stable path label without retaining an absolute path."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _atomic_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        figure.savefig(temporary_path, dpi=220, bbox_inches="tight")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_token_bytes(token: str) -> bytes:
    return token.encode("ascii")


def canonical_strict_hash(
    keyboard_token: np.ndarray, camera_degrees: np.ndarray
) -> str:
    """Hash one exact canonical 12-frame action sequence.

    The payload is version tag, 12 little-endian uint32 packed keyboard
    states, then 12x2 little-endian float32 camera deltas in frame/axis order.
    """

    keyboard_token = np.asarray(keyboard_token)
    camera_degrees = np.asarray(camera_degrees)
    if keyboard_token.shape != (BLOCK_FRAMES,):
        raise ValueError("keyboard_token block must have shape [12]")
    if camera_degrees.shape != (BLOCK_FRAMES, 2):
        raise ValueError("camera_degrees block must have shape [12,2]")
    if not np.all(np.isfinite(camera_degrees)):
        raise ValueError("camera block contains non-finite values")
    payload = (
        b"rtwm.strict-action-block.v1\0"
        + keyboard_token.astype("<u4", copy=False).tobytes(order="C")
        + camera_degrees.astype("<f4", copy=False).tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def dominant_keyboard_token(tokens: np.ndarray) -> int:
    """Return the modal exact 23-bit state; numeric minimum breaks ties."""

    values, counts = np.unique(np.asarray(tokens, dtype=np.uint32), return_counts=True)
    if values.size == 0:
        raise ValueError("cannot take a dominant token over an empty block")
    maximum = int(counts.max())
    return int(values[counts == maximum].min())


def _camera_class(total: float, threshold: float) -> int:
    if total < -threshold:
        return -1
    if total > threshold:
        return 1
    return 0


def encode_intent(keyboard_token: int, yaw_class: int, pitch_class: int) -> str:
    if not 0 <= keyboard_token < (1 << 23):
        raise ValueError("keyboard token is outside the canonical 23-bit range")
    if yaw_class not in (-1, 0, 1) or pitch_class not in (-1, 0, 1):
        raise ValueError("camera classes must be -1, 0, or 1")
    return f"{keyboard_token:06x}:{yaw_class + 1}:{pitch_class + 1}"


def decode_intent(token: str) -> tuple[int, int, int]:
    keyboard, yaw, pitch = token.split(":")
    return int(keyboard, 16), int(yaw) - 1, int(pitch) - 1


def _positive_abs_quantile(values: Sequence[float], quantile: float) -> float:
    positive = np.asarray([abs(value) for value in values if value != 0.0])
    if positive.size == 0:
        return 0.0
    return float(np.quantile(positive, quantile, method="linear"))


def fit_intent_thresholds(
    episodes: Sequence[Episode], quantile: float = INTENT_QUANTILE
) -> dict[str, Any]:
    """Fit camera dead zones and branch representatives on train only."""

    if not episodes or any(episode.split != "train" for episode in episodes):
        raise ValueError("intent fitting accepts non-empty train episodes only")
    yaw_totals: list[float] = []
    pitch_totals: list[float] = []
    for episode in episodes:
        totals = episode.camera_degrees.astype(np.float64).sum(axis=1)
        yaw_totals.extend(totals[:, 0].tolist())
        pitch_totals.extend(totals[:, 1].tolist())
    yaw_threshold = _positive_abs_quantile(yaw_totals, quantile)
    pitch_threshold = _positive_abs_quantile(pitch_totals, quantile)

    def representative(values: Sequence[float], threshold: float, sign: int) -> float:
        selected = [
            abs(value)
            for value in values
            if _camera_class(float(value), threshold) == sign
        ]
        if not selected:
            return threshold
        return float(np.median(np.asarray(selected, dtype=np.float64)))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "fit_split": "train",
        "fit_episode_count": len(episodes),
        "fit_block_count": sum(episode.keyboard_token.shape[0] for episode in episodes),
        "algorithm": {
            "keyboard": (
                "modal exact packed 23-bit state over 12 frames; "
                "smallest uint32 wins count ties"
            ),
            "camera_total": (
                "float64 sum of canonical float32 per-frame degrees over each "
                "12-frame block, axis order yaw,pitch"
            ),
            "threshold": (
                "per axis numpy.quantile(abs(nonzero train block totals), "
                f"{quantile}, method=linear)"
            ),
            "class": "-1 if total < -threshold; +1 if total > threshold; else 0",
            "representative": (
                "per axis/class median absolute train total; neutral is zero; "
                "distributed uniformly over 12 canonical camera frames"
            ),
        },
        "quantile": quantile,
        "threshold_degrees": {
            "yaw": yaw_threshold,
            "pitch": pitch_threshold,
        },
        "representative_total_degrees": {
            "yaw": {
                "negative": -representative(yaw_totals, yaw_threshold, -1),
                "neutral": 0.0,
                "positive": representative(yaw_totals, yaw_threshold, 1),
            },
            "pitch": {
                "negative": -representative(pitch_totals, pitch_threshold, -1),
                "neutral": 0.0,
                "positive": representative(pitch_totals, pitch_threshold, 1),
            },
        },
    }
    payload["fit_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def construct_targets(
    episodes: Sequence[Episode], thresholds: Mapping[str, Any]
) -> list[TargetEpisode]:
    yaw_threshold = float(thresholds["threshold_degrees"]["yaw"])
    pitch_threshold = float(thresholds["threshold_degrees"]["pitch"])
    output: list[TargetEpisode] = []
    for episode in episodes:
        strict: list[str] = []
        intent: list[str] = []
        block_count = episode.keyboard_token.shape[0]
        for block_index in range(block_count):
            strict.append(
                canonical_strict_hash(
                    episode.keyboard_token[block_index],
                    episode.camera_degrees[block_index],
                )
            )
            totals = episode.camera_degrees[block_index].astype(np.float64).sum(axis=0)
            intent.append(
                encode_intent(
                    dominant_keyboard_token(episode.keyboard_token[block_index]),
                    _camera_class(float(totals[0]), yaw_threshold),
                    _camera_class(float(totals[1]), pitch_threshold),
                )
            )
        output.append(
            TargetEpisode(
                split=episode.split,
                episode_id=episode.episode_id,
                strict_hash=tuple(strict),
                intent=tuple(intent),
            )
        )
    return output


def rank_counts(counts: Mapping[str, int]) -> list[str]:
    """Rank descending count, then ascending canonical token bytes."""

    return sorted(
        counts,
        key=lambda token: (-int(counts[token]), _canonical_token_bytes(token)),
    )


def _append_unique(output: list[str], candidates: Iterable[str], budget: int) -> None:
    seen = set(output)
    for candidate in candidates:
        if candidate not in seen:
            output.append(candidate)
            seen.add(candidate)
            if len(output) >= budget:
                return


class PredictorSet:
    """Train-only count models with deterministic backoff and top-k ties."""

    def __init__(self, train: Sequence[TargetEpisode], target: str):
        if not train or any(episode.split != "train" for episode in train):
            raise ValueError("predictor fitting accepts non-empty train episodes only")
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}")
        self.target = target
        self.global_counts: Counter[str] = Counter()
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.fit_episode_ids = tuple(sorted(episode.episode_id for episode in train))
        for episode in train:
            sequence = getattr(episode, target)
            self.global_counts.update(sequence)
            for index, value in enumerate(sequence):
                for order in range(1, min(HISTORY_ORDER, index) + 1):
                    context = tuple(sequence[index - order : index])
                    self.transitions[context][value] += 1
        self.global_rank = rank_counts(self.global_counts)

    def predict(
        self, predictor: str, history: Sequence[str], budget: int
    ) -> list[str]:
        if budget <= 0:
            return []
        if not history:
            return self.global_rank[:budget]
        if predictor == "repeat_last":
            return [history[-1]]
        if predictor == "global_frequency":
            return self.global_rank[:budget]
        if predictor not in ("markov_1", "markov_history_3"):
            raise ValueError(f"unknown predictor: {predictor}")
        maximum_order = 1 if predictor == "markov_1" else HISTORY_ORDER
        output: list[str] = []
        for order in range(min(maximum_order, len(history)), 0, -1):
            counts = self.transitions.get(tuple(history[-order:]))
            if counts:
                _append_unique(output, rank_counts(counts), budget)
                if len(output) >= budget:
                    return output
        _append_unique(output, self.global_rank, budget)
        return output


def episode_bootstrap_interval(
    successes: np.ndarray,
    denominators: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Cluster bootstrap whole episodes and return percentile 95% bounds."""

    successes = np.asarray(successes, dtype=np.int64)
    denominators = np.asarray(denominators, dtype=np.int64)
    if successes.shape != denominators.shape or successes.ndim != 1:
        raise ValueError("bootstrap inputs must be equal-length vectors")
    if denominators.sum() == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rates = np.empty(draws, dtype=np.float64)
    episode_count = len(successes)
    for start in range(0, draws, 512):
        count = min(512, draws - start)
        sampled = rng.integers(0, episode_count, size=(count, episode_count))
        numerator = successes[sampled].sum(axis=1)
        denominator = denominators[sampled].sum(axis=1)
        rates[start : start + count] = np.divide(
            numerator,
            denominator,
            out=np.full(count, np.nan),
            where=denominator != 0,
        )
    finite = rates[np.isfinite(rates)]
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def evaluate_models(
    train_targets: Sequence[TargetEpisode],
    test_targets: Sequence[TargetEpisode],
) -> tuple[list[dict[str, Any]], dict[str, PredictorSet]]:
    if len(test_targets) != 85 or any(episode.split != "test" for episode in test_targets):
        raise ValueError("evaluation requires exactly the frozen 85 test episodes")
    models = {target: PredictorSet(train_targets, target) for target in TARGETS}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        model = models[target]
        for predictor in PREDICTORS:
            for budget in BUDGETS:
                episode_success = {
                    subset: np.zeros(len(test_targets), dtype=np.int64)
                    for subset in SUBSETS
                }
                episode_total = {
                    subset: np.zeros(len(test_targets), dtype=np.int64)
                    for subset in SUBSETS
                }
                candidate_total = {subset: 0 for subset in SUBSETS}
                for episode_position, episode in enumerate(test_targets):
                    sequence = getattr(episode, target)
                    for index in range(1, len(sequence)):
                        predictions = model.predict(predictor, sequence[:index], budget)
                        changed = episode.intent[index] != episode.intent[index - 1]
                        subsets = ("all", "intent_change") if changed else ("all",)
                        for subset in subsets:
                            episode_total[subset][episode_position] += 1
                            episode_success[subset][episode_position] += int(
                                sequence[index] in predictions
                            )
                            candidate_total[subset] += len(predictions)
                for subset in SUBSETS:
                    successes = int(episode_success[subset].sum())
                    eligible = int(episode_total[subset].sum())
                    low, high = episode_bootstrap_interval(
                        episode_success[subset],
                        episode_total[subset],
                        seed=BOOTSTRAP_SEED,
                    )
                    rows.append(
                        {
                            "target": target,
                            "predictor": predictor,
                            "budget": budget,
                            "subset": subset,
                            "test_episode_count": len(test_targets),
                            "eligible_episode_count": int(
                                np.count_nonzero(episode_total[subset])
                            ),
                            "eligible_block_count": eligible,
                            "hit_count": successes,
                            "miss_count": eligible - successes,
                            "hit_rate": successes / eligible,
                            "miss_rate": 1.0 - successes / eligible,
                            "hit_rate_ci95_low": low,
                            "hit_rate_ci95_high": high,
                            "compute_candidates": candidate_total[subset],
                            "mean_candidates_per_block": (
                                candidate_total[subset] / eligible
                            ),
                        }
                    )
    return rows, models


def simulate_readiness(
    candidate_times_ms: Sequence[float],
    *,
    capacity: int,
    lead_window_ms: float = 750.0,
) -> list[float]:
    """Greedy-worker projected completion times for serial measurements."""

    if capacity not in (1, 2, 4):
        raise ValueError("capacity must be 1, 2, or 4")
    if lead_window_ms != 750.0:
        raise ValueError("the frozen action lead window is 750 ms")
    worker_available = [0.0] * capacity
    completion: list[float] = []
    for duration in candidate_times_ms:
        if duration < 0:
            raise ValueError("candidate time cannot be negative")
        worker = min(range(capacity), key=lambda index: (worker_available[index], index))
        worker_available[worker] += float(duration)
        completion.append(worker_available[worker])
    return completion


def charge_serial_work(
    generation_ms: Sequence[float],
    *,
    fork_ms: float = 0.0,
    restore_ms: Sequence[float] = (),
    capture_ms: Sequence[float] = (),
    accept_ms: float = 0.0,
) -> float:
    """Charge all generated and abandoned work in measured serial mode."""

    values = [
        *generation_ms,
        fork_ms,
        *restore_ms,
        *capture_ms,
        accept_ms,
    ]
    if any(value < 0 for value in values):
        raise ValueError("timing charges cannot be negative")
    return float(sum(values))


def assert_no_action_latency_schema(value: Any) -> None:
    """Reject forbidden latency claims and unknown trace-wide metric fields."""

    forbidden = ("caol", "action_to_output", "hidden_latency", "realized_latency")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in forbidden):
                raise ValueError(f"forbidden metric key: {key}")
            assert_no_action_latency_schema(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_action_latency_schema(child)


def materialize_intent(
    token: str, thresholds: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    keyboard_token, yaw_class, pitch_class = decode_intent(token)
    bits = (
        (np.uint32(keyboard_token) >> np.arange(23, dtype=np.uint32))
        & np.uint32(1)
    ).astype(np.float32)
    keyboard = np.repeat(bits[None, :], BLOCK_FRAMES, axis=0)
    names = {-1: "negative", 0: "neutral", 1: "positive"}
    representative = thresholds["representative_total_degrees"]
    totals = np.asarray(
        [
            representative["yaw"][names[yaw_class]],
            representative["pitch"][names[pitch_class]],
        ],
        dtype=np.float32,
    )
    camera = np.repeat((totals / BLOCK_FRAMES)[None, :], BLOCK_FRAMES, axis=0)
    return keyboard, camera


def _load_episodes(block_manifest: Mapping[str, Any]) -> list[Episode]:
    episodes: list[Episode] = []
    for entry in block_manifest["files"]:
        source_path = Path(entry["local_path"])
        if sha256_file(source_path) != entry["sha256"]:
            raise ValueError(
                f"block artifact hash mismatch for {entry['split']}/{entry['episode_id']}"
            )
        with np.load(source_path, allow_pickle=False) as arrays:
            keyboard = arrays["keyboard"].copy()
            keyboard_token = arrays["keyboard_token"].copy()
            camera = arrays["camera_degrees"].copy()
        if keyboard.shape != (entry["block_count"], BLOCK_FRAMES, 23):
            raise ValueError("canonical keyboard shape mismatch")
        if keyboard_token.shape != (entry["block_count"], BLOCK_FRAMES):
            raise ValueError("canonical keyboard-token shape mismatch")
        if camera.shape != (entry["block_count"], BLOCK_FRAMES, 2):
            raise ValueError("canonical camera shape mismatch")
        episodes.append(
            Episode(
                split=entry["split"],
                episode_id=int(entry["episode_id"]),
                source_sha256=entry["sha256"],
                keyboard=keyboard,
                keyboard_token=keyboard_token,
                camera_degrees=camera,
            )
        )
    return episodes


def _select_replay_decisions(
    episodes: Sequence[Episode],
    targets: Sequence[TargetEpisode],
    model: PredictorSet,
    protocol_sha256: str,
) -> dict[str, Any]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    candidates: dict[str, list[tuple[str, TargetEpisode, int]]] = {
        "stable_common": [],
        "intent_change": [],
    }
    for target_episode in targets:
        for index in range(1, len(target_episode.intent)):
            predictions = model.predict(
                "markov_history_3", target_episode.intent[:index], 4
            )
            if len(predictions) < 4:
                continue
            category = (
                "intent_change"
                if target_episode.intent[index] != target_episode.intent[index - 1]
                else "stable_common"
            )
            rank = hashlib.sha256(
                (
                    protocol_sha256
                    + "\0replay-v1\0"
                    + category
                    + "\0"
                    + str(target_episode.episode_id)
                    + "\0"
                    + str(index)
                ).encode("ascii")
            ).hexdigest()
            candidates[category].append((rank, target_episode, index))
    selected: list[dict[str, Any]] = []
    for category in ("stable_common", "intent_change"):
        if not candidates[category]:
            raise ValueError(f"no eligible replay decision for {category}")
        _, target_episode, index = min(candidates[category], key=lambda item: item[0])
        episode = episode_by_id[target_episode.episode_id]
        predictions = model.predict(
            "markov_history_3", target_episode.intent[:index], 4
        )
        selected.append(
            {
                "category": category,
                "split": "test",
                "episode_id": target_episode.episode_id,
                "block_index": index,
                "source_block_file_sha256": episode.source_sha256,
                "strict_hash": target_episode.strict_hash[index],
                "previous_intent": target_episode.intent[index - 1],
                "actual_intent": target_episode.intent[index],
                "candidate_intents_ranked": predictions,
                "actual_keyboard": episode.keyboard[index].astype(int).tolist(),
                "actual_camera_degrees": (
                    episode.camera_degrees[index].astype(float).tolist()
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "selection_algorithm": (
            "minimum SHA256(protocol_sha256,NUL,replay-v1,NUL,category,NUL,"
            "episode_id,NUL,block_index), requiring four history-Markov intents"
        ),
        "predictor": "markov_history_3",
        "target": "intent",
        "decisions": selected,
    }
    payload["selection_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _plot_results(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True, sharey=True)
    colors = {
        "repeat_last": "#4C78A8",
        "global_frequency": "#F58518",
        "markov_1": "#54A24B",
        "markov_history_3": "#B279A2",
    }
    for row_index, target in enumerate(TARGETS):
        for column_index, subset in enumerate(SUBSETS):
            axis = axes[row_index, column_index]
            for predictor in PREDICTORS:
                selected = [
                    row
                    for row in rows
                    if row["target"] == target
                    and row["subset"] == subset
                    and row["predictor"] == predictor
                ]
                selected.sort(key=lambda row: int(row["budget"]))
                x = [int(row["budget"]) for row in selected]
                y = [float(row["hit_rate"]) for row in selected]
                low = [float(row["hit_rate_ci95_low"]) for row in selected]
                high = [float(row["hit_rate_ci95_high"]) for row in selected]
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    label=predictor.replace("_", " "),
                    color=colors[predictor],
                )
                axis.fill_between(x, low, high, color=colors[predictor], alpha=0.12)
            axis.set_title(
                f"{target.replace('_', ' ')} — {subset.replace('_', ' ')}"
            )
            axis.grid(alpha=0.25)
            axis.set_xticks(BUDGETS)
            axis.set_ylim(0, 1)
            if row_index == 1:
                axis.set_xlabel("Generated branch budget B")
            if column_index == 0:
                axis.set_ylabel("Top-B hit rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Frozen test action-block speculation frontier", y=1.01)
    figure.subplots_adjust(bottom=0.14, hspace=0.28, wspace=0.18)
    _atomic_figure(path, figure)
    plt.close(figure)


def _run_identity(block_manifest_path: Path) -> dict[str, Any]:
    source_files = ("speculation_frontier.py", "common.py")
    identity: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(ROOT / "manifests" / "protocol.json"),
        "blocks_manifest_sha256": sha256_file(block_manifest_path),
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in source_files
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "frozen_parameters": {
            "block_frames": BLOCK_FRAMES,
            "budgets": list(BUDGETS),
            "history_order": HISTORY_ORDER,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "intent_threshold_quantile": INTENT_QUANTILE,
        },
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    return identity


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": _relative_label(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def run_evaluation(
    *,
    block_manifest_path: Path,
    output_dir: Path,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "trace_summary.json"
    identity = _run_identity(block_manifest_path)
    if summary_path.exists():
        existing = load_json(summary_path)
        if resume:
            if existing["identity_sha256"] != identity["identity_sha256"]:
                raise RuntimeError("strict resume identity mismatch")
            for artifact in existing["artifacts"]:
                path = ROOT / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError("strict resume artifact hash mismatch")
            return existing
        if not force:
            raise FileExistsError("results exist; pass --resume or --force")

    block_manifest = load_json(block_manifest_path)
    if block_manifest["block_frames"] != BLOCK_FRAMES:
        raise ValueError("frozen block size changed")
    episodes = _load_episodes(block_manifest)
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    if len(train) != 512 or len(test) != 85:
        raise ValueError("frozen split must contain 512 train and 85 test episodes")

    thresholds = fit_intent_thresholds(train)
    threshold_path = output_dir / "intent_fit.json"
    atomic_write_json(threshold_path, thresholds)
    # Test targets are not constructed until the complete train-fit artifact is
    # materialized. This ordering is part of the frozen leakage guard.
    train_targets = construct_targets(train, thresholds)
    test_targets = construct_targets(test, thresholds)
    rows, models = evaluate_models(train_targets, test_targets)

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append(
            {
                key: (
                    f"{value:.10f}"
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
        )
    metrics_path = output_dir / "trace_metrics.csv"
    _atomic_csv(metrics_path, csv_rows)

    selection = _select_replay_decisions(
        test,
        test_targets,
        models["intent"],
        identity["protocol_sha256"],
    )
    selection_path = output_dir / "replay_selection.json"
    atomic_write_json(selection_path, selection)

    figure_path = output_dir / "speculation_frontier.png"
    _plot_results(rows, figure_path)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "identity_sha256": identity["identity_sha256"],
        "identity": identity,
        "target_definitions": {
            "strict_hash": (
                "SHA256 over version tag, exact 12 packed uint32 keyboard states, "
                "and exact 12x2 canonical float32 camera degrees"
            ),
            "intent": (
                "dominant exact 23-bit keyboard state plus independently ternary "
                "yaw/pitch classes from train-fitted block-total dead zones"
            ),
        },
        "fit_policy": (
            "thresholds, global frequencies, and all Markov transitions use only "
            "the 512 train episodes; frozen test targets are constructed afterward"
        ),
        "eligibility": (
            "block index >=1 within each test episode; intent-change subset requires "
            "current intent token != immediately previous episode-local intent token"
        ),
        "ranking_and_duplicates": (
            "descending train count; canonical ASCII token ascending breaks count "
            "ties; deterministic longest-context backoff; duplicate candidates "
            "removed; shorter-context/global ranks fill unused budget"
        ),
        "timing_scope": (
            "this trace artifact reports semantic hits/misses and generated-candidate "
            "compute only; measured serial systems replay and worker-capacity "
            "projections are separate artifacts"
        ),
        "test_episode_count": 85,
        "test_eligible_block_count": next(
            row["eligible_block_count"]
            for row in rows
            if row["target"] == "intent"
            and row["predictor"] == "repeat_last"
            and row["budget"] == 1
            and row["subset"] == "all"
        ),
        "test_intent_change_block_count": next(
            row["eligible_block_count"]
            for row in rows
            if row["target"] == "intent"
            and row["predictor"] == "repeat_last"
            and row["budget"] == 1
            and row["subset"] == "intent_change"
        ),
        "bootstrap": {
            "unit": "whole test episode",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "artifacts": [],
    }
    assert_no_action_latency_schema(summary)
    artifacts = [
        _artifact_entry(threshold_path),
        _artifact_entry(metrics_path),
        _artifact_entry(selection_path),
        _artifact_entry(figure_path),
    ]
    summary["artifacts"] = artifacts
    atomic_write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks-manifest",
        type=Path,
        default=ROOT / "manifests" / "blocks.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    summary = run_evaluation(
        block_manifest_path=args.blocks_manifest,
        output_dir=args.output_dir,
        resume=args.resume,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "identity_sha256": summary["identity_sha256"],
                "test_episode_count": summary["test_episode_count"],
                "test_eligible_block_count": summary["test_eligible_block_count"],
                "test_intent_change_block_count": (
                    summary["test_intent_change_block_count"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
