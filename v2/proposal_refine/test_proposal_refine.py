from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from einops import rearrange

from .conditioning import make_shallow_conditioning_x0
from .core import (
    ExactProposalTransaction,
    exact_action_sha256,
    parent_identity,
)
from .intent_refine import accept_intent_refine
from .intent_semantic_benchmark import _raw_metrics
from .freeze_intent_confidence import measured_costs, wilson_lower
from .trace_replay import _latency_model, percentile
from .pre_roll_confidence import (
    _costs,
    _rates,
    fit_counts,
    predict_with_confidence,
)


@dataclass
class FakeConfig:
    independent_denoising_step_list: bool = False
    context_noise: int = 0


class FakeScheduler:
    def add_noise(self, value, noise, timestep):
        scale = timestep.to(value.dtype).reshape(-1, 1, 1, 1) / 1000.0
        return value + noise * scale


class FakeModel:
    def __init__(self):
        self.denoising_step_list = (1000, 750, 500, 250)
        self.config = FakeConfig()
        self.scheduler = FakeScheduler()
        self.crossattn_cache = None
        self.kv_cache1 = [
            {
                "k_players": torch.zeros((1, 2, 32, 1, 1)),
                "v_players": torch.zeros((1, 2, 32, 1, 1)),
                "k_z": torch.zeros((1, 32, 1, 1)),
                "v_z": torch.zeros((1, 32, 1, 1)),
                "global_end_index": torch.tensor([0]),
                "local_end_index": torch.tensor([0]),
                "z_local_end_index": torch.tensor([0]),
            }
        ]


class FakeSession:
    def __init__(self):
        self.model = FakeModel()
        self.block_index = 0
        self.nfpb = 1
        self.n_views = 2
        self.frames_per_tblock = 2
        self.token_stride = 2
        self.noise = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)
        self.output = torch.zeros_like(self.noise)

    def fork_delta(self):
        entry = self.model.kv_cache1[0]
        return {
            "block_index": self.block_index,
            "kv": {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in entry.items()
            },
        }

    def restore_delta(self, snapshot):
        entry = self.model.kv_cache1[0]
        for key, value in snapshot["kv"].items():
            entry[key].copy_(value)
        self.block_index = snapshot["block_index"]


def make_action(value: float) -> dict[str, torch.Tensor]:
    return {
        "keyboard": torch.full((1, 12, 23), value),
        "camera": torch.full((1, 12, 2), -value),
    }


def make_x0(action_value: float):
    def x0(noisy, timestep, *, kv_cache, current_end, **_):
        entry = kv_cache[0]
        slot = current_end // 2 - 1
        marker = noisy.mean() + timestep.float().mean() / 1000 + action_value
        entry["k_players"][:, :, slot].fill_(marker)
        entry["v_players"][:, :, slot].fill_(marker + 1)
        entry["k_z"][:, slot].fill_(marker + 2)
        entry["v_z"][:, slot].fill_(marker + 3)
        entry["global_end_index"].fill_(current_end)
        entry["local_end_index"].fill_(slot + 1)
        entry["z_local_end_index"].fill_(slot + 1)
        return noisy * 0.25 + marker

    return x0


def live_state(session):
    entry = session.model.kv_cache1[0]
    return {
        key: value.clone()
        for key, value in entry.items()
    }, session.output.clone(), session.block_index


def assert_live_equal(first, second):
    first_kv, first_output, first_block = first
    second_kv, second_output, second_block = second
    assert first_block == second_block
    assert torch.equal(first_output, second_output)
    assert first_kv.keys() == second_kv.keys()
    for key in first_kv:
        assert torch.equal(first_kv[key], second_kv[key]), key


def test_action_hash_includes_exact_bytes_dtype_and_shape():
    action = make_action(1.0)
    original = exact_action_sha256(action)
    changed = {key: value.clone() for key, value in action.items()}
    changed["camera"][0, 11, 1] += torch.finfo(torch.float32).eps
    assert exact_action_sha256(changed) != original
    changed = {key: value.double() for key, value in action.items()}
    assert exact_action_sha256(changed) != original


def test_proposal_is_uncommitted_and_exact_hit_matches_full():
    session = FakeSession()
    action = make_action(1.0)
    x0 = make_x0(1.0)
    source = {"core.py": "a" * 64}
    transaction = ExactProposalTransaction(
        session, logical_session_id="session-7", source_hashes=source
    )
    parent = session.fork_delta()
    parent_id = parent_identity(session, "session-7")
    output_object = session.output

    proposal = transaction.propose(x0, action)
    assert session.block_index == 0
    assert session.output is output_object
    assert parent_identity(session, "session-7") == parent_id
    assert proposal.parent_identity == parent_id
    assert proposal.first_runtime_timestep == 1000
    assert proposal.source_hashes == source
    assert len(proposal.deterministic_transition_noise) == 3
    assert len(proposal.noise_sha256) == 4

    transaction.discard(proposal)
    reference = transaction.run_full(x0, action)
    reference_state = live_state(session)
    session.restore_delta(parent)

    proposal = transaction.propose(x0, action)
    accepted = transaction.accept_exact(proposal, x0, action)
    accepted_state = live_state(session)
    assert accepted.path == "exact_hit_1_plus_3"
    assert torch.equal(reference.latent, accepted.latent)
    assert_live_equal(reference_state, accepted_state)


def test_intent_refine_is_approximate_and_commits_actual_action():
    actual = make_action(2.0)
    representative = make_action(1.0)
    session = FakeSession()
    before = live_state(session)
    transaction = ExactProposalTransaction(
        session, logical_session_id="intent-refine", source_hashes={}
    )
    proposal = transaction.propose(make_x0(1.0), representative)
    assert_live_equal(before, live_state(session))
    result = accept_intent_refine(
        transaction, proposal, make_x0(2.0), actual
    )

    reference_session = FakeSession()
    reference = ExactProposalTransaction(
        reference_session, logical_session_id="intent-refine", source_hashes={}
    ).run_full(make_x0(2.0), actual)
    assert result.path == "approx_intent_1_plus_actual_3"
    assert result.action_sha256 == exact_action_sha256(actual)
    assert not torch.equal(result.latent, reference.latent)
    assert session.block_index == 1
    assert proposal.consumed
    assert transaction.active is None


def test_raw_metrics_enforce_decoder_support_boundary():
    generator = torch.Generator().manual_seed(11)
    reference = torch.randint(
        0, 256, (2, 9, 8, 8, 3), generator=generator, dtype=torch.uint8
    ).numpy()
    candidate = reference.copy()
    candidate[:, 4:, 0, 0, 0] ^= 1
    metrics = _raw_metrics(reference, candidate, support_frame=3)

    assert metrics["before_support_exact"]
    assert metrics["before_support_changed_elements"] == 0
    assert all(view["frame_count"] == 6 for view in metrics["by_view"])
    assert all(view["psnr_db"] > 30.0 for view in metrics["by_view"])
    assert all(view["ssim_mean"] > 0.95 for view in metrics["by_view"])


def test_intent_cost_gate_charges_proposal_on_hits_and_misses():
    costs = measured_costs(
        {
            "cost_model": {
                "proposal_pre_action_mean_ms": 500.0,
                "baseline_action_to_host_mean_ms": 4250.0,
                "hit_action_to_host_mean_ms": 3750.0,
                "miss_action_to_host_mean_ms": 4250.0,
            }
        }
    )
    assert costs["gross_hit_action_saving_ms"] == 500.0
    assert costs["net_hit_saved_ms_after_charging_proposal"] == 0.0
    assert costs["miss_penalty_ms_after_charging_abandoned_proposal"] == 500.0
    assert costs["break_even_intent_precision"] == 1.0
    assert wilson_lower(100, 100) < 1.0


def test_miss_discards_work_and_runs_full_observed_action():
    session = FakeSession()
    proposed_action = make_action(1.0)
    observed_action = make_action(2.0)
    transaction = ExactProposalTransaction(
        session, logical_session_id="miss-session", source_hashes={}
    )
    parent = session.fork_delta()
    proposal = transaction.propose(make_x0(1.0), proposed_action)
    missed = transaction.miss(proposal, make_x0(2.0), observed_action)
    miss_state = live_state(session)

    session.restore_delta(parent)
    reference = ExactProposalTransaction(
        session, logical_session_id="miss-session", source_hashes={}
    ).run_full(make_x0(2.0), observed_action)
    assert missed.path == "miss_full_observed"
    assert torch.equal(missed.latent, reference.latent)
    assert_live_equal(miss_state, live_state(session))


def test_candidate_order_cannot_change_reserved_transition_noise():
    session = FakeSession()
    transaction = ExactProposalTransaction(
        session, logical_session_id="order-session", source_hashes={}
    )
    first = transaction.propose(make_x0(1.0), make_action(1.0))
    first_hashes = first.noise_sha256
    transaction.discard(first)
    second = transaction.propose(make_x0(9.0), make_action(9.0))
    assert second.noise_sha256 == first_hashes


def test_exact_accept_rejects_nonidentical_action():
    session = FakeSession()
    transaction = ExactProposalTransaction(
        session, logical_session_id="reject-session", source_hashes={}
    )
    proposal = transaction.propose(make_x0(1.0), make_action(1.0))
    with pytest.raises(ValueError, match="exactly match"):
        transaction.accept_exact(proposal, make_x0(2.0), make_action(2.0))


def test_trace_latency_model_charges_rank_and_separates_ready_events():
    result = _latency_model(
        [800.0, 900.0],
        refinement_ms=2400.0,
        commit_ms=700.0,
        full_denoise_ms=3200.0,
        decode_device_samples=[100.0],
        host_copy_samples=[10.0],
        budget=2,
    )
    assert len(result["proposal_ready_by_rank"]) == 2
    assert result["proposal_ready_by_rank"][0]["p95_within_750ms"] is False
    assert (
        result["proposal_ready_by_rank"][1]["latency"]["p50_ms"]
        > result["proposal_ready_by_rank"][0]["latency"]["p50_ms"]
    )
    assert (
        result["host_frame_ready_by_hit_rank"][0]["latency"]["p50_ms"]
        > result["final_commit_ready_by_hit_rank"][0]["latency"]["p50_ms"]
    )
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_exact_confidence_uses_longest_context_and_charges_misses():
    @dataclass(frozen=True)
    class Episode:
        strict_hash: tuple[str, ...]

    model = fit_counts(
        [
            Episode(("a", "b", "c", "d")),
            Episode(("a", "b", "c", "d")),
            Episode(("x", "b", "c", "e")),
        ],
        "strict_hash",
    )
    prediction = predict_with_confidence(model, ("a", "b", "c"))
    assert prediction["prediction"] == "d"
    assert prediction["context_order"] == 3
    assert prediction["confidence"] == 1.0

    rows = [
        {"confidence": 0.9, "hit": True, "pre_roll_eligible": True},
        {"confidence": 0.8, "hit": False, "pre_roll_eligible": True},
        {"confidence": 0.95, "hit": True, "pre_roll_eligible": False},
    ]
    rates = _rates(rows, 0.85, pre_roll_only=True)
    assert rates == {
        "eligible_count": 2,
        "selected_count": 1,
        "hit_count": 1,
        "miss_count": 0,
        "coverage": 0.5,
        "exact_hit_precision": 1.0,
    }
    costs = _costs(
        rates,
        {
            "proposal_pre_action_mean_ms": 500.0,
            "baseline_action_to_host_mean_ms": 5000.0,
            "hit_action_to_host_mean_ms": 4000.0,
            "miss_action_to_host_mean_ms": 5000.0,
        },
    )
    assert costs["charged_work_ms"] == 9500.0
    assert costs["charged_work_delta_ms"] == -500.0


def test_shallow_conditioning_wrapper_matches_stock_closure_exactly():
    class GeneratorModel:
        def generator(self, *, noisy_image_or_video, conditional_dict, **_):
            condition = conditional_dict["gt_frames"].permute(0, 2, 1, 3, 4)
            return None, noisy_image_or_video + condition

    self = GeneratorModel()
    n_views = 2
    conditional_dict = {
        "gt_frames": torch.arange(32, dtype=torch.float32).reshape(
            1, 1, 8, 2, 2
        ),
        "condition_video_input_mask_B_C_T_H_W": torch.ones(1, 1, 8, 2, 2),
        "view_indices_B_T": torch.arange(8).reshape(1, 8),
        "nested_immutable": {"token": torch.tensor([3.0])},
    }

    def stock_x0(noise_x, timestep, kv_cache=None, **kwargs):
        del timestep, kv_cache
        start = kwargs["start_frame_for_rope"]
        noise = noise_x.permute(0, 2, 1, 3, 4)
        noise_unfold = rearrange(
            noise, "b c (v t) h w -> b v c t h w", v=n_views
        )
        end = start + noise_unfold.shape[3]
        copied = conditional_dict.copy()
        gt = rearrange(
            conditional_dict["gt_frames"],
            "b c (v t) h w -> b v c t h w",
            v=n_views,
        )
        copied["gt_frames"] = rearrange(
            gt[:, :, :, start:end], "b v c t h w -> b c (v t) h w"
        )
        _, result = self.generator(
            noisy_image_or_video=noise.permute(0, 2, 1, 3, 4),
            conditional_dict=copied,
        )
        return result

    optimized, metadata = make_shallow_conditioning_x0(stock_x0)
    noisy = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    kwargs = {
        "start_frame_for_rope": 2,
        "current_start": 4,
        "current_end": 8,
        "crossattn_cache": None,
    }
    expected = stock_x0(noisy, torch.tensor([[1000, 1000]]), **kwargs)
    actual = optimized(noisy, torch.tensor([[1000, 1000]]), **kwargs)
    assert torch.equal(actual, expected)
    assert metadata["stock_deepcopy_removed"] is True
    assert conditional_dict["gt_frames"].shape == (1, 1, 8, 2, 2)
