"""Explicitly approximate one-intent-step plus three-actual-step refinement."""

from __future__ import annotations

import time
from typing import Any, Mapping

import torch

if __package__:
    from .core import (
        ExactProposalTransaction,
        ProposalState,
        TransactionResult,
        _add_noise,
        _commit,
        _evaluate,
        _write_output,
    )
else:
    from core import (  # type: ignore[no-redef]
        ExactProposalTransaction,
        ProposalState,
        TransactionResult,
        _add_noise,
        _commit,
        _evaluate,
        _write_output,
    )


def accept_intent_refine(
    transaction: ExactProposalTransaction,
    state: ProposalState,
    actual_x0_fn: Any,
    actual_action_tensors: Mapping[str, torch.Tensor],
) -> TransactionResult:
    """Consume an intent proposal using exact observed conditioning for steps 1–3.

    This is not exact acceptance. The retained x_s1 came from the train-derived
    intent representative, so callers must pass the separate semantic gate.
    """

    actual_hash = transaction._validate(
        state, actual_action_tensors, exact_action=False
    )
    if actual_hash == state.exact_action_sha256:
        raise ValueError("intent refinement requires a nonidentical exact action")
    session = transaction.session
    session.restore_delta(state.parent_snapshot)
    steps = tuple(state.runtime_schedule_identity["steps"])
    noisy = state.retained_next_noisy_latent
    denoise_started = time.perf_counter()
    denoised = None
    for index in range(1, len(steps)):
        denoised = _evaluate(session, actual_x0_fn, noisy, steps[index])
        if index < len(steps) - 1:
            noisy = _add_noise(
                session,
                denoised,
                state.deterministic_transition_noise[index],
                steps[index + 1],
            )
    assert denoised is not None
    denoise_ms = (time.perf_counter() - denoise_started) * 1000.0
    _write_output(session, denoised)
    commit_started = time.perf_counter()
    _commit(session, actual_x0_fn, denoised, state.deterministic_context_noise)
    commit_ms = (time.perf_counter() - commit_started) * 1000.0
    block = int(session.block_index)
    session.block_index += 1
    state.consumed = True
    transaction.active = None
    return TransactionResult(
        block_index=block,
        latent=denoised.detach(),
        denoise_host_ms=denoise_ms,
        commit_host_ms=commit_ms,
        path="approx_intent_1_plus_actual_3",
        action_sha256=actual_hash,
    )
