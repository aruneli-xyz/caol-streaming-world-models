"""Exact one-plus-three proposal transactions for Gamma block sessions.

The transaction deliberately does not use the process-global CUDA RNG.  Every
diffusion transition is keyed by logical session, block, and transition, so
candidate ordering cannot change an accepted or fallback result.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from einops import rearrange


SCHEMA = "rtwm-v2-exact-proposal-transaction-1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def exact_action_sha256(action_tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact contiguous bytes."""

    digest = hashlib.sha256(b"rtwm.exact-action-tensors.v1\0")
    if not action_tensors:
        raise ValueError("at least one action tensor is required")
    for name in sorted(action_tensors):
        tensor = action_tensors[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"action {name!r} is not a tensor")
        value = tensor.detach().contiguous().cpu()
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(_canonical_bytes(metadata))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _steps_for_block(session: Any) -> tuple[int, ...]:
    raw = session.model.denoising_step_list
    if bool(getattr(session.model.config, "independent_denoising_step_list", False)):
        raw = raw[session.block_index]
    steps = tuple(int(value) for value in raw)
    if len(steps) != 4:
        raise ValueError(f"exact 1+3 requires four runtime steps, got {steps}")
    return steps


def schedule_identity(session: Any) -> dict[str, Any]:
    steps = _steps_for_block(session)
    payload = {
        "schema": "rtwm.runtime-schedule.v1",
        "block_index": int(session.block_index),
        "steps": list(steps),
        "context_noise": int(session.model.config.context_noise),
        "scheduler_type": (
            f"{type(session.model.scheduler).__module__}."
            f"{type(session.model.scheduler).__qualname__}"
        ),
    }
    payload["sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def parent_identity(session: Any, logical_session_id: str) -> dict[str, Any]:
    indices = []
    for layer, entry in enumerate(session.model.kv_cache1):
        indices.append(
            {
                "layer": layer,
                "global_end_index": int(entry["global_end_index"].item()),
                "local_end_index": int(entry["local_end_index"].item()),
                "z_local_end_index": (
                    int(entry["z_local_end_index"].item())
                    if "z_local_end_index" in entry
                    else None
                ),
            }
        )
    payload = {
        "schema": "rtwm.proposal-parent.v1",
        "logical_session_id": logical_session_id,
        "block_index": int(session.block_index),
        "indices": indices,
    }
    payload["sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _logical_seed(
    logical_session_id: str, block_index: int, transition: str
) -> int:
    payload = (
        f"{SCHEMA}\0{logical_session_id}\0{block_index}\0{transition}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def reserve_transition_noise(
    reference: torch.Tensor,
    *,
    logical_session_id: str,
    block_index: int,
    transition: str,
) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(_logical_seed(logical_session_id, block_index, transition))
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _initial_noisy(session: Any) -> torch.Tensor:
    begin = session.block_index * session.nfpb
    end = begin + session.nfpb
    noise_by_view = rearrange(
        session.noise, "b (v t) c h w -> b v t c h w", v=session.n_views
    )
    return rearrange(
        noise_by_view[:, :, begin:end], "b v t c h w -> b (v t) c h w"
    )


def _timestep(session: Any, value: int) -> torch.Tensor:
    return torch.full(
        (1, session.frames_per_tblock),
        value,
        device=session.noise.device,
        dtype=torch.int64,
    )


def _evaluate(
    session: Any, x0_fn: Any, noisy: torch.Tensor, timestep: int
) -> torch.Tensor:
    begin = session.block_index * session.nfpb
    end = begin + session.nfpb
    return x0_fn(
        noisy,
        _timestep(session, timestep),
        kv_cache=session.model.kv_cache1,
        crossattn_cache=session.model.crossattn_cache,
        current_start=begin * session.token_stride,
        current_end=end * session.token_stride,
        start_frame_for_rope=begin,
    )


def _add_noise(
    session: Any, denoised: torch.Tensor, noise: torch.Tensor, timestep: int
) -> torch.Tensor:
    return session.model.scheduler.add_noise(
        denoised.flatten(0, 1),
        noise.flatten(0, 1),
        torch.full(
            (session.frames_per_tblock,),
            timestep,
            device=denoised.device,
            dtype=torch.long,
        ),
    ).unflatten(0, denoised.shape[:2])


def _write_output(session: Any, denoised: torch.Tensor) -> None:
    begin = session.block_index * session.nfpb
    end = begin + session.nfpb
    output = rearrange(
        session.output, "b (v t) c h w -> b v t c h w", v=session.n_views
    )
    output[:, :, begin:end] = rearrange(
        denoised, "b (v t) c h w -> b v t c h w", v=session.n_views
    )
    session.output = rearrange(output, "b v t c h w -> b (v t) c h w")


def _commit(
    session: Any,
    x0_fn: Any,
    denoised: torch.Tensor,
    context_noise: torch.Tensor,
) -> None:
    context_timestep = int(session.model.config.context_noise)
    context = (
        _add_noise(session, denoised, context_noise, context_timestep)
        if context_timestep > 0
        else denoised
    )
    _evaluate(session, x0_fn, context, context_timestep)


@dataclass
class ProposalState:
    parent_identity: Mapping[str, Any]
    exact_action_sha256: str
    runtime_schedule_identity: Mapping[str, Any]
    first_runtime_timestep: int
    retained_next_noisy_latent: torch.Tensor
    deterministic_transition_noise: tuple[torch.Tensor, ...]
    deterministic_context_noise: torch.Tensor
    noise_sha256: tuple[str, ...]
    source_hashes: Mapping[str, str]
    parent_snapshot: Any = field(repr=False)
    proposal_host_ms: float = 0.0
    consumed: bool = False


@dataclass
class TransactionResult:
    block_index: int
    latent: torch.Tensor
    denoise_host_ms: float
    commit_host_ms: float
    path: str
    action_sha256: str


class ExactProposalTransaction:
    """One active exact proposal transaction for a blockwise session."""

    def __init__(
        self,
        session: Any,
        *,
        logical_session_id: str,
        source_hashes: Mapping[str, str],
    ):
        if not logical_session_id:
            raise ValueError("logical_session_id is required")
        self.session = session
        self.logical_session_id = logical_session_id
        self.source_hashes = dict(source_hashes)
        self.active: ProposalState | None = None

    def _reserve_noises(self, reference: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block = int(self.session.block_index)
        transitions = tuple(
            reserve_transition_noise(
                reference,
                logical_session_id=self.logical_session_id,
                block_index=block,
                transition=f"denoise-{index}-to-{index + 1}",
            )
            for index in range(3)
        )
        context = reserve_transition_noise(
            reference,
            logical_session_id=self.logical_session_id,
            block_index=block,
            transition="context-commit",
        )
        return transitions, context

    def propose(
        self, x0_fn: Any, action_tensors: Mapping[str, torch.Tensor]
    ) -> ProposalState:
        if self.active is not None:
            raise RuntimeError("a proposal is already active")
        session = self.session
        identity = parent_identity(session, self.logical_session_id)
        schedule = schedule_identity(session)
        steps = tuple(schedule["steps"])
        snapshot = session.fork_delta()
        output_before = session.output
        transition_noise, context_noise = self._reserve_noises(_initial_noisy(session))
        started = time.perf_counter()
        try:
            first = _evaluate(session, x0_fn, _initial_noisy(session), steps[0])
            next_noisy = _add_noise(
                session, first, transition_noise[0], steps[1]
            ).detach()
        finally:
            session.restore_delta(snapshot)
        if session.output is not output_before:
            raise RuntimeError("proposal mutated the decoder/output buffer")
        if parent_identity(session, self.logical_session_id) != identity:
            raise RuntimeError("proposal did not restore its parent boundary")
        state = ProposalState(
            parent_identity=identity,
            exact_action_sha256=exact_action_sha256(action_tensors),
            runtime_schedule_identity=schedule,
            first_runtime_timestep=steps[0],
            retained_next_noisy_latent=next_noisy,
            deterministic_transition_noise=transition_noise,
            deterministic_context_noise=context_noise,
            noise_sha256=tuple(
                tensor_sha256(value) for value in (*transition_noise, context_noise)
            ),
            source_hashes=self.source_hashes,
            parent_snapshot=snapshot,
            proposal_host_ms=(time.perf_counter() - started) * 1000.0,
        )
        self.active = state
        return state

    def _validate(
        self,
        state: ProposalState,
        action_tensors: Mapping[str, torch.Tensor],
        *,
        exact_action: bool,
    ) -> str:
        if state is not self.active or state.consumed:
            raise RuntimeError("proposal is not active")
        if parent_identity(self.session, self.logical_session_id) != state.parent_identity:
            raise RuntimeError("live session is not at proposal parent")
        observed = exact_action_sha256(action_tensors)
        if exact_action and observed != state.exact_action_sha256:
            raise ValueError("observed action does not exactly match proposal")
        return observed

    def accept_exact(
        self,
        state: ProposalState,
        x0_fn: Any,
        action_tensors: Mapping[str, torch.Tensor],
    ) -> TransactionResult:
        action_hash = self._validate(state, action_tensors, exact_action=True)
        session = self.session
        session.restore_delta(state.parent_snapshot)
        steps = tuple(state.runtime_schedule_identity["steps"])
        noisy = state.retained_next_noisy_latent
        denoise_started = time.perf_counter()
        denoised = None
        for index in range(1, len(steps)):
            denoised = _evaluate(session, x0_fn, noisy, steps[index])
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
        _commit(session, x0_fn, denoised, state.deterministic_context_noise)
        commit_ms = (time.perf_counter() - commit_started) * 1000.0
        block = int(session.block_index)
        session.block_index += 1
        state.consumed = True
        self.active = None
        return TransactionResult(
            block_index=block,
            latent=denoised.detach(),
            denoise_host_ms=denoise_ms,
            commit_host_ms=commit_ms,
            path="exact_hit_1_plus_3",
            action_sha256=action_hash,
        )

    def discard(self, state: ProposalState) -> None:
        self._validate(state, {}, exact_action=False) if False else None
        if state is not self.active or state.consumed:
            raise RuntimeError("proposal is not active")
        if parent_identity(self.session, self.logical_session_id) != state.parent_identity:
            raise RuntimeError("live session is not at proposal parent")
        state.consumed = True
        self.active = None

    def run_full(
        self,
        x0_fn: Any,
        action_tensors: Mapping[str, torch.Tensor],
        *,
        path: str = "full_observed",
    ) -> TransactionResult:
        if self.active is not None:
            raise RuntimeError("discard the active proposal before a full run")
        session = self.session
        steps = _steps_for_block(session)
        transition_noise, context_noise = self._reserve_noises(_initial_noisy(session))
        noisy = _initial_noisy(session)
        denoise_started = time.perf_counter()
        denoised = None
        for index, timestep in enumerate(steps):
            denoised = _evaluate(session, x0_fn, noisy, timestep)
            if index < len(steps) - 1:
                noisy = _add_noise(
                    session, denoised, transition_noise[index], steps[index + 1]
                )
        assert denoised is not None
        denoise_ms = (time.perf_counter() - denoise_started) * 1000.0
        _write_output(session, denoised)
        commit_started = time.perf_counter()
        _commit(session, x0_fn, denoised, context_noise)
        commit_ms = (time.perf_counter() - commit_started) * 1000.0
        block = int(session.block_index)
        session.block_index += 1
        return TransactionResult(
            block_index=block,
            latent=denoised.detach(),
            denoise_host_ms=denoise_ms,
            commit_host_ms=commit_ms,
            path=path,
            action_sha256=exact_action_sha256(action_tensors),
        )

    def miss(
        self,
        state: ProposalState,
        observed_x0_fn: Any,
        observed_action_tensors: Mapping[str, torch.Tensor],
    ) -> TransactionResult:
        observed_hash = self._validate(
            state, observed_action_tensors, exact_action=False
        )
        if observed_hash == state.exact_action_sha256:
            raise ValueError("miss path cannot be used for an exact action hit")
        self.session.restore_delta(state.parent_snapshot)
        state.consumed = True
        self.active = None
        return self.run_full(
            observed_x0_fn,
            observed_action_tensors,
            path="miss_full_observed",
        )
