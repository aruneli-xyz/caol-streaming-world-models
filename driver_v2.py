"""Blockwise streaming session for Gamma-World (driver v2).

Replicates the temporal-block loop of `generate_samples_from_batch`
(self_forcing_dmd_mv) as an externally-steppable session, which the stock
engine does not expose. Capabilities the one-shot API lacks:

  - step one temporal block at a time (true streaming);
  - choose the action conditioning per block (mutable action stream);
  - fork/restore the KV cache at block boundaries (speculative branches,
    rollback-and-re-denoise);
  - per-block wall-clock timing (the latency numbers everything in this
    project needs);
  - per-block denoising-step count (adaptive compute).

Faithfulness: the math is copied from the upstream loop (denoise steps ->
re-noise -> exit -> context-noise cache write) for inference-only,
single-GPU (cp_size=1, is_training=False, context_noise per config).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from einops import rearrange


def clone_kv_cache(kv_cache, device="cpu"):
    """Snapshot the KV cache. The sparse-hub cache is tens of GB, so the
    default snapshots to host memory (fork cost = one PCIe transfer);
    a same-device clone would double GPU residency and OOM. Delta
    snapshots (only the block's written token range + end indices) are the
    corresponding optimization used by the measured delta-snapshot path."""
    out = []
    for entry in kv_cache:
        out.append({k: (v.detach().to(device, copy=True) if torch.is_tensor(v) else v)
                    for k, v in entry.items()})
    return out


def restore_kv_cache(kv_cache, snapshot):
    """Copy snapshot tensors back into the live cache in place."""
    for live, saved in zip(kv_cache, snapshot):
        for k, v in saved.items():
            if torch.is_tensor(v):
                live[k].copy_(v, non_blocking=True)
            else:
                live[k] = v
    torch.cuda.synchronize()


INDEX_KEYS = ("global_end_index", "local_end_index", "z_local_end_index")

# Delta snapshots for the sparse-hub KV cache
# ------------------------------------------
# Why this is sound: block writes are slice assignments at positions derived
# from the end indices, and attention only reads tokens below the end
# indices, so data beyond a restored end index is dead. The one mutation that
# moves *existing* data is the rolling-window eviction
# (_sparse_hub_rolling_step): when the cache is full, generating a block
# shifts everything above the sink left by exactly the evicted-block size and
# discards the prefix. Both effects are deterministic functions of the
# indices, so an exact undo needs only (a) the three end-index scalars and
# (b) a copy of the prefix that eviction will destroy. Restore = shift the
# surviving content back right, rewrite the saved prefix, reset the indices.
# Everything stays on-GPU, so fork/restore cost is HBM bandwidth, not PCIe.
#
# Scope: one speculated block per branch (the speculation use case). A
# multi-block branch accumulates several rolls and would need a deeper
# prefix; restore_kv_delta asserts the branch advanced at most one block.


def fork_kv_delta(kv_cache, block_p, block_z, sink_p=0, sink_z=0):
    """O(one block) snapshot at a block boundary; stays on the same device."""
    snap = []
    for e in kv_cache:
        L = int(e["local_end_index"].item())
        Z = int(e["z_local_end_index"].item())
        G = int(e["global_end_index"].item())
        ev_p = max(0, L + block_p - e["k_players"].shape[2])
        ev_z = max(0, Z + block_z - e["k_z"].shape[1])
        d = dict(G=G, L=L, Z=Z, ev_p=ev_p, ev_z=ev_z)
        if ev_p:  # the roll will destroy this prefix
            d["pk"] = e["k_players"][:, :, sink_p:sink_p + ev_p].clone()
            d["pv"] = e["v_players"][:, :, sink_p:sink_p + ev_p].clone()
        if ev_z:
            d["zk"] = e["k_z"][:, sink_z:sink_z + ev_z].clone()
            d["zv"] = e["v_z"][:, sink_z:sink_z + ev_z].clone()
        snap.append(d)
    torch.cuda.synchronize()
    return snap


def restore_kv_delta(kv_cache, snap, block_p, sink_p=0, sink_z=0):
    """Exact undo of at most one generated block since fork_kv_delta."""
    for e, d in zip(kv_cache, snap):
        adv = int(e["global_end_index"].item()) - d["G"]
        assert 0 <= adv <= block_p * e["k_players"].shape[1], \
            f"delta restore supports one block per branch, advance={adv}"
        if adv > 0 and d["ev_p"] > 0:
            # undo the roll: shift survivors back right, rewrite the prefix
            L, ev = d["L"], d["ev_p"]
            keep = L - ev - sink_p
            for kk, pk in (("k_players", "pk"), ("v_players", "pv")):
                tmp = e[kk][:, :, sink_p:sink_p + keep].clone()
                e[kk][:, :, sink_p + ev:sink_p + ev + keep] = tmp
                e[kk][:, :, sink_p:sink_p + ev] = d[pk]
            Zi, evz = d["Z"], d["ev_z"]
            keep_z = Zi - evz - sink_z
            for kk, zk in (("k_z", "zk"), ("v_z", "zv")):
                tmp = e[kk][:, sink_z:sink_z + keep_z].clone()
                e[kk][:, sink_z + evz:sink_z + evz + keep_z] = tmp
                e[kk][:, sink_z:sink_z + evz] = d[zk]
        # anything beyond the restored end indices is dead data
        e["global_end_index"].fill_(d["G"])
        e["local_end_index"].fill_(d["L"])
        e["z_local_end_index"].fill_(d["Z"])
    torch.cuda.synchronize()


def capture_kv_block(kv_cache, block_p, block_z):
    """Save the KV slice a just-generated block wrote (per layer).

    Two branches forked from the same boundary apply the identical roll (the
    shift moves pre-fork data and is a deterministic function of the saved
    indices), so their caches differ only in this slice. Writing it back over
    the other branch's post-state switches branches without regeneration.
    """
    out = []
    for e in kv_cache:
        L = int(e["local_end_index"].item())
        Z = int(e["z_local_end_index"].item())
        out.append(dict(
            G=int(e["global_end_index"].item()), L=L, Z=Z,
            block_p=block_p, block_z=block_z,
            pk=e["k_players"][:, :, L - block_p:L].clone(),
            pv=e["v_players"][:, :, L - block_p:L].clone(),
            zk=e["k_z"][:, Z - block_z:Z].clone(),
            zv=e["v_z"][:, Z - block_z:Z].clone()))
    torch.cuda.synchronize()
    return out


def apply_kv_block(kv_cache, cap):
    """Overwrite the live cache's last-block slice with a captured branch."""
    for e, d in zip(kv_cache, cap):
        assert int(e["local_end_index"].item()) == d["L"], \
            "apply_kv_block: live cache is not at the captured branch boundary"
        e["k_players"][:, :, d["L"] - d["block_p"]:d["L"]] = d["pk"]
        e["v_players"][:, :, d["L"] - d["block_p"]:d["L"]] = d["pv"]
        e["k_z"][:, d["Z"] - d["block_z"]:d["Z"]] = d["zk"]
        e["v_z"][:, d["Z"] - d["block_z"]:d["Z"]] = d["zv"]
    torch.cuda.synchronize()


def snapshot_nbytes(snap):
    total = 0
    for d in snap:
        for v in d.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


@dataclass
class BlockResult:
    block_index: int
    latent: torch.Tensor          # (B, V*nfpb, C, H, W) denoised latents
    denoise_ms: float
    steps_used: int
    commit_ms: float
    block_started_ns: int
    denoise_complete_ns: int
    latent_committed_ns: int


@dataclass
class DecodeResult:
    video: torch.Tensor
    decode_started_ns: int
    decode_complete_ns: int
    decode_ms: float


class BlockwiseSession:
    """One streaming rollout; actions chosen per block via x0_fn handles."""

    def __init__(self, engine, batch_for_shapes, seed: int = 1):
        self.model = engine.model
        model = self.model
        from gamma_world._src.imaginaire.utils import misc

        # --- shapes (mirrors generate_samples_from_batch preamble) ---
        b = batch_for_shapes
        num_px = int(b["num_video_frames_per_view"].cpu().item())
        self.num_pixel_frames_per_view = num_px
        n_views = b["view_indices"].shape[1] // num_px
        _T, _H, _W = b["video"].shape[-3:]
        lat_t = model.tokenizer.get_latent_num_frames(_T // n_views)
        state_shape = [lat_t, model.config.state_ch,
                       _H // model.tokenizer.spatial_compression_factor,
                       _W // model.tokenizer.spatial_compression_factor]
        flat = (lat_t * n_views, state_shape[1], state_shape[2], state_shape[3])
        self.n_views = n_views
        self.lat_t = lat_t
        self.nfpb = int(model.num_frame_per_block)
        self.num_blocks = lat_t // self.nfpb
        self.noise = misc.arch_invariant_rand(
            (1,) + tuple(flat), torch.float32, model.tensor_kwargs["device"], seed)
        misc.set_random_seed(seed=seed, by_rank=False)
        model.frame_seq_length = int(self.noise.shape[-1] * self.noise.shape[-2] / 4)
        self.token_stride = model.frame_seq_length * n_views
        self.frames_per_tblock = n_views * self.nfpb

        self.output = torch.zeros(
            (1, flat[0], flat[1], flat[2], flat[3]),
            device=self.noise.device, dtype=self.noise.dtype)

        model._initialize_kv_cache(batch_size=1, n_views=n_views,
                                   dtype=model.tensor_kwargs["dtype"],
                                   device=model.tensor_kwargs["device"],
                                   num_training_frames=lat_t, is_training=False)
        model.crossattn_cache = None
        self.block_index = 0
        # per-view tokens one temporal block writes into the sparse-hub cache
        self.block_p = self.nfpb * model.frame_seq_length
        z_num = getattr(model.net, "z_num", 8)
        self.block_z = self.nfpb * z_num

    # -------------------------------------------------------------- fork

    def fork(self):
        return dict(kv=clone_kv_cache(self.model.kv_cache1),
                    block_index=self.block_index,
                    output=self.output.clone())

    def restore(self, snap):
        restore_kv_cache(self.model.kv_cache1, snap["kv"])
        self.block_index = snap["block_index"]
        self.output = snap["output"].clone()

    # -------------------------------------------- delta fork (O(one block))

    def fork_delta(self):
        """Cheap on-GPU snapshot at the current block boundary. Valid for
        branches that generate at most one block before restore."""
        return dict(kv=fork_kv_delta(self.model.kv_cache1,
                                     self.block_p, self.block_z),
                    block_index=self.block_index)

    def restore_delta(self, snap):
        restore_kv_delta(self.model.kv_cache1, snap["kv"], self.block_p)
        self.block_index = snap["block_index"]
        # output slices beyond block_index are dead; each step overwrites its
        # own block slice, so no output rollback is needed.

    def capture_branch(self):
        """Capture the last generated block (KV slice + output latents) so a
        sibling branch can be swapped in later without regeneration."""
        bi = self.block_index - 1
        lo, hi = bi * self.nfpb, (bi + 1) * self.nfpb
        out_v = rearrange(self.output, "b (v t) c h w -> b v t c h w",
                          v=self.n_views)
        return dict(kv=capture_kv_block(self.model.kv_cache1,
                                        self.block_p, self.block_z),
                    block_index=self.block_index,
                    out_block=out_v[:, :, lo:hi].clone())

    def apply_branch(self, cap):
        """Overwrite the live last-block state with a captured sibling branch.
        The session must be at the same post-block boundary."""
        assert self.block_index == cap["block_index"], \
            "apply_branch: session not at the captured boundary"
        apply_kv_block(self.model.kv_cache1, cap["kv"])
        bi = self.block_index - 1
        lo, hi = bi * self.nfpb, (bi + 1) * self.nfpb
        out_v = rearrange(self.output, "b (v t) c h w -> b v t c h w",
                          v=self.n_views)
        out_v[:, :, lo:hi] = cap["out_block"]
        self.output = rearrange(out_v, "b v t c h w -> b (v t) c h w")

    # -------------------------------------------------------------- step

    def step_block(self, x0_fn, denoising_step_list=None) -> BlockResult:
        """Generate the next temporal block under the given prediction fn."""
        model = self.model
        steps = denoising_step_list if denoising_step_list is not None \
            else model.denoising_step_list
        bi = self.block_index
        assert bi < self.num_blocks, "session finished"
        cur, end = bi * self.nfpb, (bi + 1) * self.nfpb

        noise_v = rearrange(self.noise, "b (v t) c h w -> b v t c h w", v=self.n_views)
        noisy = rearrange(noise_v[:, :, cur:end], "b v t c h w -> b (v t) c h w")

        torch.cuda.synchronize()
        block_started_ns = time.perf_counter_ns()
        n_steps = len(steps)
        for idx in range(n_steps):
            ts = torch.ones((1, self.frames_per_tblock), device=self.noise.device,
                            dtype=torch.int64) * int(steps[idx])
            # Raw model-level x0_fn signature: (noise_x, timestep, kv_cache=,
            # **kwargs) -- positional first arg, extras through kwargs.
            denoised = x0_fn(noisy, ts,
                             kv_cache=model.kv_cache1,
                             crossattn_cache=model.crossattn_cache,
                             current_start=cur * self.token_stride,
                             current_end=end * self.token_stride,
                             start_frame_for_rope=cur)
            if idx == n_steps - 1:
                break
            nxt = int(steps[idx + 1])
            cn = torch.randn_like(denoised.flatten(0, 1))
            noisy = model.scheduler.add_noise(
                denoised.flatten(0, 1), cn,
                nxt * torch.ones([self.frames_per_tblock], device=self.noise.device,
                                 dtype=torch.long)
            ).unflatten(0, denoised.shape[:2])
        torch.cuda.synchronize()
        denoise_complete_ns = time.perf_counter_ns()
        denoise_ms = (denoise_complete_ns - block_started_ns) / 1_000_000

        out_v = rearrange(self.output, "b (v t) c h w -> b v t c h w", v=self.n_views)
        out_v[:, :, cur:end] = rearrange(denoised, "b (v t) c h w -> b v t c h w",
                                         v=self.n_views)
        self.output = rearrange(out_v, "b v t c h w -> b (v t) c h w")

        # context-cache write (context_noise handling mirrors upstream)
        commit_started_ns = time.perf_counter_ns()
        ctx_pred = denoised
        ctx_ts = torch.ones((1, self.frames_per_tblock), device=self.noise.device,
                            dtype=torch.int64) * int(model.config.context_noise)
        if model.config.context_noise > 0:
            cn = torch.randn_like(denoised.flatten(0, 1))
            ctx_pred = model.scheduler.add_noise(
                denoised.flatten(0, 1), cn,
                int(model.config.context_noise) * torch.ones(
                    [self.frames_per_tblock], device=self.noise.device, dtype=torch.long)
            ).unflatten(0, denoised.shape[:2])
        x0_fn(ctx_pred, ctx_ts,
              kv_cache=model.kv_cache1, crossattn_cache=model.crossattn_cache,
              current_start=cur * self.token_stride,
              current_end=end * self.token_stride,
              start_frame_for_rope=cur)
        torch.cuda.synchronize()
        latent_committed_ns = time.perf_counter_ns()
        commit_ms = (latent_committed_ns - commit_started_ns) / 1_000_000

        self.block_index += 1
        return BlockResult(
            bi,
            denoised.detach(),
            denoise_ms,
            n_steps,
            commit_ms,
            block_started_ns,
            denoise_complete_ns,
            latent_committed_ns,
        )

    # ------------------------------------------------------------ decode

    def decode(self) -> torch.Tensor:
        sample = self.output.permute(0, 2, 1, 3, 4)
        return self.model.decode(sample)

    def decode_current_timed(self) -> DecodeResult:
        """Decode the generated prefix in the full latent buffer.

        Future latent slots remain zero. The measured causal decoder support
        determines which rendered frames are eligible to be called ready.
        """
        torch.cuda.synchronize()
        started_ns = time.perf_counter_ns()
        video = self.decode()
        torch.cuda.synchronize()
        complete_ns = time.perf_counter_ns()
        return DecodeResult(
            video=video,
            decode_started_ns=started_ns,
            decode_complete_ns=complete_ns,
            decode_ms=(complete_ns - started_ns) / 1_000_000,
        )
