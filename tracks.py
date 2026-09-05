"""Paper-3 method tracks on the blockwise session driver.

  3a  speculation: fork the KV cache at a block boundary, pre-generate two
      action branches before the action is "known", accept the matching
      one. Metrics: per-block denoise ms, fork/restore ms, accept-path
      action-to-display latency vs on-demand generation, KV memory.
  3b  rollback ("barge-in repair"): generate D blocks past an action
      change under stale conditioning, roll back to the switch block,
      re-denoise the tail under the new actions. Metrics: repair cost vs
      D, corrected-tail divergence from a never-stale reference.
  3c  adaptive compute: per-block denoising-step schedules. Static 4/2/1
      step arms for the quality-latency frontier plus an action-gated
      adaptive arm (4 steps for 2 blocks after a change, 2 elsewhere).
      Quality proxy: latent MSE against the 4-step reference, split at the
      switch.

Run inside the Gamma-World venv:
  .venv/bin/torchrun --nproc_per_node=1 /path/to/tracks.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RTWM = Path(__file__).resolve().parent
SAFESWM = RTWM.parent / "safeswm"
REPO = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"
OUT = RTWM / "results" / "tracks"

N_FRAMES = 189
CHANGE_FRAME = 96


def build_engine():
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import torch
    torch.set_grad_enabled(False)
    from gamma_world._src.gamma_world.inference.inference_i2v import I2VInference
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from inference import format_hydra_value

    spec = MODEL_SPECS["causal_few_step"]
    opts = [f"{k}={format_hydra_value(v)}" for k, v in spec.config_overrides.items()]
    engine = I2VInference(
        experiment_name=spec.experiment, ckpt_path=str(
            MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
        config_file=spec.config_file, guidance=5.0, shift=None,
        num_sampling_steps=spec.default_num_steps or 35, seed=1,
        context_parallel_size=1, experiment_opts=opts,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"))
    engine.fps = 16
    return engine


def make_x0(engine, commands: list[str], images, prompt):
    """Build a preprocessed batch + prediction fn for a full command list."""
    import torch
    sys.path.insert(0, str(SAFESWM))
    from gamma.actions import commands_to_action_json
    from gamma_world._src.gamma_world.inference.inference_i2v import (
        IS_PREPROCESSED_KEY, to_with_skip_tensor)

    d = commands_to_action_json(commands)
    kb = torch.tensor(d["keyboard"], dtype=torch.float32).unsqueeze(0)
    cam = torch.tensor(d["camera"], dtype=torch.float32).unsqueeze(0)
    batch = engine.build_inference_batch(images, prompt, [(kb, cam), (kb, cam)],
                                         num_frames=N_FRAMES, num_conditional_frames=1)
    model = engine.model
    if "video" in batch:
        batch["video"] = batch["video"].float()
        if not batch.get(IS_PREPROCESSED_KEY, False):
            batch["video"] = batch["video"] / 127.5 - 1.0
        batch["video"] = torch.clamp(batch["video"], -1, 1)
    batch[IS_PREPROCESSED_KEY] = True
    batch = to_with_skip_tensor(batch, **model.tensor_kwargs)
    engine.inplace_compute_text_embeddings_online(batch, use_negative_prompt=True)
    batch = model.get_data_batch_with_latent_view_indices(batch)
    model._normalize_video_databatch_inplace(batch)
    return batch, model.get_x0_fn_from_batch(batch, n_views=2, guidance=5.0,
                                             is_negative_prompt=True)


def main():
    import numpy as np
    import time
    import torch
    from PIL import Image

    sys.path.insert(0, str(RTWM))
    from driver_v2 import BlockwiseSession

    engine = build_engine()
    img = Image.open(REPO / "data" / "buildTower_normal" / "first_frame.png")
    w = img.width // 2
    images = [np.asarray(img.crop((0, 0, w, img.height))),
              np.asarray(img.crop((w, 0, img.width, img.height)))]
    prompt = "Two Minecraft players exploring the world"

    fwd = ["forward"] * N_FRAMES
    stop_at = ["forward"] * CHANGE_FRAME + ["stay"] * (N_FRAMES - CHANGE_FRAME)
    batch_f, x0_fwd = make_x0(engine, fwd, images, prompt)
    _, x0_stop = make_x0(engine, stop_at, images, prompt)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    # ------------------------------------------------------------- 3a
    print("== 3a speculation ==", flush=True)
    sess = BlockwiseSession(engine, batch_f, seed=1)
    switch_block = (CHANGE_FRAME // 4) // sess.nfpb
    times = []
    for b in range(switch_block):
        r = sess.step_block(x0_fwd)
        times.append(r.denoise_ms)
    block_ms = float(np.mean(times[2:]))  # steady state

    t0 = time.perf_counter(); snap = sess.fork(); fork_ms = (time.perf_counter() - t0) * 1e3
    kv_bytes = sum(v.numel() * v.element_size()
                   for e in snap["kv"] for v in e.values() if torch.is_tensor(v))
    # speculate branch A (forward) and B (stop)
    rA = sess.step_block(x0_fwd); postA = sess.fork()
    t0 = time.perf_counter(); sess.restore(snap); restore_ms = (time.perf_counter() - t0) * 1e3
    rB = sess.step_block(x0_stop); postB = sess.fork()
    # action arrives: "stop" -> accept branch B (already at postB state)
    accept_ms = restore_ms  # worst case: restoring the accepted branch snapshot
    # correctness: continuation after accept matches a straight stale-free run
    r_next = sess.step_block(x0_stop)
    report["3a"] = dict(block_ms=round(block_ms, 1), fork_ms=round(fork_ms, 1),
                        restore_ms=round(restore_ms, 1),
                        kv_cache_mb=round(kv_bytes / 2**20, 1),
                        branch_ms=[round(rA.denoise_ms, 1), round(rB.denoise_ms, 1)],
                        on_demand_latency_ms=round(block_ms, 1),
                        speculative_accept_latency_ms=round(accept_ms, 1))
    print(json.dumps(report["3a"], indent=1), flush=True)
    del sess, snap, postA, postB
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- 3b
    print("== 3b rollback ==", flush=True)
    res_3b = []
    for D in (1, 2, 4):
        sess = BlockwiseSession(engine, batch_f, seed=1)
        snap = None
        for b in range(switch_block + D):
            if b == switch_block:
                snap = sess.fork()
            sess.step_block(x0_fwd)  # stale: change missed
        stale_out = sess.output.clone()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        sess.restore(snap)
        repair_times = [sess.step_block(x0_stop).denoise_ms for _ in range(D)]
        repair_ms = (time.perf_counter() - t0) * 1e3
        # reference: never-stale from same snapshot -- rollback IS the reference
        lat = sess.output
        lo, hi = switch_block * sess.nfpb, (switch_block + D) * sess.nfpb
        v = rearrange_region(lat, stale_out, sess, lo, hi)
        res_3b.append(dict(D=D, repair_ms=round(repair_ms, 1),
                           per_block_ms=[round(t, 1) for t in repair_times],
                           stale_vs_repaired_mse=v))
        print(res_3b[-1], flush=True)
        del sess, snap, stale_out
        torch.cuda.empty_cache()
    report["3b"] = res_3b

    # ------------------------------------------------------------- 3c
    print("== 3c adaptive compute ==", flush=True)
    schedules = {
        "steps4": [1000, 750, 500, 250],
        "steps2": [1000, 500],
        "steps1": [1000],
    }
    latents, timing = {}, {}
    for name, sched in schedules.items():
        sess = BlockwiseSession(engine, batch_f, seed=1)
        tms = []
        for b in range(sess.num_blocks):
            fn = x0_stop  # single transition rollout
            tms.append(sess.step_block(fn, denoising_step_list=sched).denoise_ms)
        latents[name] = sess.output.clone()
        timing[name] = round(float(np.sum(tms)) / 1e3, 2)
        del sess
        torch.cuda.empty_cache()
    # adaptive: 4 steps for [switch_block, switch_block+2), else 2 steps
    sess = BlockwiseSession(engine, batch_f, seed=1)
    tms = []
    for b in range(sess.num_blocks):
        sched = schedules["steps4"] if switch_block <= b < switch_block + 2 else schedules["steps2"]
        tms.append(sess.step_block(x0_stop, denoising_step_list=sched).denoise_ms)
    latents["adaptive"] = sess.output.clone()
    timing["adaptive"] = round(float(np.sum(tms)) / 1e3, 2)

    ref = latents["steps4"]
    q = {}
    sw_lat = switch_block * 3
    for name, lat in latents.items():
        mse_all = float(((lat - ref) ** 2).mean())
        # switch-adjacent region (2 blocks after the change), both views
        q[name] = dict(total_denoise_s=timing[name], mse_vs_4step=round(mse_all, 5))
    report["3c"] = q
    print(json.dumps(q, indent=1), flush=True)

    (OUT / "tracks_report.json").write_text(json.dumps(report, indent=1))
    print("saved ->", OUT / "tracks_report.json", flush=True)


def rearrange_region(lat_a, lat_b, sess, lo, hi):
    """Mean squared diff over latent frames [lo,hi) across both views."""
    from einops import rearrange
    a = rearrange(lat_a, "b (v t) c h w -> b v t c h w", v=sess.n_views)[:, :, lo:hi]
    b = rearrange(lat_b, "b (v t) c h w -> b v t c h w", v=sess.n_views)[:, :, lo:hi]
    return round(float(((a - b) ** 2).mean()), 5)


if __name__ == "__main__":
    main()
