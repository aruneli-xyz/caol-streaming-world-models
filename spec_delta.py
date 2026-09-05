"""3a follow-up: speculation with delta KV snapshots (on-GPU, O(one block)).

The tracks pilot showed full-cache speculation is a net loss: cloning the
37.8 GB sparse-hub cache to host costs 21.4 s (fork) + 4.2 s (restore),
several times the 4.7 s block time it is meant to hide. This experiment
replaces the full clone with the delta primitives in driver_v2:

  fork_delta      save 3 index scalars/layer + the prefix the rolling window
                  will evict (0 bytes pre-roll, ~one block post-roll)
  restore_delta   undo the roll shift in place, reset indices
  capture_branch  save the one-block KV slice + latents a branch wrote
  apply_branch    swap a captured sibling branch into the live cache

Protocol (fork at the first rolling block, the hard case):
  1. Ground-truth runs: two full sessions that take branch B (stop) or
     branch A (forward) at the switch and continue one block. RNG is
     re-seeded before every step so trajectories are comparable.
  2. Delta run: generate to the switch, fork_delta, generate branch A,
     capture it, restore_delta, generate branch B (compare to ground truth
     B -- validates the roll undo), then apply_branch(A) and continue
     (compare to ground truth A continuation -- validates that attention
     over the swapped KV reproduces the true branch-A state).
  3. Latency accounting: fork/restore/capture/apply wall times and snapshot
     sizes vs the full-clone numbers.

Run inside the Gamma-World venv:
  .venv/bin/torchrun --nproc_per_node=1 /path/to/spec_delta.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RTWM = Path(__file__).resolve().parent
OUT = RTWM / "results" / "tracks"

sys.path.insert(0, str(RTWM))
from tracks import build_engine, make_x0, N_FRAMES, CHANGE_FRAME, REPO  # noqa: E402


def reseed(seed=1234):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def block_lat(sess, bi):
    from einops import rearrange
    v = rearrange(sess.output, "b (v t) c h w -> b v t c h w", v=sess.n_views)
    return v[:, :, bi * sess.nfpb:(bi + 1) * sess.nfpb].clone()


def mad(a, b):
    return float((a - b).abs().max())


def main():
    import numpy as np
    import torch
    from PIL import Image

    from driver_v2 import BlockwiseSession, snapshot_nbytes

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

    report = {}

    def run_prefix():
        sess = BlockwiseSession(engine, batch_f, seed=1)
        sb = (CHANGE_FRAME // 4) // sess.nfpb
        tms = []
        for _ in range(sb):
            reseed()
            tms.append(sess.step_block(x0_fwd).denoise_ms)
        return sess, sb, tms

    # ---- ground truth: branch B (stop) then continue -------------------
    print("== ground truth B ==", flush=True)
    sess, sb, _ = run_prefix()
    reseed(); sess.step_block(x0_stop)
    gtB_switch = block_lat(sess, sb)
    reseed(); sess.step_block(x0_stop)
    gtB_cont = block_lat(sess, sb + 1)
    del sess; torch.cuda.empty_cache()

    # ---- ground truth: branch A (forward) then continue (stop) ---------
    print("== ground truth A ==", flush=True)
    sess, sb, _ = run_prefix()
    reseed(); sess.step_block(x0_fwd)
    gtA_switch = block_lat(sess, sb)
    reseed(); sess.step_block(x0_stop)
    gtA_cont = block_lat(sess, sb + 1)
    del sess; torch.cuda.empty_cache()

    # ---- delta speculation run -----------------------------------------
    print("== delta speculation ==", flush=True)
    sess, sb, prefix_tms = run_prefix()
    block_ms = float(np.mean(prefix_tms[2:]))

    t0 = time.perf_counter(); snap = sess.fork_delta()
    fork_ms = (time.perf_counter() - t0) * 1e3
    fork_bytes = snapshot_nbytes(snap["kv"])

    reseed(); rA = sess.step_block(x0_fwd)
    err_specA = mad(block_lat(sess, sb), gtA_switch)

    t0 = time.perf_counter(); capA = sess.capture_branch()
    capture_ms = (time.perf_counter() - t0) * 1e3
    cap_bytes = snapshot_nbytes(capA["kv"])

    t0 = time.perf_counter(); sess.restore_delta(snap)
    restore_ms = (time.perf_counter() - t0) * 1e3

    reseed(); rB = sess.step_block(x0_stop)
    err_after_restore = mad(block_lat(sess, sb), gtB_switch)

    # action arrives: it was A -> swap the captured branch in
    t0 = time.perf_counter(); sess.apply_branch(capA)
    apply_ms = (time.perf_counter() - t0) * 1e3
    err_after_apply = mad(block_lat(sess, sb), gtA_switch)

    reseed(); sess.step_block(x0_stop)
    err_continuation = mad(block_lat(sess, sb + 1), gtA_cont)

    report["3a_delta"] = dict(
        switch_block=sb,
        block_ms=round(block_ms, 1),
        branch_ms=[round(rA.denoise_ms, 1), round(rB.denoise_ms, 1)],
        fork_ms=round(fork_ms, 1),
        restore_ms=round(restore_ms, 1),
        capture_ms=round(capture_ms, 1),
        apply_ms=round(apply_ms, 1),
        fork_snapshot_mb=round(fork_bytes / 2**20, 1),
        capture_mb=round(cap_bytes / 2**20, 1),
        on_demand_latency_ms=round(block_ms, 1),
        speculative_accept_latency_ms=round(apply_ms, 1),
        # correctness vs ground truth (max abs latent diff; 0 = exact)
        err_speculated_branch=err_specA,
        err_restore_then_regen=err_after_restore,
        err_apply_branch=err_after_apply,
        err_continuation_after_apply=err_continuation,
    )
    print(json.dumps(report["3a_delta"], indent=1), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "spec_delta_report.json"
    path.write_text(json.dumps(report, indent=1))
    print("saved ->", path, flush=True)


if __name__ == "__main__":
    main()
