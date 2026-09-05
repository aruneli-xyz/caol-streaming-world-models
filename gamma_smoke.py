"""Minimal real Gamma-World block-causal rollout, for notebook validation.

Builds the real engine, runs a few temporal blocks through the streaming
session, decodes, and writes frame0 + per-block latency to results/.
Run inside the Gamma-World venv:
  .venv/bin/torchrun --nproc_per_node=1 /path/to/gamma_smoke.py --blocks 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RTWM = Path(__file__).resolve().parent
sys.path.insert(0, str(RTWM))
OUT = RTWM / "results" / "smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=3)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from tracks import build_engine, make_x0, N_FRAMES, REPO
    from driver_v2 import BlockwiseSession

    OUT.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    engine = build_engine()
    load_s = time.perf_counter() - t_load

    img = Image.open(REPO / "data" / "buildTower_normal" / "first_frame.png")
    w = img.width // 2
    images = [np.asarray(img.crop((0, 0, w, img.height))),
              np.asarray(img.crop((w, 0, img.width, img.height)))]
    prompt = "Two Minecraft players exploring the world"
    batch, x0 = make_x0(engine, ["forward"] * N_FRAMES, images, prompt)

    sess = BlockwiseSession(engine, batch, seed=1)
    n = min(args.blocks, sess.num_blocks)
    per_block_ms = []
    for _ in range(n):
        per_block_ms.append(round(sess.step_block(x0).denoise_ms, 1))

    video = sess.decode()  # (B, C, T, H, W) or (B, T, C, H, W)
    v = video.detach().float().cpu()
    print("decoded video tensor shape:", tuple(v.shape), flush=True)

    report = dict(
        load_s=round(load_s, 1),
        n_views=sess.n_views,
        nfpb=sess.nfpb,
        num_blocks=sess.num_blocks,
        blocks_run=n,
        per_block_ms=per_block_ms,
        video_shape=list(v.shape),
    )
    (OUT / "gamma_smoke.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    print("saved ->", OUT / "gamma_smoke.json", flush=True)


if __name__ == "__main__":
    main()
