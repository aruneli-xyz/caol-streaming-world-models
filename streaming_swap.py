"""Streaming driver v1: mid-rollout action re-conditioning for Gamma-World.

Seam: `generate_samples_from_batch` (self_forcing_dmd_mv) calls
`get_x0_fn_from_batch(data_batch, ...)` exactly once, and every denoising
call it makes carries `current_start` in token units
(latent_frame_index * frame_seq_length * n_views). We monkeypatch
`get_x0_fn_from_batch` to build TWO prediction functions -- one from the
pre-change action batch, one from the post-change batch -- and return a
dispatcher that switches on `current_start`. The KV cache is untouched:
prefix blocks were conditioned on the old actions, suffix blocks see the
new ones. This is genuine mid-stream re-conditioning at temporal-block
granularity without restarting generation.

Experiment (serving CAOL): a forward->stop transition at pixel frame 96.
Arms:
  offline  the post-change action sequence is known upfront
           (zero-admission-lag onset reference)
  swap+D   generation starts under all-forward conditioning; the action
           switch reaches the model D temporal blocks after the block
           containing the change frame (D=0,1,2,4), simulating action
           consumption at chunk boundaries with different chunk sizes.
Scoring: optical-flow onset (a2e.py) => serving CAOL vs D curve.

Run inside the Gamma-World venv (single GPU):
  external/Gamma-World/.venv/bin/torchrun --nproc_per_node=1 \
      ../../..../rtwm/streaming_swap.py run
Then: python streaming_swap.py score   (research venv, CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RTWM = Path(__file__).resolve().parent
SAFESWM = RTWM.parent / "safeswm"
REPO = SAFESWM / "external" / "Gamma-World"
MODELS = SAFESWM / "models"

N_FRAMES = 189
CHANGE_FRAME = 96
OUT = RTWM / "results" / "serving_a2e"
ARMS = [("offline", None), ("swap+0", 0), ("swap+1", 1), ("swap+2", 2), ("swap+4", 4)]


def build_action_tensors(pre: str, post: str, change_frame: int):
    import torch
    sys.path.insert(0, str(SAFESWM))
    from gamma.actions import commands_to_action_json

    cmds = [pre] * change_frame + [post] * (N_FRAMES - change_frame)
    d = commands_to_action_json(cmds)
    kb = torch.tensor(d["keyboard"], dtype=torch.float32).unsqueeze(0)
    cam = torch.tensor(d["camera"], dtype=torch.float32).unsqueeze(0)
    return kb, cam


def run() -> None:
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from PIL import Image

    torch.set_grad_enabled(False)
    from gamma_world._src.gamma_world.inference.inference_i2v import I2VInference
    from gamma_world._src.gamma_world.inference.model_specs import MODEL_SPECS
    from gamma_world._src.imaginaire.visualize.video import save_img_or_video

    sys.path.insert(0, str(REPO / "scripts"))
    from inference import format_hydra_value  # noqa: F401 (spec opts helper)

    spec = MODEL_SPECS["causal_few_step"]
    experiment_opts = [f"{k}={format_hydra_value(v)}" for k, v in spec.config_overrides.items()]
    engine = I2VInference(
        experiment_name=spec.experiment,
        ckpt_path=str(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
        config_file=spec.config_file,
        guidance=5.0,
        shift=None,
        num_sampling_steps=spec.default_num_steps or 35,
        seed=1,
        context_parallel_size=1,
        experiment_opts=experiment_opts,
        vae_pth=str(MODELS / "gamma-world" / "tokenizer.pth"),
        text_encoder_pth=str(MODELS / "Cosmos-Reason1-7B"),
    )
    engine.fps = 16
    model = engine.model

    # buildTower_normal: strongest measured ego-motion contrast for
    # stop/start transitions in the detector-onset sweep (flow 1.4 -> 0.015).
    img = Image.open(REPO / "data" / "buildTower_normal" / "first_frame.png")
    w = img.width // 2
    images = [np.asarray(img.crop((0, 0, w, img.height))),
              np.asarray(img.crop((w, 0, img.width, img.height)))]
    prompt = "Two Minecraft players exploring the world"

    kb_pre, cam_pre = build_action_tensors("forward", "forward", CHANGE_FRAME)
    kb_post, cam_post = build_action_tensors("forward", "stay", CHANGE_FRAME)

    n_views = 2
    nfpb = int(model.num_frame_per_block)
    change_latent = CHANGE_FRAME // 4                  # 4 px frames per latent
    change_block = change_latent // nfpb

    def make_batch(kb, cam):
        return engine.build_inference_batch(
            images, prompt, [(kb, cam), (kb, cam)],
            num_frames=N_FRAMES, num_conditional_frames=1)

    def preprocess_like_generate(batch):
        """Mirror generate_from_batch's preamble so the batch is valid input
        for get_x0_fn_from_batch."""
        from gamma_world._src.gamma_world.inference.inference_i2v import (
            IS_PREPROCESSED_KEY, to_with_skip_tensor)
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
        return batch

    orig_get = model.get_x0_fn_from_batch
    results = []
    OUT.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in os.environ.get("RTWM_SEEDS", "1").split(",")]
    for gen_seed in seeds:
      for arm, delay in ARMS:
        engine.clear_cache()
        if arm == "offline":
            model.get_x0_fn_from_batch = orig_get
            batch = make_batch(kb_post, cam_post)
            video = engine.generate_from_batch(batch, guidance=5.0, seed=gen_seed)
        else:
            switch_latent = (change_block + delay) * nfpb
            post_batch = preprocess_like_generate(make_batch(kb_post, cam_post))

            def patched_get(data_batch, **kw):
                fn_pre = orig_get(data_batch, **kw)
                fn_post = orig_get(post_batch, **kw)

                def dispatcher(noisy_image_or_video, timestep, kv_cache=None,
                               crossattn_cache=None, current_start=None,
                               current_end=None, start_frame_for_rope=None):
                    # start_frame_for_rope is the latent frame index of the
                    # block being denoised: dispatch on it directly (token
                    # arithmetic on current_start proved unreliable; the
                    # runtime frame_seq_length differs from latent HxW/4).
                    use_post = (start_frame_for_rope is not None
                                and start_frame_for_rope >= switch_latent)
                    fn = fn_post if use_post else fn_pre
                    return fn(noisy_image_or_video, timestep, kv_cache=kv_cache,
                              crossattn_cache=crossattn_cache,
                              current_start=current_start, current_end=current_end,
                              start_frame_for_rope=start_frame_for_rope)
                return dispatcher

            model.get_x0_fn_from_batch = patched_get
            try:
                batch = make_batch(kb_pre, cam_pre)
                video = engine.generate_from_batch(batch, guidance=5.0, seed=gen_seed)
            finally:
                model.get_x0_fn_from_batch = orig_get

        out_dir = OUT / f"seed{gen_seed}" / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        v = ((video + 1.0) / 2.0).clamp(0, 1)
        save_img_or_video(v[0], str(out_dir / "generated"), fps=16)
        eff_latent = (change_block + (delay or 0)) * nfpb if arm != "offline" else change_latent
        results.append(dict(arm=arm, seed=gen_seed, delay_blocks=delay,
                            change_frame=CHANGE_FRAME,
                            effective_switch_px=int(eff_latent * 4),
                            nfpb=nfpb, change_block=change_block))
        print(f"[serving-a2e] rendered seed{gen_seed}/{arm}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(results, indent=1))


def score() -> None:
    sys.path.insert(0, str(RTWM))
    from a2e import flow_signal, onset_frame

    manifest = json.loads((OUT / "manifest.json").read_text())
    print(f"{'arm':10s} {'switch_px':>9} {'onset_v0':>8} {'onset_v1':>8} {'CAOL_v0':>7} {'CAOL_v1':>7}")
    rows = []
    for m in manifest:
        video = OUT / m["arm"] / "generated.mp4"
        onsets = []
        for view in (0, 1):
            sig = flow_signal(video, view)
            onsets.append(onset_frame(sig, m["change_frame"], rising=False))
        a2e = [(o - m["change_frame"]) if o is not None else None for o in onsets]
        rows.append(dict(**m, onset_v0=onsets[0], onset_v1=onsets[1],
                         a2e_v0=a2e[0], a2e_v1=a2e[1]))
        print(f"{m['arm']:10s} {m['effective_switch_px']:>9} "
              f"{str(onsets[0]):>8} {str(onsets[1]):>8} {str(a2e[0]):>7} {str(a2e[1]):>7}")
    (OUT / "serving_a2e_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score"])
    if ap.parse_args().stage == "run":
        run()
    else:
        score()
