"""Generator for rtwm_tutorial.ipynb -- the real-Gamma-World tutorial.

No toy models. Every experiment is the real Gamma-World checkpoint: the
notebook runs a real short rollout live, and visualizes the committed real
measurements (detector-defined CAOL, serving decomposition, speculation, rollback,
adaptive compute). Plain-English (ELI5) explanations accompany every term.

Build (uses the research venv for authoring):
  ../../.venv/bin/python _build_rtwm_tutorial.py
Execute (research venv can read results + shell out to the Gamma-World venv):
  ../../.venv/bin/python -m nbconvert --to notebook --execute --inplace rtwm_tutorial.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# Real-Time World Models: the Gamma-World inference tutorial

**Everything here runs the real Gamma-World checkpoint. There are no toy models.** The notebook (1) runs a real short rollout live on the GPU, and (2) visualizes the real measurements committed in `rtwm/results/` for the experiments that take minutes each, showing the exact command to reproduce every one.

Gamma-World is an action-conditioned, autoregressive video world model (a Minecraft "playable world"): you feed it a first frame plus a stream of keyboard/mouse actions, and it generates the video of that world responding, block of frames at a time, forever. It is the hardest class of model to serve, and this project (Paper 3, "Real-Time World Models") is about the *inference* side: how fast the world reacts to your controls, and the serving tricks (speculation, rollback, adaptive compute) that make it interactive.

**How to read this notebook.** Every technical term gets a plain-English **ELI5** line before the precise definition. The sections follow the paper: the pipeline, then the metric (control-aligned onset latency), then the serving decomposition, then the three method tracks.

> **Where it runs:** a GPU box with the Gamma-World environment installed (this repo used an H200 with `safeswm/external/Gamma-World/.venv`). The heavy model is invoked through that venv; the plots and tables run in a plain Python kernel. It is not a free-Colab notebook: the model needs the Cosmos text encoder and the checkpoint (~30 GB) and a bespoke environment.
""")

# ---------------------------------------------------------------- install
md(r"""
## Install (run first on Colab)

*ELI5: grab the plotting libraries.* This notebook plots and tables the real committed measurements and (on the H200 box) runs a live Gamma-World rollout. On Colab, install the light dependencies below; the **live rollout and the video frame strips need the Gamma-World environment + committed media on the H200 box**, so on Colab those specific cells print a note instead. Clone/upload the `rtwm/` folder so the committed result tables (`a2e_results.csv`, `scale_summary.csv`, `tracks_report.json`) are present.
""")

code(r"""
import sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print("Colab detected; installing packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "pandas", "matplotlib", "imageio[ffmpeg]"], check=False)
    print("done.")
else:
    print("Not on Colab; assuming pandas / matplotlib / imageio are already installed.")
""")

# ---------------------------------------------------------------- setup
md(r"""
## Setup: locate the repo, the model environment, and the results
""")

code(r"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Locate the rtwm project regardless of where the kernel started.
CANDS = [Path.cwd(), Path.cwd().parent]
RTWM = next((p for p in CANDS if (p / "driver_v2.py").exists()), CANDS[-1])
RESULTS = RTWM / "results"
GAMMA_DIR = RTWM.parent / "safeswm" / "external" / "Gamma-World"
TORCHRUN = GAMMA_DIR / ".venv" / "bin" / "torchrun"

print("rtwm project :", RTWM)
print("results dir  :", RESULTS, "(exists:", RESULTS.exists(), ")")
print("Gamma-World  :", GAMMA_DIR, "(venv torchrun exists:", TORCHRUN.exists(), ")")
""")

# ---------------------------------------------------------------- glossary
md(r"""
## The vocabulary, explained simply

Every term used in this notebook, ELI5 first.

- **World model** — *ELI5: a video game engine that was learned, not programmed.* A neural net that, given a starting image and your controls, predicts what you would see next.
- **Latent** — *ELI5: a small compressed sketch of a frame.* The model works on compact codes, not raw pixels.
- **VAE (variational autoencoder)** — *ELI5: a zip/unzip for images.* The *encoder* compresses pixels to a latent; the *decoder* expands a latent back to pixels. Gamma-World's VAE compresses time by 4 and each spatial side by 16.
- **DiT (diffusion transformer)** — *ELI5: the artist.* The transformer that turns noise into a clean latent, guided by your actions and the past.
- **Denoising step** — *ELI5: one brush stroke from static toward a clear picture.* Diffusion starts from noise and refines over a few steps. Gamma-World's distilled model uses ~4 steps.
- **CFG (classifier-free guidance)** — *ELI5: "follow the instructions harder."* A quality trick that runs the artist twice per step (once ignoring the prompt, once following it) and exaggerates the difference. It doubles the compute per step; distilled models usually skip it.
- **Distillation** — *ELI5: train a fast student to copy a slow teacher.* Turns a 30-to-50-step model into a 4-step one.
- **KV cache (key/value cache)** — *ELI5: the model's short-term memory of everything it drew so far.* Attention reuses it so each new block is cheap instead of re-reading all history. Gamma-World's is a "sparse-hub" cache, tens of GB.
- **Block-causal streaming** — *ELI5: draw the movie a few frames at a time, left to right, never peeking ahead.* Each block attends to a rolling window of the past via the KV cache.
- **Action conditioning** — *ELI5: your keyboard/mouse becomes part of the prompt* for each block.
- **CAOL (control-aligned onset latency)** — *ELI5: how long from pressing a key to a detector-aligned rendered motion onset.* The headline non-causal interactivity metric.
- **Fork** — *ELI5: save-game the model's memory so you can try a branch and come back.*
- **Rollback** — *ELI5: undo bad frames and redraw them* once you notice they used stale controls.
- **Speculation** — *ELI5: pre-draw the most likely next move so it is ready instantly if the player does it.*
- **Adaptive compute** — *ELI5: think harder only when it matters* (more denoising steps right after a control change, fewer when nothing is happening).
""")

# ---------------------------------------------------------------- the model
md(r"""
## The real model: what Gamma-World actually is

*ELI5: a learned Minecraft that two players can walk around in, generated frame-block by frame-block.*

Structural facts that shape everything about serving it (printed by the live run below):

- **Block-causal, rolling state.** It generates **16 temporal blocks of 3 latent frames** each (`num_frame_per_block = 3`), for **2 player views**. Each block is denoised against a **sparse-hub KV cache** of everything generated so far (`local_attn_size = 24`, `head_dim = 128`, tens of GB at full length).
- **Distilled, few-step.** The released causal checkpoint denoises in ~4 steps per block (no long 30-step schedule), with guidance 5.0. The text prompt is encoded once by a 7B **Cosmos-Reason1** encoder.
- **The VAE advances on a coarse clock.** Four pixel frames per latent interval is a useful response scale, not a strict lower bound on a frame-indexed detector. The measured raw decoder support identifies the exact alignment.
- **Latent, not pixel.** The DiT denoises 48-channel latents; the VAE decoder turns them into 720x1280 pixels per view at the very end.
""")

md(r"""
### Run a real rollout, live

*ELI5: actually start the learned game, take three "forward" steps, and time each one.*

The next cell invokes the real Gamma-World engine through its venv (`gamma_smoke.py`: build engine, step 3 blocks under a "forward" action, decode). It prints per-block generation time and the decoded video shape. Loading the 7B encoder + checkpoint takes ~30 s; each block is a few seconds on the session driver.
""")

code(r"""
smoke_json = RESULTS / "smoke" / "gamma_smoke.json"
if TORCHRUN.exists():
    cmd = [str(TORCHRUN), "--nproc_per_node=1", str(RTWM / "gamma_smoke.py"), "--blocks", "3"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", NVTE_FUSED_ATTN="0")
    print("running:", " ".join(cmd), "\n(this really loads and runs Gamma-World; ~1-2 min)\n", flush=True)
    p = subprocess.run(cmd, cwd=str(RTWM), env=env, capture_output=True, text=True, timeout=600)
    tail = "\n".join(p.stdout.splitlines()[-12:])
    print(tail)
    if p.returncode != 0:
        print("\n[stderr tail]\n", "\n".join(p.stderr.splitlines()[-8:]))
else:
    print("Gamma-World venv not found here; showing the real numbers measured on H200 instead.")

# Real numbers measured on H200, embedded so this cell works off the box (e.g. Colab).
GAMMA_REF = {"load_s": 30.6, "n_views": 2, "nfpb": 3, "num_blocks": 16,
             "per_block_ms": [1975.1, 2465.6, 3104.8], "video_shape": [1, 3, 378, 720, 1280]}
r = json.loads(smoke_json.read_text()) if smoke_json.exists() else GAMMA_REF
print("\n--- real rollout summary ---")
print(f"engine load: {r['load_s']} s | views: {r['n_views']} | frames/block: {r['nfpb']} | blocks total: {r['num_blocks']}")
print(f"per-block generation ms: {r['per_block_ms']}")
print(f"decoded video tensor shape (B,C,T,H,W): {r['video_shape']}")
print("\nThat is the real block-causal loop: one block = a few seconds of denoising against the rolling KV cache.")
""")

md(r"""
### What a real frame looks like

*ELI5: peek at actual generated frames.* We read committed rollout videos (two player views side by side, 720x2560).
""")

code(r"""
import imageio.v2 as iio

def load_frames(mp4, idxs):
    rd = iio.get_reader(str(mp4)); n = rd.count_frames()
    out = [(i, rd.get_data(min(i, n - 1))) for i in idxs]; rd.close()
    return out, n

def show_strip(mp4, idxs, title, view=0):
    if not Path(mp4).exists():
        print(f"[skipped: {Path(mp4).name} not present here -- rollout videos are gitignored and live on the H200 box]")
        return None
    frames, n = load_frames(mp4, idxs)
    fig, ax = plt.subplots(1, len(frames), figsize=(3.1 * len(frames), 2.6))
    if len(frames) == 1: ax = [ax]
    for a, (i, f) in zip(ax, frames):
        half = f.shape[1] // 2
        a.imshow(f[:, view * half:(view + 1) * half]); a.set_title(f"frame {i}", fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=11); plt.tight_layout(); plt.show()
    return n

mp4 = RESULTS / "serving_a2e" / "seed1" / "offline" / "generated.mp4"
n = show_strip(mp4, [0, 40, 96, 150], "Real Gamma-World rollout (player view 0): walking forward", view=0)
print("frames in this rollout:", n)
""")

# ---------------------------------------------------------------- CAOL intrinsic
md(r"""
## Metric: control-aligned onset latency (CAOL)

*ELI5: change a control, then count generated frames until the optical-flow detector sees a sustained motion change.*

We command a control change at a known frame and detect a future-baselined, half-amplitude **motion onset** in each rendered view. This is a detector-defined threshold crossing, not a validated first-perceptible effect. The VAE's four-frame temporal granularity is a comparison scale, not a strict floor.

Below: the real detector-onset table across 24 scenarios (2 scene types x 4 transitions x 3 change-frames x 2 views). Reproduce with:

```
Gamma-World/.venv/bin/torchrun --nproc_per_node=1 rtwm/a2e.py
```
""")

code(r"""
df = pd.read_csv(RESULTS / "a2e" / "a2e_results.csv")
detected = df[df["responded"] == 1]
print(f"view-level rows: {len(df)} | detected onsets: {len(detected)} | undetected: {len(df) - len(detected)}")
print(f"detector-defined onset lag (frames): mean {detected['a2e_frames'].mean():.1f} | "
      f"median {detected['a2e_frames'].median():.0f} | max {detected['a2e_frames'].max():.0f}")
print("\nby transition type (mean detected-onset lag in frames):")
print(detected.groupby('transition')['a2e_frames'].mean().round(1).to_string())

fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.hist(detected["a2e_frames"], bins=range(0, 22, 1), color="#4a7d4a", edgecolor="white")
ax.axvline(4, color="crimson", ls="--", label="VAE temporal granularity (4 frames)")
ax.set_xlabel("detector-defined onset lag (frames)"); ax.set_ylabel("count"); ax.legend()
ax.set_title("Real Gamma-World magnitude-detector onset lag")
plt.tight_layout(); plt.show()
""")

md(r"""
The takeaway that motivates the paper: detector-defined motion onsets occur on roughly the same scale as the model's four-frame latent clock, while some outcomes remain undetected. Serving adds a separately measurable control-admission lag, which we isolate next.
""")

# ---------------------------------------------------------------- serving CAOL
md(r"""
## The serving decomposition: generated-frame CAOL = admission + onset offset

*ELI5: in a real stream you can only hand the model new controls at a block boundary, so your key press waits for the next block. We separate that wait from when the optical-flow detector later sees a motion change.*

We drive "forward", then command "stop" at frame 96, and vary how many blocks late the action is delivered (`swap+0/1/2/4`), across 4 seeds. `swap+0` = admission at the boundary; `swap+k` = delivered k blocks late. The scorer assigns temporal labels:

- **post_admission** -- the mean-view onset occurred at or after delivery.
- **pre_admission** -- the onset occurred before the changed action arrived.
- **undetected** -- neither view produced a detector onset.

These labels describe timing, not causality: a post-admission onset can still
come from collision or model-driven dynamics.

Reproduce with:
```
Gamma-World/.venv/bin/torchrun --nproc_per_node=1 rtwm/streaming_swap.py
```
""")

code(r"""
sv = pd.read_csv(RESULTS / "serving_a2e" / "scale_summary.csv")
print(sv.to_string(index=False))

order = ["offline", "swap+0", "swap+1", "swap+2", "swap+4"]
classes = ["post_admission", "pre_admission", "undetected"]
colors = {"post_admission": "#4a7d4a", "pre_admission": "#d98a3d", "undetected": "#b03a3a"}
counts = {c: [int(((sv["arm"] == a) & (sv["cls"] == c)).sum()) for a in order] for c in classes}

fig, ax = plt.subplots(figsize=(7, 3.4))
bottom = np.zeros(len(order))
for c in classes:
    ax.bar(order, counts[c], bottom=bottom, label=c, color=colors[c]); bottom += np.array(counts[c])
ax.set_ylabel("rollouts (of 4 seeds)"); ax.set_title("Detector-onset timing vs admission delay (real Gamma-World)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("\nDelayed admission is associated with fewer post-admission and more pre-admission onsets.")
print("Matched unchanged-action controls are needed before attributing either timing category causally.")
""")

md(r"""
### Seeing onset timing

*ELI5: side by side, compare an on-time action with a two-block-late action.* Left is `swap+0`, right is `swap+2`, around the control change at frame 96; the strips illustrate timing but do not establish what caused the motion change.
""")

code(r"""
base = RESULTS / "serving_a2e" / "seed1"
for arm in ["swap+0", "swap+2"]:
    show_strip(base / arm / "generated.mp4", [90, 104, 120, 150],
               f"{arm}: stop commanded at frame 96 (view 0)", view=0)
""")

# ---------------------------------------------------------------- 3a speculation
md(r"""
## Track 3a: speculation via KV-cache fork

*ELI5: right before the player acts, secretly pre-draw the likely block. If they do that, show it instantly; if not, throw it away. To try a branch you must save-game the model's memory (fork the KV cache) and restore it.*

The catch is the fork cost. Gamma-World's sparse-hub KV cache is **~37 GB**, so a naive same-GPU clone OOMs; the first implementation snapshots to host RAM over PCIe, which costs **21 s** -- several times the block time it was meant to hide, so full-clone speculation is a *net loss*. The fix (the `driver_v2` **delta snapshot**) saves only the 3 index scalars per layer plus the one block the rolling window will evict, all on-GPU. Reproduce:

```
Gamma-World/.venv/bin/torchrun --nproc_per_node=1 rtwm/tracks.py        # full-clone (3a)
Gamma-World/.venv/bin/torchrun --nproc_per_node=1 rtwm/spec_delta.py    # delta snapshot (3a follow-up)
```
""")

code(r"""
tr = json.loads((RESULTS / "tracks" / "tracks_report.json").read_text())["3a"]
# spec_delta_report.json is gitignored; embed the real measured values as a fallback off-box.
_SPEC_DELTA_REF = {"on_demand_latency_ms": 4730.1, "fork_ms": 51.9, "restore_ms": 84.8,
                   "capture_ms": 7.2, "apply_ms": 6.6, "fork_snapshot_mb": 4730.2,
                   "err_apply_branch": 0.0}
_sd = RESULTS / "tracks" / "spec_delta_report.json"
dl = json.loads(_sd.read_text())["3a_delta"] if _sd.exists() else _SPEC_DELTA_REF

print("fork strategy       fork          restore/accept    snapshot     correctness")
print(f"full clone (host)   {tr['fork_ms']/1000:5.1f} s      {tr['restore_ms']/1000:5.2f} s          {tr['kv_cache_mb']/1024:5.1f} GB     exact")
print(f"delta (on-GPU)      {dl['fork_ms']:5.1f} ms     {dl['apply_ms']:5.1f} ms           {dl['fork_snapshot_mb']/1024:5.2f} GB     exact (err {dl['err_apply_branch']})")

on_demand = dl["on_demand_latency_ms"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
bars = ["on-demand\n(generate block)", "speculative accept\nfull-clone", "speculative accept\ndelta"]
vals = [on_demand, tr["restore_ms"], dl["apply_ms"]]
ax.bar(bars, vals, color=["#888", "#d98a3d", "#4a7d4a"])
for i, v in enumerate(vals): ax.text(i, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("action-to-display latency (ms)"); ax.set_yscale("log")
ax.set_title("Accepting a speculated block: delta fork hides CAOL, full-clone does not")
plt.tight_layout(); plt.show()
print(f"\nOn a hit, delta speculation shows the block in {dl['apply_ms']:.1f} ms vs generating it in {on_demand:.0f} ms.")
print("The full-clone variant's 21 s fork made it a net loss -- this is the same insight behind the")
print("paged copy-on-write fork contributed to vllm-omni (copy the block table, move zero tensor bytes).")
""")

# ---------------------------------------------------------------- 3b rollback
md(r"""
## Track 3b: rollback ("barge-in repair")

*ELI5: if you only notice a few blocks later that the world was using stale controls, rewind to the moment the control changed and redraw from there.*

We generate D blocks past a control change under stale conditioning, roll back to the switch, and re-denoise the tail under the new action. Repair cost is linear in D; the stale-vs-repaired latent difference grows with how late you noticed. Reproduce: `tracks.py` (3b).
""")

code(r"""
b3 = json.loads((RESULTS / "tracks" / "tracks_report.json").read_text())["3b"]
D = [r["D"] for r in b3]; rep = [r["repair_ms"] / 1000 for r in b3]; mse = [r["stale_vs_repaired_mse"] for r in b3]
fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
ax1.plot(D, rep, "o-", color="crimson", label="repair time (s)"); ax1.set_xlabel("discovery delay D (blocks)")
ax1.set_ylabel("repair time (s)", color="crimson"); ax1.set_xticks(D)
ax2 = ax1.twinx(); ax2.plot(D, mse, "s--", color="steelblue"); ax2.set_ylabel("stale vs repaired (latent MSE)", color="steelblue")
ax1.set_title("Rollback: repair cost and staleness both grow with how late you notice")
plt.tight_layout(); plt.show()
print("Both axes of the frontier are measurable: repair is ~linear in D; noticing later means more to redraw AND more stale frames already shown.")
""")

# ---------------------------------------------------------------- 3c adaptive
md(r"""
## Track 3c: adaptive compute

*ELI5: think hard only when it matters -- spend more denoising steps right after a control change, fewer when the scene is calm.*

We compare fixed schedules (4 / 2 / 1 steps per block) against an action-gated adaptive schedule (4 steps for 2 blocks after a change, 2 steps otherwise). Quality is latent MSE vs the 4-step reference. Reproduce: `tracks.py` (3c).
""")

code(r"""
c3 = json.loads((RESULTS / "tracks" / "tracks_report.json").read_text())["3c"]
names = ["steps4", "steps2", "steps1", "adaptive"]
t = [c3[n]["total_denoise_s"] for n in names]; q = [c3[n]["mse_vs_4step"] for n in names]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.scatter(t, q, s=80, color=["#4a7d4a", "#7d9d4a", "#b03a3a", "#3d6ad9"])
for n, x, y in zip(names, t, q):
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("total rollout denoise time (s)"); ax.set_ylabel("quality cost: latent MSE vs 4-step")
ax.set_title("Compute-quality frontier (real Gamma-World)")
plt.tight_layout(); plt.show()
print("Halving steps roughly halves generation time; the action-gated adaptive schedule buys near-2-step")
print("cost (48 s vs 42 s) at near-2-step quality while protecting the frames that define CAOL.")
""")

# ---------------------------------------------------------------- how served
md(r"""
## How it is served: the streaming session driver

*ELI5: the stock model only does one-shot generation; we wrote a driver that lets us step one block at a time and save/restore the memory, which is what every experiment above needs.*

`rtwm/driver_v2.py` (`BlockwiseSession`) replicates the real model's temporal-block loop as an externally steppable session, adding: per-block stepping, per-block action conditioning, per-block timing, per-block denoise-step count, and the KV-cache fork/restore/delta primitives. The math (denoise steps, re-noise to context, cache write) is copied from the upstream loop, so measurements transfer.
""")

code(r"""
_drv = RTWM / "driver_v2.py"
if _drv.exists():
    print(_drv.read_text().split("class BlockwiseSession")[0].split("INDEX_KEYS")[0][-900:])
else:
    print("[driver_v2.py not present here -- see it in the rtwm repo]")
print("... (full driver in rtwm/driver_v2.py: fork / restore / fork_delta / capture_branch / step_block / decode)")
""")

# ---------------------------------------------------------------- close
md(r"""
## Summary and pointers

*ELI5 recap:* detector-visible motion changes occur on roughly the model's latent temporal scale in the evaluated rollouts, while the serving system can add control-admission delay. Pre/post-admission labels describe timing, not what caused the onset. Speculation, rollback, and adaptive compute are systems levers for engineering this path.

- **Paper:** `rtwm/docs/master.tex` (arXiv preprint) -- this notebook reproduces its figures on the real checkpoint.
- **Serving primitives:** `rtwm/driver_v2.py` (session + fork/delta), `rtwm/streaming_swap.py` (mid-stream re-conditioning), `rtwm/tracks.py` and `rtwm/spec_delta.py` (the three tracks).
- **Systems connection:** the delta/copy-on-write fork is proposed upstream in vLLM-Omni PR #4909: fork paged KV state by copying block tables and bumping refcounts, moving zero tensor bytes at fork time.
- **Companion:** `world-infer-lab/notebooks/inference_pipeline_lab.ipynb` measures the serving-optimization ladder (steps, CFG, VAE, batching, cost) on real diffusion models.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3"}}
out = "rtwm_tutorial.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
