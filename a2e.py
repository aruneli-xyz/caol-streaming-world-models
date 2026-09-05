"""Experiment 1: intrinsic control-aligned onset latency (CAOL) of Gamma-World.

Renders rollouts whose command sequence steps at a known frame (stop,
reverse, turn, start), then measures a detector-defined RENDERED ego-motion
onset using dense optical flow on each player's view.

Response onset detection: per-frame mean flow magnitude for every transition,
baselined on the pre-change window. This is directly suited to start/stop but
can miss reverse/turn responses that change direction at similar speed;
onset = rendered-frame endpoint of the first inter-frame signal that crosses
halfway between pre-change and post-change steady-state medians and stays
there for 3+ frames. CAOL = onset - change_frame in generated-frame units.

The VAE compresses 4 pixel frames per latent frame. This is a temporal
granularity for comparison, not a strict floor on frame-indexed onset.

Usage:
  python a2e.py generate   # write eval samples
  python a2e.py render     # GPU
  python a2e.py score      # optical-flow onset measurement -> CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RTWM = Path(__file__).resolve().parent
SAFESWM = RTWM.parent / "safeswm"
sys.path.insert(0, str(SAFESWM))

import cv2
import numpy as np

from gamma.actions import write_sample_dir
from gamma.gamma_env import MODELS, REPO

N_FRAMES = 189
OUT = RTWM / "results" / "a2e"
FIRST_FRAMES = {
    "flat": REPO / "data" / "buildHouse_flat" / "first_frame.png",
    "normal": REPO / "data" / "buildTower_normal" / "first_frame.png",
}
PROMPT = "Two Minecraft players exploring the world"

# (name, pre-change command, post-change command)
TRANSITIONS = [
    ("stop", "forward", "stay"),
    ("reverse", "forward", "back"),
    ("turn", "forward", "left"),
    ("start", "stay", "forward"),
]
CHANGE_FRAMES = [48, 96, 144]


def make_samples() -> list[dict]:
    samples = []
    for frame_key in FIRST_FRAMES:
        for name, pre, post in TRANSITIONS:
            for cf in CHANGE_FRAMES:
                cmds = [pre] * cf + [post] * (N_FRAMES - cf)
                sample = f"a2e_{frame_key}_{name}_cf{cf}"
                # Both players get the same step change: doubles measurement
                # views and keeps the scenario symmetric.
                write_sample_dir(OUT / "eval" / sample / sample,
                                 FIRST_FRAMES[frame_key], PROMPT, [cmds, cmds])
                samples.append(dict(sample=sample, frame_key=frame_key,
                                    transition=name, change_frame=cf))
    (OUT / "manifest.json").write_text(json.dumps(samples, indent=1))
    print(f"{len(samples)} samples -> {OUT/'eval'}")
    return samples


def render() -> None:
    import subprocess

    for sample_dir in sorted((OUT / "eval").iterdir()):
        name = sample_dir.name
        out_dir = OUT / "out" / name
        if (out_dir / name / "generated.mp4").exists():
            continue
        cmd = [
            str(REPO / ".venv" / "bin" / "torchrun"), "--nproc_per_node=1",
            str(REPO / "scripts" / "inference.py"),
            "--mode", "causal_few_step", "--eval-dir", str(sample_dir),
            "--n-players", "2", "--num-frames", str(N_FRAMES),
            "--checkpoint", str(MODELS / "gamma-world" / "causal-few-step" / "model.safetensors"),
            "--vae", str(MODELS / "gamma-world" / "tokenizer.pth"),
            "--text-encoder", str(MODELS / "Cosmos-Reason1-7B"),
            "--output", str(out_dir),
        ]
        import subprocess as sp
        r = sp.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"render failed {name}: {r.stdout[-800:]}")
        print(f"rendered {name}")


def flow_signal(video: Path, view: int = 0) -> np.ndarray:
    """Mean optical-flow magnitude for one player's view.

    Signal index ``t`` is the inter-frame flow from rendered frame ``t`` to
    rendered frame ``t + 1``.
    """
    cap = cv2.VideoCapture(str(video))
    mags, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw = frame.shape[1] // 2
        gray = cv2.cvtColor(frame[:, view * vw:(view + 1) * vw], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (240, 160))
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=-1)
            mags.append(float(mag[20:-40].mean()))  # crop sky edge + HUD
        prev = gray
    cap.release()
    return np.array(mags)


def onset_frame(signal: np.ndarray, change: int, rising: bool) -> int | None:
    """Rendered-frame endpoint of the first sustained midpoint crossing."""
    pre = np.median(signal[max(0, change - 24):change - 2])
    post = np.median(signal[min(len(signal) - 1, change + 16):min(len(signal), change + 40)])
    if abs(post - pre) < 0.15 * max(pre, post, 1e-6):
        return None  # no detectable steady-state change -> no response
    mid = (pre + post) / 2
    for t in range(change, min(len(signal) - 3, change + 40)):
        window = signal[t:t + 3]
        crossed = (window > mid).all() if post > pre else (window < mid).all()
        if crossed:
            # signal[t] spans rendered frames t -> t+1, so the rendered frame
            # at which this motion first becomes available is t+1.
            return t + 1
    return None


def score() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = []
    for m in manifest:
        video = OUT / "out" / m["sample"] / m["sample"] / "generated.mp4"
        if not video.exists():
            continue
        for view in (0, 1):
            sig = flow_signal(video, view)
            rising = m["transition"] == "start"
            onset = onset_frame(sig, m["change_frame"], rising)
            rows.append(dict(**m, view=view,
                             onset=onset,
                             a2e_frames=(onset - m["change_frame"]) if onset else None,
                             responded=int(onset is not None)))
    with open(OUT / "a2e_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"n={len(df)} measurements")
    print("\n== magnitude-detectable response rate and CAOL by transition ==")
    g = df.groupby("transition").agg(respond_rate=("responded", "mean"),
                                     a2e_frames=("a2e_frames", "mean"),
                                     a2e_std=("a2e_frames", "std")).round(2)
    print(g.to_string())
    print("\n== by change frame ==")
    print(df.groupby("change_frame").agg(respond_rate=("responded", "mean"),
                                         a2e_frames=("a2e_frames", "mean")).round(2).to_string())
    print("\n(VAE temporal granularity = 4 pixel frames per latent frame)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "render", "score", "all"])
    args = ap.parse_args()
    if args.stage in ("generate", "all"):
        make_samples()
    if args.stage in ("render", "all"):
        render()
    if args.stage in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
