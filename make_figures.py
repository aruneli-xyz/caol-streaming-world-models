"""Generate manuscript figures from measured data. CPU-only.

  fig 1  detector-defined CAOL by transition (bars + onset rate)
  fig 2  serving CAOL decomposition (pilot): perceived latency vs
         scheduling delay with the intrinsic+scheduling prediction line
  fig 3  flow-signal trace around the command change (offline arm)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RTWM = Path(__file__).resolve().parent
FIG = RTWM / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- fig 1: detector-defined CAOL ----------------------------------------
df = pd.read_csv(RTWM / "results" / "a2e" / "a2e_results.csv")
g = df.groupby("transition").agg(rate=("responded", "mean"),
                                 a2e=("a2e_frames", "mean"),
                                 std=("a2e_frames", "std"))
order = ["stop", "start", "reverse", "turn"]
g = g.reindex(order)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(order, g["a2e"], yerr=g["std"], color="#4a7d4a", capsize=4)
axes[0].axhline(4, color="red", ls="--", lw=1,
                label="VAE temporal granularity (4 frames)")
axes[0].set_ylabel("magnitude-detectable onset lag (pixel frames)")
axes[0].set_title("Detector-defined onset lag by transition")
axes[0].legend(fontsize=8)
axes[1].bar(order, g["rate"], color="#6a9fb5")
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("magnitude-detectable onset rate")
axes[1].set_title("Fraction detected by flow magnitude")
for i, v in enumerate(g["rate"]):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "intrinsic_a2e.png", dpi=150)
print("fig 1 done")

# ---- fig 2: serving CAOL decomposition (pilot) ---------------------------
classified = RTWM / "results" / "serving_a2e" / "scale_results.json"
rows = json.loads(classified.read_text())
post_admission = pd.DataFrame(r for r in rows if r["cls"] == "post_admission")
g_serving = post_admission.groupby("arm").agg(
    switch_px=("switch_px", "first"),
    perceived=("perceived", "mean"),
    std=("perceived", "std"),
)
arm_order = [a for a in ("offline", "swap+0", "swap+1") if a in g_serving.index]
g_serving = g_serving.reindex(arm_order)
xs = (g_serving["switch_px"] - 96).to_numpy()
ys = g_serving["perceived"].to_numpy()
stds = g_serving["std"].fillna(0).to_numpy()
labels = {"offline": "offline", "swap+0": "delay 0", "swap+1": "delay 1"}

fig, ax = plt.subplots(figsize=(6, 4))
intrinsic = 9
line_x = np.array([0, max(xs) + 4])
ax.plot(line_x, intrinsic + line_x, "k--", lw=1,
        label=f"observed post-admission lag ({intrinsic}) + admission lag")
ax.errorbar(xs, ys, yerr=stds, fmt="o", ms=8, capsize=4,
            color="#4a7d4a", zorder=3)
offsets = {"offline": (7, 7), "swap+0": (7, -13), "swap+1": (7, -4)}
for arm, x, y in zip(arm_order, xs, ys):
    ax.annotate(labels[arm], (x, y), textcoords="offset points",
                xytext=offsets[arm], fontsize=9)
ax.set_xlabel("control-admission lag (pixel frames)")
ax.set_ylabel("mean-view CAOL_frames (control change to onset)")
ax.set_title("Control-admission pilot (forward$\\to$stop, 4 seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "serving_a2e.png", dpi=150)
print("fig 2 done")

# ---- fig 3: flow trace ---------------------------------------------------
sys.path.insert(0, str(RTWM))
from a2e import flow_signal  # noqa: E402

base = RTWM / "results" / "serving_a2e"
video = base / "offline" / "generated.mp4"
if video.exists():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for view, color in ((0, "#4a7d4a"), (1, "#6a9fb5")):
        s = flow_signal(video, view)
        ax.plot(s, color=color, lw=1.2, label=f"player {view} view")
    ax.axvline(96, color="red", ls="--", lw=1, label="commanded stop (frame 96)")
    ax.set_xlabel("pixel frame")
    ax.set_ylabel("mean optical-flow magnitude")
    ax.set_title("Rendered ego-motion around a commanded stop (offline arm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "flow_trace.png", dpi=150)
    print("fig 3 done")
