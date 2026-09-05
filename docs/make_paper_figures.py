"""Generate the result figures for master.tex from the measured numbers.

Values mirror rtwm/results/serving_a2e/ and rtwm/results/tracks/. Run:
    python make_paper_figures.py
Outputs land in figures/.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

GREEN, ORANGE, GRAY, RED = "#2d6a4f", "#ca6702", "#6c757d", "#ae2012"

# ---------------------------------------------------------------- fig: classification
# Temporal onset classification, 4 seeds per arm (results/serving_a2e/).
arms = ["offline", "delay 0", "delay 1", "delay 2", "delay 4"]
post_admission = np.array([4, 3, 2, 0, 0])
pre_admission = np.array([0, 0, 2, 3, 3])
undetected = 4 - post_admission - pre_admission

fig, ax = plt.subplots(figsize=(6.2, 2.9))
x = np.arange(len(arms))
ax.bar(x, post_admission, 0.62, label="post-admission onset", color=GREEN)
ax.bar(x, pre_admission, 0.62, bottom=post_admission,
       label="pre-admission onset", color=RED)
ax.bar(x, undetected, 0.62, bottom=post_admission + pre_admission,
       label="undetected", color=GRAY, alpha=0.55)
ax.set_xticks(x, arms)
ax.set_ylabel("seeds (of 4)")
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_title("Onset timing by control-admission arm (4-seed pilot)")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout()
fig.savefig("figures/classification.png", bbox_inches="tight")
print("wrote figures/classification.png")

# ---------------------------------------------------------------- fig: method frontiers
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.2))

# (a) fork cost per branch, log scale (tab:fork values)
ax = axes[0]
mechs = ["host snapshot", "device clone", "delta snapshot",
         "paged deep copy", "paged CoW"]
fork_ms = [21400, 19.2, 52, 3.2874, 0.005013]
annotations = [
    "21.4 s; 37 GiB copied",
    "19.2 ms; 37 GiB duplicated",
    "52 ms; 0–4.6 GiB copied",
    "3.29 ms; 5.36 GiB, n=50",
    "5.01 $\\mu$s/child; 0 KV bytes, n=1000",
]
ys = np.arange(len(mechs))
bars = ax.barh(ys, fork_ms, 0.58, color=[GRAY, GRAY, ORANGE, GRAY, GREEN])
ax.set_xscale("log")
ax.set_yticks(ys, mechs)
ax.invert_yaxis()
ax.set_xlabel("fork time (ms, log scale)")
ax.set_title("(a) Speculation: fork cost")
ax.tick_params(axis="y", labelsize=7.5)
for b, m in zip(bars, annotations):
    ax.text(b.get_width() * 1.35, b.get_y() + b.get_height() / 2, m,
            va="center", ha="left", fontsize=6.5, color="#333")
ax.set_xlim(1e-3, 2e6)
ax.grid(axis="x", which="major", color="#ddd", lw=0.5)

# (b) rollback frontier (tracks_report 3b)
ax = axes[1]
D = [1, 2, 4]
repair_s = [12.0, 19.8, 35.3]
mse = [0.00068, 0.00083, 0.00121]
ax.plot(D, repair_s, "o-", color=GREEN, label="repair cost (s)")
ax.set_xlabel("discovery delay D (blocks)")
ax.set_ylabel("repair cost (s)", color=GREEN)
ax.set_xticks(D)
ax2 = ax.twinx()
ax2.plot(D, mse, "s--", color=ORANGE, label="stale divergence")
ax2.set_ylabel("stale vs repaired MSE", color=ORANGE)
ax2.spines.top.set_visible(False)
ax.set_title("(b) Rollback: repair vs staleness")

# (c) schedule frontier (tracks_report 3c)
ax = axes[2]
names = ["4 steps", "2 steps", "1 step", "action-gated"]
times = [82.5, 41.9, 21.5, 48.1]
mses = [0.0, 0.024, 0.074, 0.023]
colors = [GRAY, GRAY, GRAY, GREEN]
offsets = [(-14, 8), (-20, 9), (6, 3), (6, -11)]
for n, t, m, c, off in zip(names, times, mses, colors, offsets):
    ax.scatter(t, m, s=55, color=c, zorder=3)
    ax.annotate(n, (t, m), textcoords="offset points", xytext=off, fontsize=8,
                color=c if c == GREEN else "#333")
ax.plot(times[:3], mses[:3], "--", color=GRAY, lw=1, alpha=0.6, zorder=1)
ax.set_xlabel("full-rollout denoise time (s)")
ax.set_ylabel("latent MSE vs 4-step")
ax.set_title("(c) Adaptive compute: latent-error frontier")
ax.set_xlim(15, 95)

fig.tight_layout(w_pad=2.5)
fig.savefig("figures/method_frontiers.png", bbox_inches="tight")
print("wrote figures/method_frontiers.png")
