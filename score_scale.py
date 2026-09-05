"""Score the multi-seed serving-CAOL grid (seeds x arms)."""

import json
import sys
from pathlib import Path

import numpy as np

RTWM = Path(__file__).resolve().parent
sys.path.insert(0, str(RTWM))
from a2e import flow_signal  # noqa: E402

BASE = RTWM / "results" / "serving_a2e"


def onset_wide(signal, change, search_to):
    pre = np.median(signal[max(0, change - 24):change - 2])
    post = np.median(signal[search_to + 8:search_to + 32])
    if abs(post - pre) < 0.15 * max(pre, post, 1e-6):
        return None
    mid = (pre + post) / 2
    for t in range(change, min(len(signal) - 3, search_to + 40)):
        w = signal[t:t + 3]
        ok = (w < mid).all() if post < pre else (w > mid).all()
        if ok:
            return t + 1  # signal[t] is flow from rendered frame t to t+1
    return None


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    rows = []
    for m in manifest:
        video = BASE / f"seed{m['seed']}" / m["arm"] / "generated.mp4"
        if not video.exists():
            continue
        sw = m["effective_switch_px"]
        onsets = [onset_wide(flow_signal(video, v), m["change_frame"], sw) for v in (0, 1)]
        o = [x for x in onsets if x is not None]
        onset = float(np.mean(o)) if o else None
        # Temporal classification only: timing does not establish that the
        # admitted action caused a post-admission onset.
        if onset is None:
            cls = "undetected"
        elif onset < sw:
            cls = "pre_admission"
        else:
            cls = "post_admission"
        rows.append(dict(arm=m["arm"], seed=m["seed"], switch_px=sw,
                         view_onsets=onsets, onset=onset,
                         cls=cls,
                         perceived=(onset - 96) if cls == "post_admission" else None,
                         post_switch=(onset - sw) if cls == "post_admission" else None))
    (BASE / "scale_results.json").write_text(json.dumps(rows, indent=1))

    import pandas as pd
    df = pd.DataFrame(rows)
    order = ["offline", "swap+0", "swap+1", "swap+2", "swap+4"]
    print("== classification counts ==")
    print(df.pivot_table(index="arm", columns="cls", values="seed",
                         aggfunc="count", fill_value=0).reindex(order).to_string())
    print("\n== post-admission onsets only ==")
    c = df[df["cls"] == "post_admission"]
    g = c.groupby("arm").agg(n=("seed", "count"),
                             perceived_mean=("perceived", "mean"),
                             perceived_std=("perceived", "std"),
                             post_switch_mean=("post_switch", "mean")).round(2)
    print(g.reindex([a for a in order if a in g.index]).to_string())
    df.to_csv(BASE / "scale_summary.csv", index=False)


if __name__ == "__main__":
    main()
