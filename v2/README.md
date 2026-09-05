# RTWM v2: Matched-Counterfactual Responsiveness

This directory is isolated from the frozen v1 preprint. v2 tests whether a
control change causes a rendered motion divergence by comparing paired
Gamma-World rollouts with identical scene, seed, prefix actions, admission
time, and inference schedule.

## Primary estimand

For each confirmatory scene, seed, admission policy, and view:

- intervention: `forward -> stay` at pixel frame 96;
- control: `forward -> forward` for all 189 frames;
- admission frame: 108 (`d0`) or 120 (`d1`), corresponding to 12 or
  24 frames after the aligned command;
- paired signal: trimmed-mean L2 disagreement between the aligned dense
  optical-flow vector fields.

The primary paired onset is the rendered-frame endpoint of the first three
consecutive paired-flow samples strictly above a scene/view threshold. Each
threshold is fit from disjoint duplicate null rollouts, frozen, and validated
on a held-out null seed before test generation. Results are reported per view,
scene-seed pair, and seed. Views and scenes are repeated observations, not
independent seeds.

## Detector specification

All detectors use Farneback flow with:

`pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5,
poly_sigma=1.2, flags=0`.

Frames are split into horizontal player views, converted to grayscale, and
resized to 240x160 using `INTER_AREA`. The region of interest is
`x=[12,228), y=[20,120)`. Spatial summaries use a 10% trimmed mean.

Signals:

- start/stop: flow magnitude;
- reverse: radial flow, positive for expansion and negative for contraction;
- left/right: signed horizontal flow.

For a signal with pre-change indices `[change-24, change-2)`:

`baseline = median(pre)`

`sigma = 1.4826 * MAD(pre)`

`delta = max(4*sigma, 0.15*abs(baseline), 0.03)`

Transition targets are fixed in `config/confirmatory.json`. No detector parameter
is selected from evaluated transition outcomes. v1 future-midpoint
magnitude scoring is retained as an explicit ablation.

## Indexing

- rendered frames: `0..F-1`;
- action index 96 is the first post-transition action;
- flow sample `t` spans rendered frames `t -> t+1`;
- a crossing at sample `t` reports onset frame `t+1`;
- raw decoder support for latent index `t>0` begins at rendered frame
  `4*t-3`;
- confirmatory searches begin at the measured raw-support frame, not at the
  nominal command frame;
- MP4 is diagnostic only because codec reconstruction moved differences
  earlier than raw decoder support.

## Confirmatory stages

The exploratory pilot remains archived as a failed protocol iteration. The
confirmatory protocol uses both `buildTower_normal` and `buildHouse_flat`:

- calibration: null-duplicate seeds 101-103;
- held-out null validation: seed 104;
- confirmatory test: seeds 201-206, one unchanged-forward control and two
  stop-admission policies per scene.

## Validation gates

- 189 raw decoded frames and even side-by-side width;
- finite flow coverage above 99.9%;
- all synthetic sign/index tests pass;
- all four scene/view calibration locks exist;
- zero strict crossings on held-out null validation;
- exact latent prefixes before admission;
- no raw pixel difference before measured decoder support;
- exact artifact, config, protocol, source, checkpoint, and tokenizer hashes;
- view onset disagreement above eight frames is flagged, never discarded.

## Files

- `config/pilot.json`: frozen protocol and schemas;
- `flow_detector.py`: transition-aware detector;
- `test_flow_detector.py`: synthetic sign/index tests;
- `score_detector.py`: v1/v2 detector ablations;
- `validate_directional_detector.py`: bounded held-out reverse/turn audit;
- `matched_counterfactual.py`: resumable Gamma-World generation;
- `score_matched.py`: paired causal scoring;
- `score_confirmatory.py`: validity-first confirmatory scoring and seed-level
  uncertainty;
- `incremental_decode/`: exact cached incremental decode and wall-clock chain;
- `results/`: manifests, scores, summaries, and trace plots.

Generated videos and tensor dumps remain ignored by git. JSONL/CSV
summaries and plots are retained.

## Current status

Legacy offline and four-seed serving results used an incompatible action helper
and are invalidated; they are not evidence. Canonical STOP scoring at
`results/canonical_stop_scoring/` retains all 24 planned pairs, with both/any
detection at 100%, onset lags of 9 and 21 frames, continuous effects
0.0172288 and 0.0171567 detector-grid px/frame, and effective throughput
0.834309 `[0.831234, 0.837557]` versus 0.835026
`[0.831762, 0.838335]` (0.085889% relative difference).

The exact incremental chain at
`results/incremental_decode/stop_wallclock_chain_v1/` reports
21,476.2±29.2 ms and 32,930.3±958.3 ms action-to-host-ready (`n=2`, sample
SD), raw-uint8 equivalence in both views with maximum error 0, and no drops.
The host callback is not physical presentation.

Fresh canonical back and true-yaw data contain 24/24 valid interventions and
100% held-out causal divergence in both scenes. The frozen signed-direction
audit gate fails, so directional obedience explicitly fails and only causal
divergence is claimable. Human annotation infrastructure has 48 primary views
and 16 QC items, but no annotators; human validation is pending. Artifacts and
limitations are detailed in `FINDINGS.md`.

The completed `proposal_refine/` study implements an exact uncommitted
one-plus-three transaction: propose with denoise forward 1, retain the noisy
latent without context commit, then refine with forwards 2–4 on an exact hit
or discard and run all four forwards on a miss. All 44 executable checkpoints
have maximum error 0. The requested steady-roll +8 continuation is blocked by
the stock tokenizer horizon (51 normalization slots required, 50 available),
so not every requested gate passed.

The scoped block-0 `B=1` pre-roll gate passes: proposal p50/p95 is
500.74/506.53 ms against 750 ms; a known exact hit saves p95 484.20 ms to
final commit and 541.41 ms to synchronous host readiness, with zero errors or
drops. This readiness result is measured only at block 0. The trace contains
582/28,310 decisions (2.06%) at unsaturated indices 1–7, but readiness at those
later boundaries was not measured. First-/steady-roll exact proposal p95 is
1752.99/1754.88 ms, so the global gate fails and `B=2` is untested.

The train-only exact confidence gate covers 2.41% at 42.86% precision; a
block-0-timing projection adds 3440.75 ms expected charged work versus
all-on-demand. The approximate-intent
arm has p95 1596.02 ms, fails latent MSE, PSNR, SSIM, and continuation gates,
and is frozen to reject all; it supports no perceptual, semantic, directional,
or CAOL claim. Fused attention was faster but byte-inexact and rejected;
compile/CUDA graph arms failed, and cross-attention/workspace arms were
unsupported. Rolling sparse-hub forward compute—not fork/capture—is the
measured bottleneck. The earlier approximately 0.94 s stock observation was
one denoise forward; a complete candidate is four forwards plus context
commit. No general profitable speedup is established.
