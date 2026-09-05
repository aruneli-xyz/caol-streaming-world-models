# RTWM v2 findings

Status: canonical STOP, exact incremental timing, and canonical directional
artifacts are complete. All legacy offline 48-view, four-seed serving,
confirmatory-throughput, and full-buffer frame-ready action claims used an
incompatible helper or superseded chain and are invalidated. Numbers in the
archival diagnosis below explain rejection only; they are not evidence.

## Invalidated exploratory run (diagnosis only)

- Gamma-World commit: `6a95de85c439d8ea73eae34c88fbfd4e89ea02e2`
  with local instrumentation present (`dirty: true` recorded in manifest).
- Scene: `buildTower_normal`.
- Seeds: 1-4.
- Arms per seed: unchanged-forward control, stop at delay 0, stop at delay 1.
- One duplicate forward-control canary.
- 13 completed rollouts, 3,528 seconds total measured generation time
  (271 seconds/rollout mean).
- 8 matched scene-seed-delay pairs and 16 paired views.

## Validation gates

Passed:

- all synthetic detector sign/index tests;
- expected frame counts;
- zero null-control false positives under the pre-specified stop detector;
- duplicate control MP4 hash and decoded pixels are exactly equal;
- the targeted pre-decode latent diagnostic has exact matched prefixes.

Failed:

- decoded pixel prefixes are not equal before nominal action admission.

Stage 2 (`buildHouse_flat`) was therefore not run.

## Decoder and codec diagnosis

The exploratory MP4 pairs appeared to diverge long before nominal admission:

- delay 0 (`f_a=96`): earliest difference at rendered frame 54;
- delay 1 (`f_a=108`): earliest difference at frame 66 or 70.

Pair-level pre-admission maximum pixel errors range from 16 to 58 grayscale
levels after the detector resize. The targeted latent rerun confirms exact
matched prefixes, so this is not stochastic replay failure.

The targeted seed-1, delay-0 latent rerun resolves the ambiguity:

- latent shape: `[1, 16, 2, 48, 90, 160]`;
- maximum latent error before admission latent 24: `0.0`;
- first difference in both views: latent 24 exactly;
- full-tail maximum latent difference: `1.796875`.

The later raw-decoder sweep resolves the source of the apparent look-ahead:

- 24 hybrid latent probes have zero changed raw pixels before expected
  support;
- latent 24 first affects rendered frame 93;
- latent 27 first affects rendered frame 105;
- latent 30 first affects rendered frame 117;
- support follows `4*t-3` for latent index `t>0`;
- repeated raw decoding is bit-exact;
- MP4 reconstruction moves detected differences 39 frames earlier.

Thus the VAE's raw temporal support is measurable and monotone. The severe
early divergence was introduced by MP4 reconstruction, not raw decoder
look-ahead. Confirmatory scoring therefore uses raw pre-codec frames and
begins each search at measured support.

## Invalidated exploratory paired score

The pre-specified paired detector uses
`flow_magnitude(control) - flow_magnitude(stop)` with a minimum threshold of
0.03 pixels/frame. It detects no paired onset in 0/16 views.

Observed maximum post-admission paired contrasts are approximately
0.001-0.004 pixels/frame, well below the frozen threshold. Individual
pre-only stop detectors are also indeterminate because the forward-motion
baseline is only about 0.01 pixels/frame. These are null/indeterminate
results under the pre-specified detector, not evidence that STOP has no
latent effect: the latent diagnostic shows post-admission divergence.

## Invalidated legacy directional ablation

On the existing v1 videos:

- start: v2 detects 12/12 views;
- stop: 2/12;
- reverse: 0/12;
- left: 0/12.

Late-window sign checks are consistent for start (12/12) and mostly for
stop (10/12), but weak for reverse radial flow (7/12) and left horizontal
flow (5/12). The fixed directional detector is retained as a failed
ablation; it is not retuned on these evaluated outcomes.

The bounded held-out rerun uses change frames 48/96 for an audit and frame 144
for validation. Synthetic signs pass, but held-out onset detection is 0/4
views for reverse and 0/4 for left. Reverse has the expected radial sign in
3/4 held-out views; left has the expected horizontal sign in only 1/4. These
detectors are not supported for response claims, so the confirmatory causal
study remains forward-to-stop only.

## Confirmatory protocol

The confirmatory redesign uses:

- two scenes: `buildTower_normal` and `buildHouse_flat`;
- null calibration seeds 101-103 and held-out null-validation seed 104;
- test seeds 201-206;
- unchanged-forward controls paired with stop interventions admitted at
  latent 27 or 30 (rendered admission frames 108 and 120);
- direct dense flow-field L2 divergence, rather than a difference of scalar
  magnitudes;
- four immutable scene/view threshold locks and zero held-out strict
  crossings;
- exact latent-prefix, raw-support, artifact-hash, source, and protocol gates.

## Canonical STOP result

Source: `results/canonical_stop_scoring/`. All 24 planned
scene-seed-delay pairs pass with no invalid pairs. The canonical action
protocol, source/config/protocol identities, fresh controls/nulls, and scorer
outputs are hash-bound. Treating seed as the statistical unit:

- both-view and any-view paired detection are 100% for both policies;
- the 12-frame admission policy has mean paired onset lag 9.0 frames
  (95% seed-bootstrap interval `[9.0, 9.0]`);
- the 24-frame policy has mean lag 21.0 frames (`[21.0, 21.0]`);
- the paired delay difference is exactly 12.0 frames across all six seeds;
- mean peak flow-field divergence is 0.0172288 and 0.0171567 detector-grid
  pixels/frame;
- measured effective generation throughput is 0.834309 FPS
  (`[0.831234, 0.837557]`) and 0.835026 FPS
  (`[0.831762, 0.838335]`);
- the relative throughput difference is 0.085889%, inside the fixed 5%
  comparability check.

The causal detector begins at measured raw support. Latent admissions 27 and
30 have nominal pixel coordinates 108 and 120 but first raw support at frames
105 and 117. The first eligible flow intervals enter those support frames,
giving control-aligned lags 9 and 21 from frame 96.

## Exact incremental wall-clock result

Source: `results/incremental_decode/stop_wallclock_chain_v1/`. Two
counterbalanced seeds per policy record action receipt, admission, denoise,
context commit, cached incremental decode, device readiness, synchronous host
transfer, and callback consumption:

- 12-frame admission: action-to-host-ready 21,476.2±29.2 ms;
- 24-frame admission: action-to-host-ready 32,930.3±958.3 ms.

The ± values are descriptive sample SD with `n=2`. All four runs are exact
against full-prefix raw uint8 decoding in both views (maximum error 0), with
zero dropped frames. The host callback is synchronous software consumption,
not physical display presentation.

## Canonical directional result

Source: `results/directional_d0_held_out/`. Fresh canonical controls, nulls,
and 24/24 valid interventions cover forward-to-back and forward-to-true-yaw.
Held-out paired causal divergence is 100% in both scenes for both transitions.
The frozen signed-direction audit gate fails, so directional obedience
explicitly fails and only causal divergence is claimable.

## Speculation frontier

The frozen VPT corpus has 597 episodes (512 train, 85 test), 28,310 eligible
test blocks, and 21,397 intent changes. Markov-1 at `B=4` hits 59.1381%
`[56.7646, 61.4038]` overall and 47.3197%
`[44.4826, 50.1101]` on changes. Strict hashes hit 9.6044% overall and
2.0984% on changes. Measured candidate blocks take about 4.70–4.72 s;
89.764 s of GPU work is charged across eight scenarios and each captured
branch state is 4.9656 GB. Readiness is 0% under a 750 ms lead even for
projected capacities `C={1,2,4}`. This is hit/readiness/compute/memory only,
not arbitrary-action CAOL; paged COW is separate.

## Completed proposal/refinement result

Source: `proposal_refine/results/`. The exact uncommitted `B=1` transaction
runs denoise forward 1 before action arrival, retains the next noisy latent,
and restores the parent without context commit. An exact action-hash hit runs
forwards 2–4 and commits once; a miss discards the proposal and runs all four
forwards on demand.

- All 44 executable latent, output, live-KV/index, and raw-uint8 checkpoints
  are exact with maximum error 0.
- The steady-roll +8 requested continuation cannot execute with the stock
  tokenizer: block 9 plus eight successors needs 51 latent normalization
  slots, while the checkpoint has 50. Therefore not every requested gate
  passed.
- At the unsaturated block-0 boundary, exact proposal p50/p95 is
  500.74/506.53 ms and the scoped 750 ms gate passes.
- On a known exact hit, p95 action-to-final-commit saving is 484.20 ms and p95
  action-to-synchronous-host-ready saving is 541.41 ms. Exact hit and miss
  paths have zero errors and dropped frames.
- The positive readiness measurement is block 0 only. The trace contains
  582/28,310 decisions (2.06%) at unsaturated indices 1–7, but readiness at
  those later boundaries was not measured. Exact first-/steady-roll proposal
  p95 is 1752.99/1754.88 ms; the global gate fails and `B=2` is untested.
- The frozen train-only exact-confidence operating point has 2.41% test
  coverage and 42.86% exact-hit precision. Applying block-0 timing to those
  later trace decisions projects +3440.75 ms charged work versus
  all-on-demand; it is not a measured aggregate benefit.
- The approximate-intent arm has p95 1596.02 ms. Final-latent MSE,
  four-block-continuation MSE, per-view PSNR, and per-view SSIM gates all fail;
  the frozen policy rejects all held-out proposals. No perceptual, semantic,
  directional, or CAOL claim is supported.
- Fused attention was faster in isolation but byte-inexact across settings and
  was rejected. Full-graph compile and CUDA graph capture failed;
  cross-attention projection caching and sparse-hub workspace preallocation
  were unsupported by the loaded interfaces.

The measured bottleneck is rolling sparse-hub forward compute, not fork or
state capture. The earlier approximately 0.94 s stock measurement was one
denoise forward, not a complete candidate; a complete candidate requires four
denoise forwards plus context commit. No general profitable speedup is
established. The positive result is the exact transaction and narrow pre-roll
boundary; the optimization result is the identified rolling bottleneck.

## Human validation

The tool/QC infrastructure has 48 canonical primary intervention views and
16 QC items. No annotators have participated; human validation remains
pending.
