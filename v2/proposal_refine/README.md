# Single-H200 exact proposal/refine experiment

This directory implements an exact one-plus-three Gamma proposal transaction.
`propose` evaluates only runtime denoise step 0, retains the next noisy latent,
then restores the parent without committing context, advancing the block, or
touching decoder output. `accept_exact` requires the exact action-tensor hash,
unchanged parent identity, and unchanged runtime schedule before running steps
1–3 and committing once. `miss` discards the proposal and runs the complete
observed action.

## Result

The B=1 readiness gate failed and was not relaxed. The retained exact
shallow-conditioning arm measured first-step synchronized-host p95 latency of
1752.99 ms at first roll and 1754.88 ms at steady roll, versus the required
750 ms. B=2 was therefore not tested.

## Separately scoped positive pre-roll gate

`pre_roll_b1_gate` is intentionally narrower than the global rolling-cache
gate. On one H200 at block 0, 10 warmups and 30 randomized repetitions measured
an exact known-hit proposal p95 of 506.53 ms, passing the unchanged 750 ms
synthetic lead window. Exact 1+3 refinement saved 475.01 ms p50 / 484.20 ms p95
from action receipt to final commit and 475.22 ms p50 / 541.41 ms p95 to
synchronous uint8 host readiness versus on-demand full generation. Known-hit
and miss paths matched final latent, all live KV/index state, raw uint8, and a
one-block continuation with maximum error 0 and zero dropped frames.

The miss path charges both the abandoned pre-action proposal and full observed
fallback: its mean total charged work was 4730.50 ms versus 4282.88 ms for
on-demand generation. Frozen VPT confidence thresholds were fit with
deterministic five-fold cross-fitting on train episodes only. The frozen exact
B=1 operating threshold selected 14 of 582 pre-roll test decisions (2.4055%
coverage), with 6 exact hits (42.8571% precision; episode-bootstrap 95% CI
10.0%–68.75%). Those pre-roll decisions are only 2.0558% of all test decisions,
but they span action-block indices 1--7; readiness was measured only at block
0. A block-0-timing projection at the frozen threshold is 3440.75 ms worse than
all-on-demand over the eligible set. Thus exact block-0 proposal readiness
works, rolling-cache readiness does not, and no measured aggregate benefit is
established at the observed exact-hit coverage/precision.

Intent confidence remains diagnostic only: it is never used for exact
acceptance or CAOL because the semantic-fidelity gate has not passed. No
pre-roll latency is projected to first-roll or steady-roll boundaries.

## Approximate intent proposal arm

The confidence-gated B=1 intent arm failed its positive serving gate. A
hash-frozen 12-decision VPT test sample was selected before GPU scoring from 95
qualifying decisions where the train-only Markov prediction matched actual
intent but the representative and actual 12-frame hashes differed. It spans 12
intent tokens, both idle/movement states, and five camera-state groups.

On one H200, one representative-intent denoise step followed by three exact
actual-action refinement steps had worst held-out final-latent MSE 1.45747
(limit 0.024), four-block continuation MSE 2.69636 (limit 0.005), per-view
minimum raw PSNR 7.839/8.215 dB (minimum 30), and per-view minimum mean SSIM
0.68306/0.75868 (minimum 0.95). Raw output remained byte-exact before decoder
support in all 12 cases. No directional-obedience or perceptual-equivalence
claim is made.

The held-out proposal p95 was 1596.02 ms across pre-roll indices 1–7, so it
does not inherit the block-0 506.53 ms readiness result. Charging the
pre-action proposal leaves only 23.83 ms mean net saving on a hit and a
447.97 ms miss penalty, requiring 94.9487% break-even intent precision. No
train-only threshold had a one-sided 95% Wilson lower bound above break-even,
so the operating policy was frozen to reject all before test scoring. Held-out
coverage was therefore 0/582, charged work equaled all-on-demand
(2,492,638.63 ms), and the episode-bootstrap saved-work 95% interval was
[0, 0] ms. The positive scoped serving claim fails all three required
conditions: systems-sample readiness, semantic fidelity, and strictly lower
held-out charged work with a confidence interval excluding zero.

All 44 executable latent/output/live-KV/raw-uint8 checkpoints had byte-exact
results with maximum error 0. The four requested steady-roll eight-block checks
cannot execute with the stock checkpoint: a parent at block 9 plus eight
successors needs 51 latent normalization slots, while the tokenizer contains
50. No normalization values were extrapolated and no exact claim is made for
those checks.

The isolated fused-attention setting was faster in these measurements but is
not retained: its latent, output, and live-KV hashes differ from the unfused
baseline. Cross-attention projection caching and sparse-hub workspace
preallocation are unsupported by the loaded interfaces. Full-graph compile and
CUDA graph capture failed on dynamic cache mutation and conditioner RNG,
respectively.

Frozen VPT trace replay keeps exact sequence hashes separate from approximate
intent tokens. It charges proposals, abandoned proposals, refinement, commit,
miss fallback, incremental decode, host copy, and peak bytes. Proposal-ready,
final-commit-ready, and host-frame-ready are reported separately. Since even
rank-1 proposal p95 exceeds 750 ms, confidence/rank-gated trace readiness is
zero. CAOL is reported only as external canonical-STOP context, never
trace-wide. No intent semantic or perceptual-equivalence claim is made.

## Files

- `core.py`: exact transaction and deterministic logical noise reservation.
- `conditioning.py`: exact shallow-conditioning closure.
- `gpu_experiment.py`: baseline profiles, exact chains, and isolated probes.
- `timing_samples.py`: 10 warmups plus 30 randomized measurements per arm.
- `continuation_extension.py`: stock tokenizer-horizon preflight.
- `trace_replay.py`: frozen VPT empirical systems replay.
- `build_artifacts.py`: deterministic table, SVG, summary, patches, and manifest.
- `pre_roll_b1_benchmark.py`: measured pre-roll known-hit/miss event chains.
- `pre_roll_confidence.py`: train-only exact B=1 confidence frontier.
- `build_pre_roll_artifacts.py`: separate pre-roll summary/table/plot/manifest.
- `intent_refine.py`: explicitly approximate intent-then-actual transaction.
- `select_intent_heldout.py`: deterministic hash-frozen systems sample.
- `freeze_intent_confidence.py`: train-only break-even threshold freeze.
- `intent_semantic_benchmark.py`: H200 semantic and four-block continuation gate.
- `evaluate_intent_confidence.py`: frozen held-out charged-work evaluation.
- `build_intent_artifacts.py`: intent summary, CSVs, SVGs, and manifest.
- `results/summary.json`: concise pass/fail and measured results.
- `results/pre_roll_b1_summary.json`: separately scoped positive pre-roll gate.
- `results/intent_pre_roll_summary.json`: failed approximate-intent gate summary.
- `results/timings_by_arm.csv`: timing table.
- `results/proposal_p95.svg`: deterministic readiness plot.
- `results/manifest.json`: hashes, sizes, and project-relative retained paths.

## Reproduce

Use the Gamma virtual environment and one H200:

```bash
python -m pytest -q
PYTORCH_ALLOC_CONF=expandable_segments:True NVTE_FUSED_ATTN=0 python gpu_experiment.py --output results/h200_exact.json
PYTORCH_ALLOC_CONF=expandable_segments:True NVTE_FUSED_ATTN=0 python timing_samples.py --output results/timing_samples_unfused.json
PYTORCH_ALLOC_CONF=expandable_segments:True NVTE_FUSED_ATTN=1 python timing_samples.py --output results/timing_samples_fused.json
python trace_replay.py
python build_artifacts.py
python pre_roll_b1_benchmark.py
python pre_roll_confidence.py
python build_pre_roll_artifacts.py
python select_intent_heldout.py
python freeze_intent_confidence.py
PYTORCH_ALLOC_CONF=expandable_segments:True NVTE_FUSED_ATTN=0 python intent_semantic_benchmark.py
python evaluate_intent_confidence.py
python build_intent_artifacts.py
```

Results are atomically written. Long-running exact and trace scripts enforce
identity matching when resumed. No paper was edited and no commit was created.
