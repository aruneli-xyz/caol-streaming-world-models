# Project notes (internal lab log, kept for context)

These are the working notes that accumulated during the project. They are
more informal than the paper and mention sibling checkouts (`../safeswm`)
used on the original machine. The paper (`master.tex`) and `v2/FINDINGS.md`
are the authoritative statements of results.


## The shared foundation (built first)

**Control-aligned onset latency (CAOL)**: operationalized here as generated-frame
lag from a control change to a detector-defined rendered motion onset.
This is distinct from frame rate and from end-to-end wall-clock latency.
The serving study separates control-admission lag from post-admission
onset lag; temporal onset labels do not by themselves establish causality.

Two components:
- **Zero-admission-lag onset measurement**: detector-defined motion-onset
  lag when the full action sequence is available upfront. This is not
  identified as the model's causal response delay.
- **Serving decomposition**: control-admission lag plus onset offset from
  admission under chunked streaming. The offset may be negative for a
  pre-admission onset; temporal labels do not establish causality.

## Method tracks (implement all, keep what moves the needle)

- **3a — Action-Tube Speculative Streaming**: per chunk, generate a small
  set of latent futures under likely action continuations (branch budget
  over forward/stop/turn/retreat); when the real action arrives, accept
  the nearest branch, repair moderate mismatches with a cheap latent
  correction, regenerate only on large mismatch. Speculative decoding for
  action-conditioned rollouts.
- **3b — Mid-chunk re-conditioning ("barge-in")**: on an action change
  mid-chunk, invalidate the suffix of the rolling KV cache, re-denoise
  only the affected tail with the updated action embedding, splice.
  Research questions: how deep can you re-condition before quality
  collapses; the rollback-depth / latency / fidelity frontier; effect of
  distillation step count. Imported discipline: barge-in handling from
  real-time conversational audio.
- **3c — Authority-adaptive compute**: cheap latent readouts (Paper 2's
  probes) gate per-chunk compute — fewer denoising steps for stable
  low-interaction chunks, full compute (or earlier decode) when control
  criticality or hazard risk is high. Foveated inference in time.

## Benchmark (evaluation for everything)

Latency-responsiveness frontier: per model and operating point (chunk size,
denoise steps, branch budget), report steady-state FPS, control-admission
lag, detector-defined onset lag, pre-/post-admission/undetected rates,
regeneration rate, memory, and long-horizon coherence. Arms: full
regeneration / fixed chunked streaming / 3a / 3b / 3c / combinations.

## Infrastructure

Gamma-World causal-few-step checkpoint (2B distilled DiT, 4 denoise steps,
KV-cached blockwise generation) on the shared H200; assets under
`../safeswm/models` and `../safeswm/external/Gamma-World`. The streaming
driver (chunk-at-a-time generation with mutable action stream) lives here
and is the enabling artifact for all three tracks.

## First results (single-seed pilots; scale-up pre-specified)

**Detector-defined CAOL** (`a2e.py`, 48 measurements over 24 rollouts):
magnitude-based onset lag is 2.4-5.5 pixel frames depending on transition
(stop 2.4, start 3.2, reverse 3.9, turn 5.5), comparable to the four-frame
VAE temporal granularity. Transition-level magnitude-detectable onset rates
are 67-100% (62-100% when grouped by change frame); reverse/turn
non-detection can reflect the detector's loss of direction.

**Serving detector-onset decomposition** (`streaming_swap.py`,
forward->stop at pixel frame 96, dispatch on `start_frame_for_rope`):

**Multi-seed scale-up (4 seeds x 5 arms, with temporal onset labels)**
finds swap+0 post-admission onsets at 9.0 +/- 0.0 frames (3/4 seeds) versus
offline 15.0 +/- 6.9 (4/4). Post-admission onsets fall to 2/4 seeds at D=1
and 0/4 at D=2 and D=4; pre-admission onsets occur in 2/4, 3/4, and 3/4
at D=1,2,4. Timing alone does not identify action causality; matched
continue-forward controls are required.

**Matched causal extension (2 scenes x 6 seeds x 2 admission policies):**
all 24 planned stop-versus-continue-forward pairs pass the raw-artifact and
decoder-support gates. Both views detect paired divergence in every seed.
Mean causal onset lag is 9 frames under 12 frames of admission lag and
21 frames under 24 frames of admission lag, while measured effective
generation throughput differs by only 1.05% (0.712 vs 0.705 FPS).
Direction-aware reverse/turn detectors fail held-out validation and are not
used for causal claims. See `v2/FINDINGS.md`.

Engineering note: the first dispatcher used token-unit thresholds
(`current_start`) and silently mis-switched because the runtime
`frame_seq_length` (3600) differs from latent HxW/4 (600); dispatch on
`start_frame_for_rope` (latent frame index) instead.

## Track pilots (session driver, single scenario, H200; `results/tracks/`)

**3a speculation — flipped from negative to positive by delta snapshots.**
The pilot's honest negative: the sparse-hub KV cache is 37.8 GB, so a full
fork means a host-RAM snapshot over PCIe (fork 21.4 s, restore 4.2 s) and
speculative accept saves nothing over on-demand generation (4.7 s/block in
the session driver). The fix (`driver_v2.py` delta primitives +
`spec_delta.py`): block writes are index-derived slice assignments and the
only mutation of *existing* cache data is the rolling-window eviction, so
an exact on-GPU undo needs only the three end-index scalars per layer plus
the prefix the roll destroys (zero bytes pre-roll, one block = 4.7 GB
post-roll). Measured at the hardest fork point (block 8, the first roll):
**fork 52 ms (413x), restore 85 ms (50x), and accepting a speculated
branch via KV-slice swap costs 6.6 ms vs 4,730 ms on-demand — a ~700x
cut in accept-path latency.** Correctness is exact: max-abs latent error
0.0 vs ground-truth runs for the speculated branch, the restore-then-
regenerate path, the branch swap, and the continuation after the swap
(`results/tracks/spec_delta_report.json`).

Positioning note (from reading the vllm-omni codebase): vLLM-Omni's
experimental AR-diffusion engine serves chunked world models (DreamZero,
Cosmos3) on vLLM's *paged* KV stack, where eviction drops block-table
entries instead of shifting data — the structural endpoint where a fork
becomes an O(table) copy-on-write at any branch depth. Nobody has built
forking on it; our delta snapshot is the no-model-surgery fix for the
contiguous rolling caches models actually ship with, and our measurements
quantify what a paged fork would be worth. vLLM-Omni also has no CAOL-like
metric, and its streaming protocol declares a mid-stream `prompt_update`
that is explicitly unsupported — the interactivity gap this paper fills is
recognized but open in the serving community.

**3b rollback works as designed.** Repair cost is linear (~6.2 s per
rolled-back block in the driver) and the stale-vs-repaired latent
divergence grows with discovery delay (MSE 0.00068 / 0.00083 / 0.00121 at
D=1/2/4) - the frontier's two axes both measurable.

**3c reduced schedules degrade gracefully.** Halving denoise steps halves
generation time (82.5 s -> 41.9 s full-rollout denoise) at latent MSE
0.024 vs the 4-step reference; 1-step quarters it at 0.074. The
action-gated schedule (4 steps for 2 blocks after the change, else 2)
costs 48.1 s. Missing from the pilot: the switch-adjacent vs steady-state
quality split, and a visual check that latent MSE tracks perceptual
quality on a checkpoint distilled for exactly 4 steps.

Driver-note: session-driver blocks run ~4.7-6.2 s vs the stock engine's
~0.96 s/block - the session path lacks the engine's optimizations, so
absolute times are driver-specific; the relative accounting is what
transfers.

## Status

- [x] Scaffold
- [x] Intrinsic CAOL measurement (experiment 1)
- [x] Streaming driver (mid-rollout action re-conditioning, validated)
- [x] Multi-seed serving-CAOL scale-up (response classification; staleness
      forfeits control)
- [x] 3a/3b/3c pilots on the session driver (see above)
- [x] Tutorial notebook (docs/rtwm_tutorial.ipynb): real Gamma-World, live rollout + committed measurements, ELI5 throughout
- [x] 3a delta-snapshot fork (fork 52 ms, accept 6.6 ms, exact; flips 3a)
- [x] CoW fork on a paged cache validated upstream (vllm-omni PR #4909:
      0.01 ms and 0 tensor bytes per branch vs 43.8 ms / 5.4 GiB deep copy)
- [x] arXiv-ready draft (docs/master.tex, 9 pp: serving table, fork-cost
      landscape table, schedule frontier table, conclusion, repro statement)
- [ ] 3c switch-region quality split + visual check
- [ ] Authority-vs-obstruction disambiguation grid (open terrain)
- [ ] Frontier benchmark grid; second world model
