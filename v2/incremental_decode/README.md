# Wan2.1 cached incremental decode probe

This isolated probe verifies that a Gamma-World latent prefix can be decoded in
3-latent blocks with one persistent Wan2.1 decoder feature cache. It reads a
hash-verified Solaris-canonical `directional_d0` latent/raw pair and does not
modify that result root.

The implementation deliberately calls `WanVAE.decode(...,
clear_decoder_cache=False)` after one cache clear at session start. It bypasses
the public tokenizer wrapper for incremental chunks because that wrapper
restarts `video_mean`/`video_std` at temporal index zero. Every chunk instead
uses the global slice `[latent_start:latent_end]`.

## CPU tests

```bash
cd $RTWM/v2/incremental_decode
python -m pytest -q
```

## Minimal H200 probe

Run with Gamma-World's environment:

```bash
cd $GAMMA_WORLD
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
  $RTWM/v2/incremental_decode/gpu_probe.py \
  --source-root $RTWM/v2/results/directional_d0 \
  --rollout-id buildTower_normal__seed201__control \
  --max-blocks 2 \
  --sink hash \
  --output $RTWM/v2/results/incremental_decode/directional_control_2blocks
```

`--max-blocks` controls the small prefix size; blocks are always three latent
frames. Gamma `[B,C,V*T,H,W]` input is split into a `[B*V,C,T,H,W]` decoder
batch, preserving one independent cache stream per view.

The probe writes raw full/incremental uint8 arrays, per-block JSONL events, and
a sealed manifest using atomic replacements. Existing results resume only when
the manifest seal, complete identity (including source/dependency hashes), and
all retained artifact hashes match. Otherwise the probe refuses the resume.
All retained paths are relative.

## Timing gate and scope

Incremental timings become claimable only if concatenated incremental raw
uint8 equals the one-shot decode of the identical latent prefix exactly for
both views and every frame. Any nonzero maximum error withholds timing; the
threshold is never relaxed. The corresponding prefix of the independently
generated, full-length canonical directional raw artifact is also compared and
hash-recorded as a provenance diagnostic, but it is not the identical-input
incremental equivalence gate.
Events distinguish decoder return, CUDA device-ready, host-ready, and optional
synchronous sink callback completion. The sink callback only demonstrates host
consumption. It is explicitly **not physical display presentation**.

The benchmark covers VAE decode/quantization/host transfer for a supplied
latent prefix. It does not include world-model denoising, action admission,
network transport, compositor/display scanout, or end-user input latency.

## Canonical STOP wall-clock chain

`stop_wallclock_config.json` freezes a separate replacement for the legacy
frame-ready benchmark: buildTower_normal, seeds 301–302, d0 latent 27 and d1
latent 30, counterbalanced as d0/d1 then d1/d0. Synthetic action receipt is
injected at latent 24. Inputs are loaded directly from hash-verified action
tensor artifacts in the fresh canonical STOP/control result roots; neither
`tracks.make_x0` nor `safeswm/gamma/actions.py` is used.

```bash
cd $GAMMA_WORLD
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python \
  $RTWM/v2/incremental_decode/stop_wallclock.py
```

Each live run generates blocks from session start through its admission block.
Every block records denoise completion, context-cache commit, cached decoder
return, CUDA device readiness, synchronous host readiness, queue/drop counts,
and optional sink consumption. A matching one-shot decode of that run's exact
latent prefix must have identical raw uint8 hashes and max error zero for both
views before any wall-clock aggregate is emitted.

The output contains sealed per-run manifests/events, summary JSON/CSV, and a
decomposition plot. With two seeds per delay, uncertainty is reported only as
descriptive sample standard deviation (`ddof=1`), not population inference.
Sink timestamps remain host callback consumption and are not physical display
presentation.
