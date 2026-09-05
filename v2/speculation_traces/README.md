# Frozen VPT speculation traces

This directory is an isolated, action-only VPT preparation pipeline for RTWM
v2. It never downloads videos. The frozen protocol selects 512 train episodes
and all 85 Solaris test episodes as whole episodes; no frame or block can cross
an episode boundary.

## Frozen protocol

`config.json` defines the selection before data access. `build_protocol.py`
joins the two local Solaris episode metadata files to all five
`vpt_indices/*.json` files. Train episodes with more than one distinct source
mapping are excluded, then eligible episodes are ranked by:

`SHA256(protocol_id + NUL + "train" + NUL + episode_id + NUL + actions_path)`

The first 512 ranks are frozen. Test is all 85 episodes sorted by ID. The
committed `manifests/protocol.json` preserves episode ID, split, frame count,
metadata video/action paths, exact public action URL, all source metadata
hashes, converter hash, and licensing notes. Regeneration is a strict check:

```bash
cd $RTWM/v2/speculation_traces
python3 build_protocol.py --check
```

The public blob being reachable does not establish a dataset license. Solaris
code is Apache-2.0, but its software license does not automatically cover the
VPT recordings/actions. Verify applicable OpenAI and Minecraft terms before
using or redistributing downloaded data.

## Action semantics

`actions.py` is a standalone implementation of
`external/solaris/src/data/minecraft.py`; it does not import or use
`safeswm/gamma/actions.py`. The exact keyboard order is:

1. `inventory`, `ESC`
2. `hotbar.1` through `hotbar.9`
3. `forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint`,
   `swapHands`
4. `attack`, `use`, `pickItem`, `drop`

Camera is `[cameraX, cameraY] == [yaw, pitch]`, with VPT pixels converted to
degrees by `360 / 2400`. Solaris hotbar-edge and episode-start stuck-attack
semantics are retained.

Each JSONL is treated as 20 FPS. At 16 FPS, keyboard state `j` comes from
source index `floor(5*j/4)`. Camera deltas are summed into target bin
`floor(4*i/5)`, preserving total yaw/pitch movement. Keyboard is also packed
losslessly into a 23-bit `uint32` token.

Camera quantization uses 256 per-axis equal-frequency bins. Its 255 inner
edges are fit only after every selected train episode has been converted.
`fit-quantizer` refuses an incomplete train set and records the protocol,
input-array hashes, fit count, algorithm, NumPy version, edges, and quantizer
hash.

## End-to-end commands

All commands use atomic replacement. Downloads keep protocol/URL-bound
`.part` metadata, resume with HTTP Range only when the server confirms the
offset, retry transient failures, parse every JSONL record, require the frozen
frame count, and record SHA-256 and bytes.

```bash
# All 597 selected action files (no MP4 requests).
python3 download.py --workers 8

# Canonical episode arrays, then train-only camera calibration (NumPy required).
PY=$GAMMA_WORLD/.venv/bin/python
$PY convert.py convert
$PY convert.py fit-quantizer

# Complete 12-frame blocks; incomplete episode tails are recorded and dropped.
$PY prepare_blocks.py

# Protocol, Solaris fixture, Gamma ordering, resampling, and safety tests.
$PY -m unittest discover -s tests -v
$PY validate.py --deep
$PY -m compileall -q .
```

For a bounded connectivity/download check, use `--limit`, for example
`python3 download.py --split test --limit 2`. The download manifest remains
bound to the full frozen protocol and merges verified files across runs.

## Outputs

- `manifests/protocol.json`: committed 597-episode frozen source manifest.
- `manifests/downloads.json`: compact downloaded-file SHA-256/byte manifest.
- `manifests/conversion.json`: compact canonical-array provenance.
- `manifests/camera_quantizer.json`: train-only fitted quantizer.
- `manifests/blocks.json`: compact 12-frame block provenance and summary.
- `raw/`, `arrays/`, `blocks/`: generated and gitignored.
- `schemas/`: JSON Schemas for the frozen protocol and download manifest.

Canonical episode NPZ fields are `keyboard[frames,23] uint8`,
`keyboard_token[frames] uint32`, `camera_degrees[frames,2] float32`, and
`source_indices_20fps[frames] int64`. Block NPZ files add
`camera_token[blocks,12,2] uint16` and reshape all fields to episode-local
12-frame blocks.

## Speculation frontier

`speculation_frontier.py` is the frozen action-trace evaluation. The strict
target hashes the exact 12 packed keyboard states and exact canonical float32
camera deltas. The branchable intent target combines the modal exact 23-bit
keyboard state (smallest integer breaks count ties) with ternary yaw/pitch
classes from block-total camera movement. Per-axis dead zones and branch
representatives are fit on all 512 train episodes and atomically recorded
before any test target is constructed.

The repeat-last, global-frequency, first-order Markov, and order-3
history-Markov arms use train counts only. Top-k count ties are resolved by
ascending canonical token bytes, duplicate candidates are removed, and Markov
backoff fills unused budget from shorter contexts and then the global rank.
All predictors share the episode-local eligibility denominator (`block >= 1`);
the change subset additionally requires the intent to differ from the previous
block. Confidence intervals resample whole test episodes.

```bash
PY=$GAMMA_WORLD/.venv/bin/python
$PY speculation_frontier.py
$PY speculation_frontier.py --resume  # verifies run identity and every artifact
```

`measured_gamma_replay.py` runs the hash-frozen replay subset through
`BlockwiseSession`'s exact one-block delta path. It builds canonical keyboard
and camera tensors directly and does not import the SafeSWM action helper.
Primary measurements are serial on one H200: every generated speculative
candidate, abandoned branch, and miss fallback is charged. Worker capacities
1, 2, and 4 are explicitly simulations using measured branch times and a
fixed 12/16 = 0.75 second lead window. The replay measures systems cost and
readiness only, not semantic visual response.

```bash
$PY -m torch.distributed.run --nproc_per_node=1 measured_gamma_replay.py
$PY measured_gamma_replay.py --resume  # strict identity-bound resume
$PY summarize_results.py
```

Generated, path-relative outputs are under `results/`: train-fit parameters,
trace CSV/JSON, replay selection and measurements, flattened replay/projection
CSVs, an umbrella hash manifest, a concise result summary, and two
publication-readable PNG figures. Paged copy-on-write allocator measurements
remain separate controlled results and are never substituted into the Gamma
replay.

