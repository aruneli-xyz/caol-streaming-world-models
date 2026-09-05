# Human response-onset tool

Local, standard-library-only annotation service for the frozen primary
estimand: show one intervention view plus its commanded transition label and
mark the first visible commanded response. It never reads or serves MP4 and
never shows a matched control beside the intervention.

**Current status:** tool/QC validation only. The included build recipe uses
fresh Solaris-canonical directional_d0 forward→back and forward→true-yaw
artifacts. It does not use legacy STOP media. There are no recruited
participants, human labels, or human-validation claims. Recruitment requires a
documented IRB or equivalent ethics determination.

## Build and validate the private item manifest

From this directory:

```sh
python3 human_tool.py build \
  --source ../results/directional_d0/manifest.json \
  --output runtime/private_manifest.json \
  --secret runtime/secret.key
python3 human_tool.py validate --manifest runtime/private_manifest.json
```

Add `--hash-sources` for a slow full SHA-256 pass over every referenced raw
array. Selection reads only status, arm, transition, and provenance fields; it
does not load detector outputs or scores. Every complete eligible intervention
rollout contributes both 1280×720 views. Canonical controls/nulls are retained
only as hidden QC.

The builder is generic over a supplied manifest with the directional schema:
it requires top-level protocol/action hashes and per-rollout action and
decoded-array hashes. The output envelope binds canonical JSON content to
`manifest_sha256`. Opaque IDs are HMAC-derived using a private 32-byte key.

## Serve

```sh
python3 human_tool.py serve \
  --manifest runtime/private_manifest.json \
  --db runtime/annotations.sqlite3 --port 8765
```

Open `http://127.0.0.1:8765`. The server is hard-bound to `127.0.0.1` using
`ThreadingHTTPServer`. The browser receives only opaque IDs, task constants,
the command label, and PNG frames. Source rollout, paths/hashes, scene, seed,
arm, transition code, view/cue/window, QC identity, known answers, and duplicate
groups remain private.

The server reads C-order uint8 `[T,720,2560,3]` NPY files by memory map, slices
one 1280×720 view, and encodes PNG in the standard library. Submission is
rejected unless all 89 frame endpoints were requested over a near-real-time
16-FPS pass. Frame stepping unlocks only after browser playback completes.
Onset source frame and lag are derived server-side.

SQLite uses WAL, foreign keys, immediate transactions, expiring assignment
leases, one open assignment per annotator, and uniqueness constraints for both
assignment and response. Assignment balances least-completed items, prevents
repeat primary groups for an annotator, and delays hidden repeats until the
original plus four primary items are complete.

## Immutable export

Stop annotation writes before export, then choose a new destination:

```sh
python3 human_tool.py export --manifest runtime/private_manifest.json \
  --db runtime/annotations.sqlite3 \
  --output runtime/exports/2026-08-21T120000Z
```

An existing destination is refused. JSONL and CSV hashes, row count, private
manifest hash, and an export hash are written to `export_manifest.json`; files
and directory are made read-only. Archive on write-once or versioned storage
for stronger immutability.

## Runtime paths to ignore later

Do not commit these private/runtime artifacts. An external gitignore may be
updated later, but this implementation intentionally does not edit it:

- `human_annotation/runtime/secret.key`
- `human_annotation/runtime/private_manifest.json`
- `human_annotation/runtime/annotations.sqlite3*`
- `human_annotation/runtime/cache/`
- `human_annotation/runtime/exports/`

The private manifest contains absolute raw-array paths and experimental
provenance. The database contains participant codes and behavioral metadata.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests cover schema/payload checks, manifest hash binding and leak resistance,
assignment balance/repeat delay simulation, SQLite uniqueness/recovery,
NPY frame index/view/hash behavior, immutable export, and live server APIs.

## Remaining work before a 100–200 clip human study

1. Generate enough fresh Solaris-canonical STOP data; validate action/source
   hashes and raw frame support.
2. Define the eligible population and freeze a 100–200 presentation manifest
   independent of all detector/human outcomes, including overlap and QC rates.
3. Finalize annotator count, overlap, power/precision target, stopping rules,
   exclusions, and randomization.
4. Obtain IRB/equivalent determination and approvals for consent,
   recruitment, privacy, retention, and compensation.
5. Run accessibility/browser/load pilots, confirm duration and pay, archive
   hashes/code/environment, then authorize recruitment.
