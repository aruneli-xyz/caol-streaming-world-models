# Detector ablation: existing v1 videos

Status: completed before the matched-counterfactual pilot.

## Validation

- Five synthetic tests pass: static flow, horizontal sign, radial
  expansion/contraction sign, rendered-frame indexing, and view splitting.
- The v1 future-midpoint detector reproduces the committed v1 results.
- The v2 primary thresholds were applied unchanged from `config/pilot.json`;
  no parameters were selected from these outcomes.

## View-level detection

| transition | v1 future midpoint | v2 pre-only directional |
|---|---:|---:|
| start | 12/12 | 12/12 |
| stop | 9/12 | 2/12 |
| reverse | 8/12 | 0/12 |
| left | 10/12 | 0/12 |

The pre-only detector is therefore not a drop-in replacement for v1. It is
well matched to start but too strict or insufficiently aligned for stop,
reverse, and left on these videos.

## Directional sign check

Late-window minus pre-window medians, used only for validation:

- start magnitude rises in 12/12 views;
- stop magnitude falls in 10/12 views;
- reverse radial flow has the expected falling sign in 7/12 views;
- left horizontal flow has the expected rising sign in 5/12 views.

This weak directional consistency explains why a fixed signed threshold
does not detect reverse/left reliably. The result is retained as an
ablation rather than used to tune a winning detector on evaluated data.

## Decision

The matched STOP/control pilot proceeds with paired magnitude contrast as
its primary causal signal. Transition-aware reverse/left detectors remain
experimental and require calibration on separate null/sign-control videos
before any larger v2 grid.
