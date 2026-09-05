# Frozen estimand and future analysis plan

## Status and scope

The current generated manifest is for tool/QC validation with fresh canonical
directional_d0 artifacts. It is not the final 100–200-clip presentation
manifest. No person has been recruited, no human response exists, and no human
validation claim is supported.

## Primary estimand

For each eligible intervention view, estimate the distribution of the first
source frame at which the labeled commanded response is visibly expressed.
The primary task displays one intervention clip and its transition label; it
does not display a matched control. Paired divergence is outside this primary
estimand and, if pursued, requires a separate secondary protocol/task.

For an onset response at zero-based clip index `i`, derive
`source_frame = 72 + i`, `lag_frames = source_frame - 96`, and
`lag_seconds = lag_frames / 16` server-side. Do not accept client-derived
source frames or lags.

## Prespecified summaries for the eventual study

- Report onset, none, uncertain, and technical-failure counts separately by
  transition and view.
- Among valid onset judgments, report median lag and interval estimates with
  clustering/resampling at the scene×seed level.
- Report the empirical onset survival curve with none treated as right-censored
  only in a clearly labeled sensitivity analysis; the main categorical
  summaries must not silently coerce none or uncertain.
- Report inter-annotator agreement on overlapping assignments and exact/within
  ±1/±2-frame agreement for onset judgments.
- Report delayed-repeat absolute frame difference and category agreement.
- Report synthetic known-onset/no-onset accuracy and canonical null/control
  false-positive rate.
- Summarize confidence, replay/step behavior, decision time, hidden-tab time,
  preload failures, and technical failures.

## Exclusions and sensitivity analyses

Technical failures are excluded from onset estimation and reported. Uncertain
responses remain a separate category. Any participant-level exclusion rule
based on QC must be finalized before recruitment, applied without consulting
intervention outcomes, and shown with and without exclusions. Never use QC to
deny compensation except under separately approved terms.

## Multiplicity and claims

Back and true-yaw transitions and the two views are prespecified strata.
Any pooled estimate must cluster by scene×seed. Clearly distinguish
confirmatory estimates from exploratory timing/QC analyses. Do not claim human
validation until the final manifest is frozen, ethics requirements are met,
data are collected, and this plan is executed on immutable exports.
