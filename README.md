# Control-Aligned Onset Latency and Copy-on-Write Speculative Branching for Streaming World Models

Code, measurements, and paper source for:

> **Control-Aligned Onset Latency and Copy-on-Write Speculative Branching for
> Streaming World Models.** Arunkumar Eli and Vasu Sharma, 2026. Preprint.
> PDF: [`docs/master.pdf`](docs/master.pdf) · source: [`docs/master.tex`](docs/master.tex)

Interactive world models are usually evaluated by frame rate, but a model can
emit frames continuously while still acting on stale controls. This work
introduces **control-aligned onset latency (CAOL)**: the delay from a control
change to the first rendered motion attributable to that change, separated
into serving-owned admission lag, decoder alignment, and detector onset. Using
matched unchanged-action rollouts on the Gamma-World checkpoint, it measures
how admission policy shifts causal onset while throughput barely moves, and
evaluates exact speculative branching of the rolling KV cache, including a
paged copy-on-write fork contributed upstream to vLLM-Omni
([PR #4909](https://github.com/vllm-project/vllm-omni/pull/4909)).

## What is in this repository

| Path | Contents |
| --- | --- |
| `driver_v2.py` | Blockwise streaming session driver for Gamma-World: mutable action stream, rolling KV cache, delta-snapshot fork/restore primitives. |
| `a2e.py`, `streaming_swap.py`, `score_scale.py` | Experiment 1 (intrinsic CAOL) and the serving-side onset decomposition with mid-stream action swaps. |
| `tracks.py`, `spec_delta.py`, `bench_cow_fork.py` | Pilots for speculative branching (full clone vs. delta snapshot) and the paged copy-on-write fork microbenchmark. |
| `gamma_smoke.py`, `make_figures.py` | Minimal real rollout used by the tutorial, and figure generation for the pilot results. |
| `results/` | Summary CSV/JSON for the v1 pilots (single-seed and four-seed serving studies, track reports). |
| `v2/` | The matched-counterfactual study that the paper's causal claims rest on: protocol, detectors, gates, scoring, plots, tests, and hash-bound result manifests. Start with [`v2/README.md`](v2/README.md) and [`v2/FINDINGS.md`](v2/FINDINGS.md). |
| `v2/proposal_refine/` | Exact one-plus-three proposal/refinement transaction and its readiness gates. |
| `v2/incremental_decode/` | Exact cached incremental decoding and host-readiness timing. |
| `v2/speculation_traces/` | Frozen, action-only VPT trace protocol (never downloads video) and the speculation-frontier analysis. |
| `v2/human_annotation/` | Annotation tool and frozen protocol. Consent and recruitment files are drafting templates and were not used for any study. |
| `v2/results/provenance/gamma_world_source.patch` | The local instrumentation patch applied to Gamma-World (SHA-256 prefix `814691ca7fbb`, as cited in the paper). |
| `docs/` | Paper source, compiled PDF, figures, the numeric macros in `results.tex`, the arXiv ancillary file `anc/data/cow_microbench.json`, the tutorial notebook, and informal project notes. |
| `docs/arxiv_source.zip` | Self-contained upload bundle for arXiv or Overleaf: `master.tex`, `results.tex`, the five included figures, and the ancillary file. |

## Environment

Analysis, scoring, plotting, and the test suite run on CPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest v2 -q
```

GPU rollouts need the Gamma-World checkout and checkpoint, which are **not**
included here:

- Gamma-World source at commit `6a95de85`, with
  `v2/results/provenance/gamma_world_source.patch` applied on top.
- Checkpoint `chijw/Gamma-World`, file `causal-few-step/model.safetensors`,
  Hugging Face snapshot `8e15162f3e49`.
- One NVIDIA H200 (about 38 GB of KV cache at the first rolling block), CUDA
  13.0, PyTorch 2.9.0+cu130, Python 3.10.

The scripts locate that checkout as a sibling directory:
`../safeswm/external/Gamma-World` relative to this repository (see the top of
`a2e.py`). The v2 READMEs use `$RTWM` for this repository and `$GAMMA_WORLD`
for the checkout. Rollouts are launched through Gamma-World's own environment,
for example:

```bash
$GAMMA_WORLD/.venv/bin/torchrun --nproc_per_node=1 $RTWM/a2e.py
```

## Reproducing the paper's numbers

- **Causal STOP study, decoder support, incremental timing:** the `v2/`
  pipeline. Preflight scripts freeze thresholds before any test generation;
  `gates.py` and the `build_*_gate.py` scripts verify the committed manifests
  by hash. `v2/results/` and `v2/proposal_refine/results/` contain the
  summaries the paper's tables and `docs/results.tex` are built from.
- **Paged copy-on-write fork microbenchmark:** `bench_cow_fork.py` against
  vLLM-Omni at revision `ef77add7` with PR #4909 applied. Raw samples are in
  `docs/anc/data/cow_microbench.json`.
- **Figures:** `docs/make_paper_figures.py` regenerates `docs/figures/` from
  the committed results.
- **Paper:** compile `docs/master.tex` with pdfLaTeX; it inputs
  `docs/results.tex` and `docs/figures/`.

Result manifests record the absolute paths of the machine the experiments ran
on. They are kept byte-identical because the gate reports hash them; the paths
are informational only.

## Citation

```bibtex
@misc{eli2026caol,
  title  = {Control-Aligned Onset Latency and Copy-on-Write Speculative
            Branching for Streaming World Models},
  author = {Eli, Arunkumar and Sharma, Vasu},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/arun-elr/caol-streaming-world-models}
}
```

## License

MIT, see [`LICENSE`](LICENSE). Gamma-World, VPT, and vLLM-Omni are separate
projects under their own licenses and are referenced, not vendored.
