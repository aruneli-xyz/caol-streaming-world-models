"""Collect retained empirical proposal timing samples in isolated processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

if __package__:
    from . import gpu_experiment as experiment
    from .conditioning import make_shallow_conditioning_x0
    from .core import ExactProposalTransaction
else:
    import gpu_experiment as experiment
    from conditioning import make_shallow_conditioning_x0
    from core import ExactProposalTransaction


def one_block_exact(
    session,
    x0_fn,
    action_tensors,
    source_hashes,
    label: str,
):
    parent = session.fork_delta()
    reference_tx = ExactProposalTransaction(
        session, logical_session_id=label, source_hashes=source_hashes
    )
    reference = reference_tx.run_full(x0_fn, action_tensors)
    reference_latent = reference.latent.clone()
    reference_output = session.output.clone()
    reference_live = experiment.digest_live_state(session)
    session.restore_delta(parent)
    candidate_tx = ExactProposalTransaction(
        session, logical_session_id=label, source_hashes=source_hashes
    )
    proposal = candidate_tx.propose(x0_fn, action_tensors)
    candidate = candidate_tx.accept_exact(proposal, x0_fn, action_tensors)
    result = {
        "latent": experiment.compare_tensor(reference_latent, candidate.latent),
        "output": experiment.compare_tensor(reference_output, session.output),
        "live_kv": experiment.compare_live_digest(
            reference_live, experiment.digest_live_state(session)
        ),
    }
    result["exact"] = all(row["exact"] for row in result.values())
    return result


def collect(output: Path) -> dict:
    config = json.loads(experiment.CONFIG.read_text())
    identity = experiment.source_identity()
    identity["timing_samples_source_sha256"] = experiment.sha256_file(Path(__file__))
    identity["identity_sha256"] = hashlib.sha256(
        experiment.canonical_bytes(identity)
    ).hexdigest()
    engine = experiment.build_engine(config)
    images = experiment.load_images()
    keyboard, camera = experiment.action_arrays("unchanged", 0)
    batch, stock_x0, tensors, _ = experiment.make_x0(
        engine, images, keyboard, camera
    )
    shallow_x0, _ = make_shallow_conditioning_x0(stock_x0)
    session = experiment.new_session(engine, batch, int(config["model"]["seed"]))
    boundaries = experiment.cache_boundaries(session)
    arms = {"stock_conditioning": stock_x0, "shallow_conditioning": shallow_x0}
    source_hashes = identity["sources"]
    warmups = int(config["profiling"]["first_forward_warmups"])
    repetitions = int(config["profiling"]["first_forward_repetitions"])
    report = {
        "schema_version": "rtwm-v2-proposal-timing-samples-1",
        "status": "running",
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "nvte_fused_attn": os.environ.get("NVTE_FUSED_ATTN"),
        "boundaries": {},
    }
    started = time.perf_counter()
    for phase in ("first_roll", "steady_roll"):
        experiment.advance_to(
            session,
            stock_x0,
            tensors,
            boundaries[phase],
            source_hashes,
            f"samples-prefix:{phase}",
        )
        for name, x0_fn in arms.items():
            for _ in range(warmups):
                tx = ExactProposalTransaction(
                    session,
                    logical_session_id=f"samples:{phase}",
                    source_hashes=source_hashes,
                )
                state, _ = experiment.timed_cuda(
                    lambda x0_fn=x0_fn: tx.propose(x0_fn, tensors)
                )
                tx.discard(state)
        values = {name: [] for name in arms}
        rng = random.Random(
            int(config["profiling"]["randomized_arm_order_seed"])
            + boundaries[phase]
        )
        for repetition in range(repetitions):
            order = list(arms)
            rng.shuffle(order)
            for name in order:
                tx = ExactProposalTransaction(
                    session,
                    logical_session_id=f"samples:{phase}",
                    source_hashes=source_hashes,
                )
                torch.cuda.reset_peak_memory_stats()
                state, timing = experiment.timed_cuda(
                    lambda name=name: tx.propose(arms[name], tensors)
                )
                values[name].append(
                    {
                        "repetition": repetition,
                        **timing,
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated()
                        ),
                    }
                )
                tx.discard(state)
        exact = one_block_exact(
            session,
            shallow_x0,
            tensors,
            source_hashes,
            f"samples-exact:{phase}",
        )
        session.restore_delta(session.fork_delta())
        report["boundaries"][phase] = {
            "block_index": boundaries[phase],
            "warmups_per_arm": warmups,
            "samples": values,
            "summaries": {
                name: {
                    "cuda_event": experiment.percentiles(
                        [row["cuda_event_ms"] for row in rows]
                    ),
                    "synchronized_host": experiment.percentiles(
                        [row["host_sync_ms"] for row in rows]
                    ),
                }
                for name, rows in values.items()
            },
            "one_block_exactness": exact,
        }
        experiment.atomic_json(output, report)
    report["runtime_seconds"] = time.perf_counter() - started
    report["status"] = "complete"
    experiment.atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "nvte_fused_attn": report["nvte_fused_attn"],
                "runtime_seconds": report["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
