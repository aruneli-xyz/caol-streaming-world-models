"""Repeat the paged AR-diffusion COW fork and deep-copy microbenchmarks.

The COW path is host-side block-table/refcount work. A GPU tensor set with the
same total resident size is kept allocated while the real
``ARDiffusionKVState.fork`` implementation is timed. The deep-copy baseline
clones that GPU-resident state, including output-tensor allocation.

Run with the vLLM-Omni environment from the PR checkout:

    /path/to/vllm-omni/.venv/bin/python bench_cow_fork.py \
        --output docs/anc/data/cow_microbench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState


RTWM = Path(__file__).resolve().parent
DEFAULT_OUT = RTWM / "docs" / "anc" / "data" / "cow_microbench.json"

# DreamZero-like metadata geometry used in the original PR microbenchmark.
NUM_LAYERS = 30
NUM_KV_HEADS = 40
HEAD_SIZE = 128
FRAME_SEQLEN = 220
WINDOW_CHUNKS = 20
FANOUT = 4


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_state() -> ARDiffusionKVState:
    config = ARDiffusionKVConfig(
        enable=True,
        chunk_size=FRAME_SEQLEN,
        window_chunks=WINDOW_CHUNKS,
        gpu_memory_fraction=1.0,
    )
    bytes_per_pool_block = (
        2
        * FRAME_SEQLEN
        * NUM_KV_HEADS
        * HEAD_SIZE
        * torch.bfloat16.itemsize
        * NUM_LAYERS
    )
    cache = ARDiffusionKVCache(
        config,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        block_size=FRAME_SEQLEN,
        max_model_len=FRAME_SEQLEN * 64,
        available_bytes=bytes_per_pool_block * 128,
        device=None,
    )
    state = ARDiffusionKVState(
        cache,
        cache.begin_request("parent.pos"),
        cache.begin_request("parent.neg"),
        num_layers=NUM_LAYERS,
    )
    for adapter in (state.pos, state.neg):
        for _ in range(WINDOW_CHUNKS):
            cache.allocate_chunk(adapter)
            cache.commit_chunk(adapter)
    return state


def allocate_resident_layers(resident_gib: float) -> list[torch.Tensor]:
    total_bytes = round(resident_gib * 2**30)
    elements = total_bytes // torch.bfloat16.itemsize
    base, remainder = divmod(elements, NUM_LAYERS)
    layers = [
        torch.empty(
            base + (1 if layer < remainder else 0),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for layer in range(NUM_LAYERS)
    ]
    for tensor in layers:
        tensor.zero_()
    torch.cuda.synchronize()
    return layers


def benchmark_cow(
    state: ARDiffusionKVState,
    warmup: int,
    repetitions: int,
) -> list[float]:
    sequence = 0

    def batch(count: int, *, record: bool) -> list[float]:
        nonlocal sequence
        children: list[ARDiffusionKVState] = []
        samples: list[float] = []
        for _ in range(count):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            child = state.fork(f"child.{sequence}.pos", f"child.{sequence}.neg")
            elapsed_us = (time.perf_counter_ns() - started) / 1_000
            children.append(child)
            sequence += 1
            if record:
                samples.append(elapsed_us)
        for child in children:
            child.close()
        return samples

    for offset in range(0, warmup, FANOUT):
        batch(min(FANOUT, warmup - offset), record=False)

    values: list[float] = []
    for offset in range(0, repetitions, FANOUT):
        values.extend(batch(min(FANOUT, repetitions - offset), record=True))
    return values


def benchmark_deep_copy(
    resident_layers: list[torch.Tensor],
    warmup: int,
    repetitions: int,
    *,
    cold_allocator: bool,
) -> list[float]:
    def clone_once() -> float:
        if cold_allocator:
            # Force the clone to obtain new device storage. This is outside the
            # timer, but ensures allocation itself occurs inside tensor.clone().
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        copies = [tensor.clone() for tensor in resident_layers]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        del copies
        return elapsed_ms

    for _ in range(warmup):
        clone_once()
    return [clone_once() for _ in range(repetitions)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow-repetitions", type=int, default=1_000)
    parser.add_argument("--cow-warmup", type=int, default=100)
    parser.add_argument("--deep-copy-repetitions", type=int, default=50)
    parser.add_argument("--deep-copy-warmup", type=int, default=5)
    parser.add_argument("--resident-gib", type=float, default=5.36)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    state = build_state()
    resident_layers = allocate_resident_layers(args.resident_gib)

    cow_us = benchmark_cow(state, args.cow_warmup, args.cow_repetitions)
    deep_copy_cold_ms = benchmark_deep_copy(
        resident_layers,
        args.deep_copy_warmup,
        args.deep_copy_repetitions,
        cold_allocator=True,
    )
    deep_copy_warm_ms = benchmark_deep_copy(
        resident_layers,
        args.deep_copy_warmup,
        args.deep_copy_repetitions,
        cold_allocator=False,
    )

    device = torch.cuda.get_device_properties(0)
    report = {
        "protocol": {
            "cow_warmup": args.cow_warmup,
            "cow_repetitions": args.cow_repetitions,
            "deep_copy_warmup": args.deep_copy_warmup,
            "deep_copy_repetitions": args.deep_copy_repetitions,
            "fanout": FANOUT,
            "resident_gib": args.resident_gib,
            "resident_tensors": NUM_LAYERS,
            "window_table_entries_per_cfg_stream": WINDOW_CHUNKS,
            "cfg_streams_per_session": 2,
            "cuda_sync_before_each_start": True,
            "cuda_sync_after_deep_copy": True,
            "deep_copy_includes_output_allocation": True,
            "deep_copy_cold_clears_cached_output_before_each_start": True,
        },
        "geometry": {
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "frame_seqlen": FRAME_SEQLEN,
            "window_chunks": WINDOW_CHUNKS,
            "dtype": "bfloat16",
        },
        "environment": {
            "gpu": device.name,
            "gpu_total_memory_mib": round(device.total_memory / 2**20),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cow_fork_us": summarize(cow_us),
        "deep_copy_cold_ms": summarize(deep_copy_cold_ms),
        "deep_copy_warm_ms": summarize(deep_copy_warm_ms),
        "raw": {
            "cow_fork_us": cow_us,
            "deep_copy_cold_ms": deep_copy_cold_ms,
            "deep_copy_warm_ms": deep_copy_warm_ms,
        },
    }
    state.close()
    del resident_layers

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "raw"}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
