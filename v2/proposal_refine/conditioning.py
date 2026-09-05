"""Conditioning fast path that avoids per-forward recursive deep copies."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from einops import rearrange


def _nonlocals(function: Any) -> dict[str, Any]:
    return dict(inspect.getclosurevars(function).nonlocals)


def unwrap_gamma_x0(x0_fn: Any) -> tuple[Any, int, dict[str, Any]]:
    """Return model, view count, and immutable conditional dictionary."""

    outer = _nonlocals(x0_fn)
    flow = outer.get("flow_pred_fn")
    if flow is None:
        flow = x0_fn
    inner = _nonlocals(flow)
    required = ("self", "n_views", "conditional_dict")
    if any(name not in inner for name in required):
        raise TypeError(
            "x0 closure is not Gamma self_forcing_dmd_mv.get_x0_fn_from_batch"
        )
    return inner["self"], int(inner["n_views"]), inner["conditional_dict"]


def make_shallow_conditioning_x0(x0_fn: Any) -> tuple[Any, dict[str, Any]]:
    """Build an exact candidate fast path.

    Gamma's stock closure recursively deep-copies the entire conditioning
    dictionary for every DiT evaluation, although it only replaces three
    top-level tensor references.  This wrapper pre-unfolds those immutable
    tensors and shallow-copies the top-level mapping before replacing slices.
    The GPU experiment must still exactness-gate this arm before retention.
    """

    model, n_views, conditional = unwrap_gamma_x0(x0_fn)

    def unfold(value: torch.Tensor) -> torch.Tensor:
        return rearrange(value, "b c (v t) h w -> b v c t h w", v=n_views)

    def fold(value: torch.Tensor) -> torch.Tensor:
        return rearrange(value, "b v c t h w -> b c (v t) h w")

    gt_unfold = (
        unfold(conditional["gt_frames"])
        if conditional.get("gt_frames") is not None
        else None
    )
    mask_key = "condition_video_input_mask_B_C_T_H_W"
    mask_unfold = (
        unfold(conditional[mask_key])
        if conditional.get(mask_key) is not None
        else None
    )
    view_key = "view_indices_B_T"
    view_unfold = (
        rearrange(conditional[view_key], "b (v t) -> b v t", v=n_views)
        if conditional.get(view_key) is not None
        else None
    )
    slice_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def sliced_condition(start: int, length: int) -> dict[str, Any]:
        key = (start, length)
        cached = slice_cache.get(key)
        if cached is not None:
            return cached
        end = start + length
        value = conditional.copy()
        if gt_unfold is not None and gt_unfold.shape[3] != length:
            value["gt_frames"] = fold(gt_unfold[:, :, :, start:end])
            if mask_unfold is not None:
                value[mask_key] = fold(mask_unfold[:, :, :, start:end])
        if view_unfold is not None and view_unfold.shape[2] != length:
            value[view_key] = rearrange(
                view_unfold[:, :, start:end], "b v t -> b (v t)"
            )
        slice_cache[key] = value
        return value

    def optimized_x0(
        noise_x: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        noise = noise_x.permute(0, 2, 1, 3, 4)
        noise_unfold = rearrange(
            noise, "b c (v t) h w -> b v c t h w", v=n_views
        )
        start = int(kwargs.get("start_frame_for_rope", 0))
        length = int(noise_unfold.shape[3])
        noise_fold = rearrange(
            noise_unfold, "b v c t h w -> b c (v t) h w"
        )
        _, denoised = model.generator(
            noisy_image_or_video=noise_fold.permute(0, 2, 1, 3, 4),
            conditional_dict=sliced_condition(start, length),
            timestep=timestep,
            kv_cache=kv_cache,
            n_views=n_views,
            **kwargs,
        )
        return denoised

    metadata = {
        "arm": "shallow_conditioning",
        "stock_deepcopy_removed": True,
        "preunfolded_keys": [
            key
            for key, value in (
                ("gt_frames", gt_unfold),
                (mask_key, mask_unfold),
                (view_key, view_unfold),
            )
            if value is not None
        ],
        "slice_cache": slice_cache,
    }
    return optimized_x0, metadata


def cross_attention_cache_compatibility(engine: Any) -> dict[str, Any]:
    blocks = list(engine.model.net.blocks)
    signatures = [
        str(inspect.signature(block.cross_attn.forward)) for block in blocks
    ]
    supported = all(
        "crossattn_cache" in inspect.signature(block.cross_attn.forward).parameters
        for block in blocks
    )
    return {
        "supported": supported,
        "block_count": len(blocks),
        "unique_forward_signatures": sorted(set(signatures)),
        "reason": (
            "all cross-attention blocks accept a projection cache"
            if supported
            else "upstream block passes crossattn_cache to Attention.forward, "
            "but the loaded Attention.forward signature does not accept it"
        ),
    }
