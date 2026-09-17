# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
# Contains logic derived from ComfyUI (GPL-3.0): the DynamicVRAM weight post-cast semantics (comfy/ops.py resolve_cast_module_with_vbar).
"""Per-rank shadow modules and their weight cast.

ComfyUI's DynamicVRAM (aimdo) keeps cast state on the module itself (`_v`, `_prefetch`,
`_v_weight`), so one module cannot be cast to two GPUs at once. Each rank instead gets a
shadow copy of the block's module tree that shares the Parameter objects (no RAM copy),
has the vbar state stripped, and casts through `shadow_cast` below, which mirrors the vbar
path's post-cast (dequant -> lowvram LoRA -> requant) without touching shared state.
"""
import dataclasses
import threading

import torch

import comfy.float
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.quant_ops import QuantizedTensor

_STRIP_PREFIXES = ("_v", "_prefetch", "_comfy_graph", "_pin_state")
_BASE_CAST = comfy.ops.cast_bias_weight   # what was installed when this module was imported (ComfyUI's, or a hook)
_PREV_CAST = None   # what was installed right before our dispatcher, at the last install
_TLS = threading.local()


def _pick(cached, orig):
    """Use the pinned-cache copy only when it still describes the same parameter."""
    if cached is None or orig is None:
        return orig
    if tuple(cached.shape) != tuple(orig.shape) or cached.dtype != orig.dtype:
        return orig
    if orig.device.type != "cpu":
        # resident on a GPU: ComfyUI force-loaded it, possibly with LoRA baked in; the cached base copy would undo that
        return orig
    return cached


def _to_device(t, device):
    # Weights force-loaded onto the other GPU may sit in aimdo-managed memory that the peer cannot map:
    # stage GPU->GPU through host RAM. CPU weights go straight up; pinned (cached) ones without blocking.
    if t is None or t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda":
        t = t.to("cpu")
    return t.to(device, non_blocking=True)


def _staged(s, param_key, src, device, memo):
    """Host/cached tensor -> device in model dtype. With a per-call memo (VAE tiling calls the same modules many times),
    the device copy is made once per top-level call and reused; post-cast still runs per call."""
    if src is None:
        return None
    if memo is None:
        return _to_model_dtype(s, param_key, _to_device(src, device))
    key = (id(s), param_key, device)
    t = memo.get(key)
    if t is None:
        t = _to_model_dtype(s, param_key, _to_device(src, device))
        memo[key] = t
    return t


def _to_model_dtype(s, param_key, t):
    # DynamicVRAM keeps the file's storage dtype on the host (e.g. H3 adaln bias is f16 on disk) and records the
    # model dtype in <param>_comfy_model_dtype; the vbar cast stages to that dtype before post-cast
    # (ModelPatcherDynamic.load setup_param geometry). Casting f16 -> fp32 directly skips the f16 -> bf16 rounding
    # and gives different numbers, which the MLP amplifies. Legacy loading converts at load time instead.
    if t is None or isinstance(t, QuantizedTensor):
        return t
    model_dtype = getattr(s, param_key + "_comfy_model_dtype", None)
    if model_dtype is not None and t.dtype != model_dtype:
        t = t.to(model_dtype)
    return t


def stage_block(block, cached, device):
    """Copy a block's pinned-cache weights to `device` in model dtype, exactly as shadow_cast would stage them, on the
    current CUDA stream. Returns {(module_path, param): device tensor} for the params whose cached copy shadow_cast
    would use (see _pick). The split calls it on a side stream to prefetch the next block."""
    out = {}
    for name, m in block.named_modules():
        if not hasattr(m, "comfy_cast_weights"):
            continue
        for p in ("weight", "bias"):
            c = cached.get((name, p))
            if c is None or _pick(c, getattr(m, p, None)) is not c:
                continue
            out[(name, p)] = _to_model_dtype(m, p, _to_device(c, device))
    return out


def _to_dequant(tensor, dtype):
    tensor = tensor.to(dtype=dtype)
    if isinstance(tensor, QuantizedTensor):
        tensor = tensor.dequantize()
    return tensor


def _post_cast(s, param_key, x, dtype, compute_dtype, want_requant):
    """Mirror of comfy.ops.resolve_cast_module_with_vbar.post_cast with resident=False."""
    if x is None:
        return None
    lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
    fns = getattr(s, param_key + "_function", [])
    orig = x
    if orig.dtype != dtype or len(fns) > 0:
        x = _to_dequant(x, dtype)
    if lowvram_fn is not None:
        x = _to_dequant(x, dtype if compute_dtype is None else compute_dtype)
        x = lowvram_fn(x)
        if want_requant and len(fns) == 0:
            seed = comfy.utils.string_to_seed(s.seed_key)
            if isinstance(orig, QuantizedTensor):
                x = orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)
            else:
                x = comfy.float.stochastic_rounding(x, orig.dtype, seed=seed)
    for f in fns:
        x = f(x)
    return x


def shadow_cast(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False,
                compute_dtype=None, want_requant=False):
    if input is not None:
        if dtype is None:
            dtype = input.params.orig_dtype if isinstance(input, QuantizedTensor) else input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device
    cached = getattr(s, "_multistream_cached", None) or {}
    memo = getattr(s, "_multistream_memo", None)
    staged = getattr(s, "_multistream_staged", None) or {}
    weight = staged.get("weight")
    if weight is None:
        weight = _staged(s, "weight", _pick(cached.get("weight"), s.weight), device, memo)
    bias = staged.get("bias")
    if bias is None:
        bias = _staged(s, "bias", _pick(cached.get("bias"), getattr(s, "bias", None)), device, memo)
    weight = _post_cast(s, "weight", weight, dtype, compute_dtype, want_requant)
    bias = _post_cast(s, "bias", bias, bias_dtype, compute_dtype, want_requant)
    return (weight, bias, None) if offloadable else (weight, bias)


def _cast_dispatch(s, *args, **kwargs):
    if getattr(s, "_multistream_rank", None) is not None:
        return shadow_cast(s, *args, **kwargs)
    if getattr(_TLS, "in_prev", False):
        # Re-entered while calling the previous hook: that hook captured our dispatcher as its own original (it was
        # imported after us) and we later chained onto it. Break the cycle with what we saw at import time.
        return _BASE_CAST(s, *args, **kwargs)
    _TLS.in_prev = True
    try:
        return (_PREV_CAST or _BASE_CAST)(s, *args, **kwargs)
    finally:
        _TLS.in_prev = False


def install_cast_hook():
    """Route shadow modules through shadow_cast; every other module falls through to the cast that was installed before.

    Chains instead of replacing: another custom node may hook cast_bias_weight too, in
    either load order and without chaining itself. If something replaced our hook since the last call, re-install on
    top of it and keep it as the fallback, so both hooks stay active; _cast_dispatch breaks the cycle that forms when
    that hook had captured ours. Idempotent; rank threads call it concurrently only after the first install."""
    global _PREV_CAST
    current = comfy.ops.cast_bias_weight
    if current is not _cast_dispatch:
        _PREV_CAST = current
        comfy.ops.cast_bias_weight = _cast_dispatch


def stage_block_resident(block, device):
    """A block's CPU weights copied to `device` in model dtype, for the VRAM block cache.

    Same selection as stage_block, but sourced from the module's OWN parameters instead of a pinned
    cache entry, so it works with the host weight cache off -- which is the configuration the VRAM
    cache exists to make viable. Weights already on a GPU are skipped for the reason _pick gives:
    ComfyUI force-loaded them, possibly with LoRA baked in, and a base copy would undo that."""
    out = {}
    for name, m in block.named_modules():
        if not hasattr(m, "comfy_cast_weights"):
            continue
        for p in ("weight", "bias"):
            t = getattr(m, p, None)
            if t is None or getattr(t, "is_meta", False) or t.device.type != "cpu":
                continue
            out[(name, p)] = _to_model_dtype(m, p, _to_device(t, device))
    return out


def tensor_bytes(t):
    """Bytes a staged tensor really occupies on the device.

    QuantizedTensor subclasses torch.Tensor and does NOT override numel()/element_size(), so the
    obvious `t.numel() * t.element_size()` reports the DEQUANTISED logical size -- about 2x the truth
    for H3's int8 weights (791 MiB/block against 369 actual, measured 2026-09-16). Every size the
    VRAM tier reasons about has to come through here, or its budget is charged twice what it spends.
    """
    if isinstance(t, QuantizedTensor):
        n = t._qdata.numel() * t._qdata.element_size()
        for f in dataclasses.fields(t._params):
            v = getattr(t._params, f.name)
            if isinstance(v, torch.Tensor):
                n += v.numel() * v.element_size()
        return n
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    return 0


def staged_size(block):
    """Bytes stage_block_resident would put on a device for `block`, without staging it.

    Needed up front: the tiering has to know how many blocks fit in VRAM BEFORE the first step, so
    the host cache can skip pinning the ones that will be resident. Mirrors stage_block_resident's
    selection exactly, and counts the MODEL dtype, which is what actually lands on the card -- for
    H3 int8 that is about 2x the on-disk size (measured 791 MiB/block against 370 on disk)."""
    total = 0
    for name, m in block.named_modules():
        if not hasattr(m, "comfy_cast_weights"):
            continue
        for p in ("weight", "bias"):
            t = getattr(m, p, None)
            if t is None or getattr(t, "is_meta", False) or t.device.type != "cpu":
                continue
            if isinstance(t, QuantizedTensor):
                total += tensor_bytes(t)
                continue
            dt = getattr(m, p + "_comfy_model_dtype", None) or t.dtype
            total += t.numel() * torch.empty((), dtype=dt).element_size()
    return total


def make_shadow(module, rank, cached=None, prefix="", staged=None):
    """Shallow module-tree copy sharing Parameters, with vbar state stripped.

    cached: optional {(module_path, param): pinned tensor} for this block (multistream.cache).
    staged: optional {(module_path, param): device tensor} already copied by stage_block (prefetch)."""
    n = module.__class__.__new__(module.__class__)
    d = {k: v for k, v in module.__dict__.items() if not k.startswith(_STRIP_PREFIXES)}
    d["_parameters"] = dict(module._parameters)
    d["_buffers"] = dict(module._buffers)
    d["_modules"] = {k: (make_shadow(c, rank, cached, f"{prefix}.{k}" if prefix else k, staged)
                         if c is not None else None)
                     for k, c in module._modules.items()}
    n.__dict__.update(d)
    if hasattr(module, "comfy_cast_weights"):
        n.comfy_cast_weights = True
        n._multistream_rank = rank
        if cached:
            entry = {p: cached[(prefix, p)] for p in ("weight", "bias") if (prefix, p) in cached}
            if entry:
                n._multistream_cached = entry
        if staged:
            s_entry = {p: staged[(prefix, p)] for p in ("weight", "bias") if (prefix, p) in staged}
            if s_entry:
                n._multistream_staged = s_entry
        for key in ("weight", "bias"):
            fn = getattr(module, key + "_lowvram_function", None)
            if isinstance(fn, comfy.model_patcher.LowVramPatch):
                # fresh instance: the original's prepared_patches belong to the vbar prefetch on cuda:0
                setattr(n, key + "_lowvram_function", comfy.model_patcher.LowVramPatch(fn.key, fn.patches))
    return n
