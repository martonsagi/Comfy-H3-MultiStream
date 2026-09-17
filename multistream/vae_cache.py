# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""VAE weight cache: pin a VAE's host weights once and route its modules through the shadow cast.

The H3 video VAE (int8 convrot, 2.95 GiB) and audio VAE (fp32, 0.56 GiB) are re-staged by DynamicVRAM after the DiT
has evicted them (measured ~22 s per scene for encode + decode + audio at a short clip).

VAE.decode/encode call load_models_gpu first and then first_stage_model.decode/encode, so the hook fills the cache
after ComfyUI's load (GPU-resident force-loaded weights are skipped and keep any baked-in patches). Bulk weights move
through cast_bias_weight: the video VAE's ops Conv3d/GroupNorm/Linear, the audio VAE's explicit qkv cast. The few
direct accesses (norm weights, scales, register tokens, snake alphas, filters) are small and stay on the normal path.

The H3 video VAE tiles internally and calls the same modules per tile, so each top-level call stages every cached
weight on the GPU once (per-call memo) and releases it when the call returns.

SelectVAEDevice shallow-copies the VAE and sets vae.first_stage_model to the retargeted patcher's model, so place the
node after it: the hook goes on the first_stage_model the VAE actually decodes with.
"""
import contextlib
import logging
import os
import time

from . import cache as ms_cache
from . import hooks as ms_hooks
from . import te_cache as ms_te_cache
from .log import gib, log, ram

_ORIG_ATTR = "_h3ms_orig_vae_methods"
_WRAP_ATTR = "_h3ms_wrap_vae_methods"
_CFG_ATTR = "_h3ms_vae_cfg"
_DEPTH_ATTR = "_h3ms_vae_depth"
_METHODS = ("decode", "encode", "decode_tiled", "encode_tiled")


def vae_cache_key(vae):
    init = getattr(getattr(vae, "patcher", None), "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    path = init[1][0]
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    return (os.path.realpath(path), "vae")


def ensure_attached(model):
    """Attach the pinned cache to model's modules; returns the attached modules ([] when off or unavailable)."""
    c = getattr(model, _CFG_ATTR, None)
    if not c or not c["enabled"]:
        return []
    # one group per VAE file: the video and audio VAEs must not evict each other
    wc = ms_cache.acquire(c["base_key"], c["reserve_gib"], "vae:" + c["base_key"][0])
    if wc is None:
        return []
    t0 = time.perf_counter()
    filled_before = len(wc.blocks)
    entry = wc.get_block(0, model)
    if entry is None:
        ms_te_cache.detach(model)
        return []
    n = ms_te_cache._attach_weight_cache(model, entry)
    if not filled_before:
        log.info("[VAE] weight cache filled (%s): %s pinned in %.1fs, %d modules | %s",
                 os.path.basename(c["base_key"][0]), gib(wc.bytes), time.perf_counter() - t0, n, ram())
    return [m for m in model.modules() if getattr(m, "_multistream_rank", None) == -1]


@contextlib.contextmanager
def staged_call(model):
    """For one top-level VAE call: attach the cache and stage each cached weight on the GPU once (per device)."""
    depth = getattr(model, _DEPTH_ATTR, 0)
    if depth:  # nested (e.g. decode_tiled -> decode): the outer call owns the memo
        yield
        return
    # in_use() spans the whole VAE call, not just the attach: the cast copies weights out of the pinned
    # tensors on every module call, so a concurrent release() must not unpin them before this returns.
    c = getattr(model, _CFG_ATTR, None)
    wc = ms_cache.acquire(c["base_key"], c["reserve_gib"], "vae:" + c["base_key"][0]) if c and c["enabled"] else None
    with wc.in_use() if wc is not None else contextlib.nullcontext() as held:
        if wc is not None and held is None:
            ms_te_cache.detach(model)   # the cache is closing; this call streams without it
            modules = []
        else:
            modules = ensure_attached(model)
        memo = {}
        for m in modules:
            m._multistream_memo = memo
        setattr(model, _DEPTH_ATTR, 1)
        try:
            yield
        finally:
            setattr(model, _DEPTH_ATTR, 0)
            for m in modules:
                m.__dict__.pop("_multistream_memo", None)
            memo.clear()


def uninstall(model):
    """Put the original decode/encode methods back and forget the pinned weights."""
    ms_te_cache.detach(model)
    return ms_hooks.uninstall(model, _ORIG_ATTR, _WRAP_ATTR, _METHODS, "VAE")


def install(vae, enabled=True, weight_cache=True, reserve_gib=0.0):
    """Hook the VAE's decode/encode so cached weights stage once per call.

    enabled=False fully REMOVES the hook: it lives on the shared first_stage_model, not on this node,
    so leaving it attached keeps it running after the node is bypassed. See multistream/hooks.py."""
    model = vae.first_stage_model
    if not enabled:
        if uninstall(model):
            ms_cache.release_async("vae", "VAE cache node disabled")
        else:
            log.info("[VAE] node disabled: nothing was hooked")
        return
    base_key = vae_cache_key(vae)
    cfg = {"base_key": base_key, "enabled": bool(weight_cache) and base_key is not None,
           "reserve_gib": float(reserve_gib)}
    if weight_cache and base_key is None:
        log.warning("[VAE] weight cache unavailable: loader has no reload factory")
    setattr(model, _CFG_ATTR, cfg)
    log.info("[VAE] cache node: %s, weight cache %s, RAM reserve %.1f GiB",
             os.path.basename(base_key[0]) if base_key else type(model).__name__,
             "on" if cfg["enabled"] else "off", cfg["reserve_gib"])
    if not cfg["enabled"]:
        ms_te_cache.detach(model)               # drop the modules' pointers first ...
        ms_cache.release_async("vae", "VAE weight cache switched off")   # ... then unpin the RAM
    if getattr(model, _ORIG_ATTR, None) is not None:
        return

    originals, wrappers = {}, {}
    for name in _METHODS:
        fn = getattr(model, name, None)
        if fn is None:
            continue
        originals[name] = fn

        def wrapped(*args, _fn=fn, **kwargs):
            with staged_call(model):
                return _fn(*args, **kwargs)

        setattr(model, name, wrapped)
        wrappers[name] = wrapped
    setattr(model, _ORIG_ATTR, originals)
    setattr(model, _WRAP_ATTR, wrappers)
    ms_hooks.remember(model, "VAE", _ORIG_ATTR, _WRAP_ATTR, tuple(originals))
