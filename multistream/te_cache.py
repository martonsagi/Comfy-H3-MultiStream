# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Text-encoder caches for multi-scene MiniMax H3 work.

Measured (tests/te_bench_aimdo.py, a 1,257-token I2VA prompt): a prompt change costs ~10 s of encoder compute,
but ~68 s after the DiT has evicted the encoder, because DynamicVRAM re-stages its 15 GB at ~263 MB/s.

1. Weight cache: the encoder's host weights (NVFP4 / AWQ / int8 / bf16 as stored) are copied once into pinned RAM
   (multistream.cache, group "te"). Its modules are then routed through the shadow cast (multistream.cast), which copies
   each weight from pinned RAM per call instead of going through aimdo's vbar re-staging.
2. Encoding cache: an LRU of encoder outputs keyed by encoder file, LoRA patch state, layer options and a content hash
   of the token/image input. Re-rendering an already encoded scene skips the encoder entirely. ComfyUI's execution
   cache only remembers the last result per node, so it loses earlier scenes as soon as the prompt changes.

Both hook the shared cond_stage_model (CLIP clones share it), so they survive SelectCLIPDevice and other clones.
"""
import contextlib
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict

import torch

import weakref

from . import cache as ms_cache
from . import hooks as ms_hooks
from . import cast as ms_cast
from .log import gib, log, ram

# roots (text encoder / VAE models) whose modules carry pinned-cache attributes; detach_all() frees their references
ATTACHED_ROOTS = weakref.WeakSet()

_ORIG_ATTR = "_h3ms_orig_encode_token_weights"
_WRAP_ATTR = "_h3ms_wrap_encode_token_weights"
_CFG_ATTR = "_h3ms_te_cfg"


def te_cache_key(clip):
    """Base key (encoder realpath, clip_type, model options) from CLIPLoader's reload factory, or None."""
    init = getattr(getattr(clip, "patcher", None), "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    args = init[1]
    paths = args[0]
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or len(paths) != 1 or not os.path.isfile(paths[0]):
        return None  # single-file encoders only
    clip_type = args[2] if len(args) > 2 else None
    opts = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
    return (os.path.realpath(paths[0]), str(clip_type), repr(sorted((str(k), str(v)) for k, v in opts.items())))


def _hash_into(obj, h):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().to("cpu").contiguous()
        h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            h.update(repr(k).encode())
            _hash_into(obj[k], h)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _hash_into(x, h)
        h.update(b"]")
    else:
        h.update(repr(obj).encode())


class CondCache:
    def __init__(self):
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self.lock:
            v = self.entries.get(key)
            if v is not None:
                self.entries.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return v

    def put(self, key, value, max_entries):
        with self.lock:
            self.entries[key] = value
            self.entries.move_to_end(key)
            while len(self.entries) > max_entries:
                self.entries.popitem(last=False)

    def clear(self):
        with self.lock:
            n = len(self.entries)
            self.entries.clear()
        log.info("[TextEncoder] output cache cleared: %d entr%s", n, "y" if n == 1 else "ies")
        return n

    def stats(self):
        with self.lock:
            return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses}


COND_CACHE = CondCache()


def _attach_weight_cache(cond_model, entry):
    """Route every module with a pinned copy through the shadow cast. Idempotent."""
    ms_cast.install_cast_hook()
    attached = 0
    for name, m in cond_model.named_modules():
        if not hasattr(m, "comfy_cast_weights"):
            continue
        cached = {p: entry[(name, p)] for p in ("weight", "bias") if (name, p) in entry}
        if cached:
            m._multistream_cached = cached
            m._multistream_rank = -1
            attached += 1
    if attached:
        ATTACHED_ROOTS.add(cond_model)
    return attached


def detach(cond_model):
    n = 0
    for m in cond_model.modules():
        if getattr(m, "_multistream_rank", None) == -1:
            del m._multistream_rank
            m.__dict__.pop("_multistream_cached", None)
            n += 1
    ATTACHED_ROOTS.discard(cond_model)
    return n


def detach_all():
    """Drop every module's reference to pinned cache tensors (call after clearing the weight caches)."""
    roots = list(ATTACHED_ROOTS)
    n = sum(detach(r) for r in roots)
    if roots:
        log.info("[Cache] detached cached weights from %d model(s), %d modules", len(roots), n)
    return n


def _sub_clip_options(cond_model):
    sub = getattr(cond_model, getattr(cond_model, "clip", ""), None)
    if sub is None:
        return ()
    return tuple(repr(getattr(sub, a, None)) for a in ("layer", "layer_idx", "return_projected_pooled"))


def uninstall(cond_model):
    """Put the original encode_token_weights back and forget the pinned weights."""
    detach(cond_model)
    return ms_hooks.uninstall(cond_model, _ORIG_ATTR, _WRAP_ATTR, ("encode_token_weights",), "TextEncoder")


def install(clip, enabled=True, weight_cache=True, cond_cache_entries=64, reserve_gib=0.0):
    """Hook clip.cond_stage_model.encode_token_weights. Calling again updates the configuration.

    enabled=False fully REMOVES the hook rather than just disabling it: the hook lives on the shared
    cond_stage_model, not on this node, so leaving it attached would keep it running (and logging)
    with the last configuration even after the node is gone. See multistream/hooks.py."""
    cond_model = clip.cond_stage_model
    if not enabled:
        if uninstall(cond_model):
            ms_cache.release_async("te", "text-encoder cache node disabled")
        else:
            log.info("[TextEncoder] node disabled: nothing was hooked")
        return
    base_key = te_cache_key(clip)
    cfg = {"base_key": base_key, "weight_cache": bool(weight_cache) and base_key is not None,
           "cond_entries": int(cond_cache_entries), "reserve_gib": float(reserve_gib)}
    if weight_cache and base_key is None:
        log.warning("[TextEncoder] weight cache unavailable: loader has no single-file reload factory")
    setattr(cond_model, _CFG_ATTR, cfg)
    log.info("[TextEncoder] cache node: model %s, weight cache %s, output cache %s, RAM reserve %.1f GiB",
             os.path.basename(base_key[0]) if base_key else type(cond_model).__name__,
             "on" if cfg["weight_cache"] else "off",
             f"{cfg['cond_entries']} entries" if cfg["cond_entries"] > 0 else "off", cfg["reserve_gib"])
    if not cfg["weight_cache"]:
        detach(cond_model)                      # drop the modules' pointers first ...
        ms_cache.release_async("te", "text-encoder weight cache switched off")   # ... then unpin the RAM
    if getattr(cond_model, _ORIG_ATTR, None) is not None:
        return
    original = cond_model.encode_token_weights
    setattr(cond_model, _ORIG_ATTR, original)

    def encode_token_weights(token_weight_pairs):
        c = getattr(cond_model, _CFG_ATTR)
        key = None
        digest = "-"
        t_call = time.perf_counter()
        if c["cond_entries"] > 0:
            h = hashlib.sha256()
            _hash_into(token_weight_pairs, h)
            digest = h.hexdigest()[:12]
            key = (c["base_key"] or id(cond_model), getattr(cond_model, "current_weight_patches_uuid", None),
                   _sub_clip_options(cond_model), h.hexdigest())
            hit = COND_CACHE.get(key)
            if hit is not None:
                s = COND_CACHE.stats()
                log.info("[TextEncoder] output cache HIT %s: encoder skipped (%.2fs incl. hashing); cache %d entries, "
                         "%d hits / %d misses", digest, time.perf_counter() - t_call, s["entries"], s["hits"], s["misses"])
                return tuple(dict(x) if isinstance(x, dict) else x for x in hit)
        wc = ms_cache.acquire(c["base_key"], c["reserve_gib"], "te") if c["weight_cache"] else None
        # in_use() spans the encode, not just the attach: the cast copies weights out of the pinned tensors
        # for every module call, so a release() must not unpin them until `original(...)` has returned.
        with wc.in_use() if wc is not None else contextlib.nullcontext() as held:
            if wc is not None and held is None:
                detach(cond_model)   # the cache is closing; this encode streams without it
                log.warning("[TextEncoder] weight cache is being released; encoding without it")
            elif wc is not None:
                t0 = time.perf_counter()
                filled_before = len(wc.blocks)
                entry = wc.get_block(0, cond_model)
                if entry is not None:
                    _attach_weight_cache(cond_model, entry)
                    if not filled_before:
                        log.info("[TextEncoder] weight cache filled: %s pinned in %.1fs | %s",
                                 gib(wc.bytes), time.perf_counter() - t0, ram())
                else:
                    detach(cond_model)
                    log.warning("[TextEncoder] weight cache not available for this encode (disabled or short on RAM)")
            t_enc = time.perf_counter()
            out = original(token_weight_pairs)
            enc_s = time.perf_counter() - t_enc
        tokens = out[0].shape[1] if hasattr(out[0], "shape") and out[0].dim() > 1 else "?"
        if key is not None:
            COND_CACHE.put(key, tuple(dict(x) if isinstance(x, dict) else x for x in out), c["cond_entries"])
        s = COND_CACHE.stats()
        log.info("[TextEncoder] encoded %s tokens in %.1fs (output cache %s: MISS %s, %d/%d entries; weight cache %s)",
                 tokens, enc_s, "on" if key is not None else "off", digest, s["entries"], c["cond_entries"],
                 "on" if c["weight_cache"] else "off")
        return out

    cond_model.encode_token_weights = encode_token_weights
    setattr(cond_model, _WRAP_ATTR, {"encode_token_weights": encode_token_weights})
    ms_hooks.remember(cond_model, "TextEncoder", _ORIG_ATTR, _WRAP_ATTR, ("encode_token_weights",))
