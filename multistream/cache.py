# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Process-wide pinned-RAM cache of DiT block weights for the split ranks.

Why: without it, every rank copies block weights from the checkpoint's mmap. When those file pages are not in
the OS page cache (cold start, or DynamicVRAM marked them cold) the first split step reads ~20 GB from disk.

What it is:
  * keyed by checkpoint realpath + size + mtime + model options, so it survives new prompts and model
    unload/reload, and is invisible to ComfyUI's model management (never evicted by it);
  * base weights only (as stored on the host: int8 qdata + scales, norms, adaln); LoRA is still applied on top
    per step by the shadow cast;
  * each tensor is a plain CPU allocation registered with cudaHostRegister at its exact size (torch's pinned
    host allocator rounds up to powers of two);
  * filled lazily per block (per-block lock) plus a background prefill thread;
  * one checkpoint at a time: acquiring a different key evicts the others; a RAM guard refuses to fill when
    fewer than RESERVE_GIB would remain available.
"""
import contextlib
import dataclasses
import logging
import os
import threading
import time
from collections import defaultdict

import torch

from comfy.quant_ops import QuantizedTensor

from .log import gib, log, ram, ram_available

DEFAULT_RESERVE_GIB = 0.0  # fill whenever the RAM exists; the node exposes a reserve for users who want headroom
MIN_HEADROOM_GIB = 4.0     # never fill the last few GiB: at the container limit the process stalls in page reclaim
_GIB = 2**30
_LOCK = threading.Lock()
_CACHES = {}


def _register(t):
    n = t.numel() * t.element_size()
    if n == 0:
        return False
    r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), n, 0)
    if int(getattr(r, "value", r)) != 0:
        raise RuntimeError(f"cudaHostRegister failed ({r}) for {n} bytes")
    return True


def _pinned_copy(t, keep):
    out = torch.empty(t.shape, dtype=t.dtype)
    out.copy_(t)
    if _register(out):
        keep.append(out)
    return out


def _pin_param(t, keep):
    if isinstance(t, QuantizedTensor):
        qd = _pinned_copy(t._qdata, keep)
        repl = {}
        for f in dataclasses.fields(t._params):
            v = getattr(t._params, f.name)
            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                repl[f.name] = _pinned_copy(v, keep)
        return QuantizedTensor(qd, t._layout_cls, dataclasses.replace(t._params, **repl))
    return _pinned_copy(t.data if isinstance(t, torch.nn.Parameter) else t, keep)


def _nbytes(t):
    if isinstance(t, QuantizedTensor):
        n = t._qdata.numel() * t._qdata.element_size()
        for f in dataclasses.fields(t._params):
            v = getattr(t._params, f.name)
            if isinstance(v, torch.Tensor):
                n += v.numel() * v.element_size()
        return n
    return t.numel() * t.element_size()


def _host_params(block):
    """(module_path, param_name, tensor) for every CPU-resident cast weight in a block."""
    for name, m in block.named_modules():
        if not hasattr(m, "comfy_cast_weights"):
            continue
        for p in ("weight", "bias"):
            t = getattr(m, p, None)
            if t is None or getattr(t, "is_meta", False) or t.device.type != "cpu":
                continue
            yield name, p, t


class WeightCache:
    def __init__(self, key, reserve_gib=DEFAULT_RESERVE_GIB):
        self.key = key
        self.reserve_gib = float(reserve_gib)
        self.blocks = {}
        self._block_locks = defaultdict(threading.Lock)
        self._keep = []
        self.bytes = 0
        self.fill_seconds = 0.0
        self.disabled = False
        self._prefill = None
        self._closed = False
        self._users = 0
        self._idle = threading.Condition()

    @contextlib.contextmanager
    def in_use(self):
        """Hold while reading this cache's pinned tensors (a split step, a TE encode, a VAE call).

        close() unregisters those pages with cudaHostUnregister, and doing that under an in-flight async
        H2D copy out of them is a memory-safety bug, not just a slow path -- the prefetch stages the next
        block on a side stream, so a reader can still have a DMA running after its Python call returned.
        The counter is what close() waits on; entering after close() has begun is refused, so a late
        reader streams from the checkpoint instead of from memory that is about to go away."""
        with self._idle:
            if self._closed:
                yield None
                return
            self._users += 1
        try:
            yield self
        finally:
            with self._idle:
                self._users -= 1
                if self._users == 0:
                    self._idle.notify_all()

    def get_block(self, idx, block):
        entry = self.blocks.get(idx)
        if entry is not None or self.disabled:
            return entry
        with self._block_locks[idx]:
            entry = self.blocks.get(idx)
            if entry is not None or self.disabled or self._closed:
                return None if self._closed else entry
            params = list(_host_params(block))
            need = sum(_nbytes(t) for _, _, t in params)
            avail = ram_available()[0]
            reserve = max(self.reserve_gib, MIN_HEADROOM_GIB)
            if avail - need < reserve * _GIB:
                self.disabled = True
                log.warning("[Cache] %s weight cache DISABLED: block %d needs %s, only %.1f GiB available "
                            "(keeping %.1f GiB free) -> streaming without cache | %s", self.key[0], idx, gib(need),
                            avail / _GIB, reserve, ram())
                return None
            t0 = time.perf_counter()
            entry = {(name, p): _pin_param(t, self._keep) for name, p, t in params}
            dt = time.perf_counter() - t0
            self.fill_seconds += dt
            self.bytes += need
            self.blocks[idx] = entry
            log.debug("[Cache] %s block %d pinned: %d tensors, %s in %.2fs (%.2f GiB/s)", self.key[0], idx,
                      len(entry), gib(need), dt, need / _GIB / dt if dt > 0 else 0.0)
            return entry

    def start_prefill(self, blocks, skip_below=0):
        """Fill in the background. `skip_below` is the number of leading blocks another tier owns --
        the VRAM block cache pins those on the card, so pinning them here as well would duplicate
        exactly what the tiering exists to avoid."""
        if self.disabled or self._prefill is not None or len(self.blocks) >= len(blocks) - skip_below:
            return

        def run():
            for i, b in enumerate(blocks):
                if i < skip_below:
                    continue
                if self._closed or self.disabled:
                    return
                try:
                    self.get_block(i, b)
                except Exception:
                    log.exception("[Cache] %s weight cache prefill failed at block %d; cache disabled", self.key[0], i)
                    self.disabled = True
                    return
            log.info("[Cache] %s weight cache complete: %d blocks, %s pinned in %.1fs | %s",
                     os.path.basename(self.key[1]), len(self.blocks), gib(self.bytes), self.fill_seconds, ram())

        self._prefill = threading.Thread(target=run, name="h3ms-cache-prefill", daemon=True)
        self._prefill.start()

    def close(self, wait_seconds=600):
        """Unregister and drop every pinned tensor. Waits for in-flight readers (see in_use); returns False
        and keeps the cache pinned if they do not finish, because unregistering under them would be unsafe."""
        log.info("[Cache] releasing %s weight cache %s: %s", self.key[0], os.path.basename(self.key[1]), gib(self.bytes))
        with self._idle:
            self._closed = True   # no new readers from here on
            if not self._idle.wait_for(lambda: self._users == 0, timeout=wait_seconds):
                self._closed = False
                log.error("[Cache] %s weight cache %s STILL IN USE after %.0fs by %d reader(s); left pinned "
                          "(%s). Retry when the prompt is idle.", self.key[0], os.path.basename(self.key[1]),
                          float(wait_seconds), self._users, gib(self.bytes))
                return False
        if self._prefill is not None and self._prefill is not threading.current_thread():
            self._prefill.join(timeout=600)
        for lock in list(self._block_locks.values()):
            with lock:
                pass
        cudart = torch.cuda.cudart()
        for t in self._keep:
            cudart.cudaHostUnregister(t.data_ptr())
        self._keep.clear()
        self.blocks.clear()
        self.bytes = 0
        return True

    def stats(self):
        return {"group": self.key[0].split(":", 1)[0], "file": os.path.basename(self.key[1]),
                "key": list(self.key), "blocks": len(self.blocks), "GiB": round(self.bytes / _GIB, 3),
                "fill_seconds": round(self.fill_seconds, 1), "disabled": self.disabled,
                "reserve_GiB": self.reserve_gib}


def full_key(base_key):
    path = base_key[0]
    st = os.stat(path)
    return (path, st.st_size, st.st_mtime_ns) + tuple(base_key[1:])


def acquire(base_key, reserve_gib=DEFAULT_RESERVE_GIB, group="dit"):
    """Cache for this checkpoint within a group ("dit", "te"); a different checkpoint in the same group evicts the
    old one first. None if the file is gone. The reserve is updated on every call (node parameter)."""
    try:
        key = (group,) + full_key(base_key)
    except OSError:
        return None
    with _LOCK:
        cache = _CACHES.get(key)
        if cache is None:
            for other in [k for k in _CACHES if k[0] == group]:
                log.info("[Cache] evicting %s weight cache %s (a different file was requested)", group,
                         os.path.basename(other[1]))
                if not _CACHES[other].close():
                    continue   # readers still on it; leave it registered and let the next acquire retry
                _CACHES.pop(other)
            cache = WeightCache(key, reserve_gib)
            _CACHES[key] = cache
            log.info("[Cache] new %s weight cache for %s (reserve %.1f GiB) | %s", group.split(":", 1)[0],
                     os.path.basename(key[1]), float(reserve_gib), ram())
        else:
            if cache.disabled and float(reserve_gib) < cache.reserve_gib:
                cache.disabled = False  # a lower reserve re-enables filling
            cache.reserve_gib = float(reserve_gib)
        return cache


def _group_of(key):
    return key[0].split(":", 1)[0]


def release(group, reason=""):
    """Close every cache in `group` ("te", "dit", "vae" -- the VAE's per-file "vae:<path>" groups all match "vae").

    This is what turning a node's `weight_cache` off calls. Before this existed, switching it off only stopped
    the cache being used and refilled: the WeightCache stayed in _CACHES with every cudaHostRegister'd page
    still pinned, so the RAM never came back and, being pinned, could not even be swapped.
    """
    freed, kept = 0, 0
    with _LOCK:
        for key in [k for k in _CACHES if k[0] == group or _group_of(k) == group]:
            cache = _CACHES[key]
            n = cache.bytes
            if cache.close():
                _CACHES.pop(key)
                freed += n
            else:
                kept += n
    if freed or kept:
        log.info("[Cache] released %s weight cache(s)%s: %s freed%s | %s", group,
                 f" ({reason})" if reason else "", gib(freed),
                 f", {gib(kept)} still in use and left pinned" if kept else "", ram())
    return freed


def release_async(group, reason=""):
    """release() off the calling thread. install() runs inside prompt execution, where a reader may hold the
    cache for the length of a sampler step; blocking the node there would stall the prompt."""
    t = threading.Thread(target=release, args=(group, reason), name=f"h3ms-cache-release-{group}", daemon=True)
    t.start()
    return t


def clear_all():
    freed, kept = 0, 0
    with _LOCK:
        for key in list(_CACHES):
            cache = _CACHES[key]
            n = cache.bytes
            if cache.close():
                _CACHES.pop(key)
                freed += n
            else:
                kept += n   # readers still on it; close() logged which and left it pinned
    log.info("[Cache] all weight caches cleared: %s released%s | %s", gib(freed),
             f", {gib(kept)} still in use and left pinned" if kept else "", ram())
    return freed


def all_stats():
    with _LOCK:
        return [c.stats() for c in _CACHES.values()]
