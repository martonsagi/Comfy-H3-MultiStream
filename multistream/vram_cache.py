# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Keep the first N transformer blocks' weights resident in VRAM, sized at run time.

What this is. THE FIRST TIER OF THE WEIGHT CACHE, not a separate cache. multistream.cache pins block
weights in host RAM; this puts as many leading blocks as fit on the cards instead, and the host tier
takes the overflow. split.run_split_stack decides the boundary before the host prefill starts and
passes it as start_prefill(skip_below=...), so a block the cards own is never pinned in RAM as well.

Why. The prefetch already hides the host -> device copy behind compute (README: keeping more weights
in VRAM "gave no further gain"), so this is NOT a speed feature. What it buys is HOST memory: on a
2x16 GiB box the difference between pinning all 50 blocks (18 GiB unswappable) and pinning only the
overflow. A block costs about 2x its on-disk size once staged in model dtype -- 791 MiB against 370
measured on 2026-09-16 -- so expect the tier to hold fewer blocks than file sizes suggest.

Sizing is dynamic because the working set is not: it moves with resolution and clip length. The
budget is measured after multistream.split.free_secondary_ranks has evicted ComfyUI's models from the
secondary ranks, so it reflects a card that is actually free.

TWO PROPERTIES THIS MUST HAVE, both learned the hard way on 2026-09-16:

  * IT MUST EVICT ITSELF. ComfyUI's free_memory only walks current_loaded_models, so allocations made
    here are invisible to it and it can never reclaim them -- and worse, they make get_free_memory
    report less, so ComfyUI starts unloading ITS models to compensate. Release happens on the
    ModelPatcher's ON_CLEANUP callback, which comfy/samplers.py fires from a `finally` at the end of
    outer_sample: before the VAE decode node runs, and on the interrupt path too.
  * IT MUST NOT OUTLIVE A SAMPLING RUN. The staged copies are base weights; LoRA is applied per call
    on top. Patches can change between prompts, so holding them across runs would silently apply a
    stale base. ON_CLEANUP covers this for free.

Blocks are filled in index order and never evicted individually: the block loop is a sequential
0 -> 49 scan every step, the worst case for LRU, which would throw away exactly the block needed next.
"""
import os
import threading

import torch

from .log import gib, log

_LOCK = threading.Lock()
_BLOCKS = {}     # (device str, block index) -> {(module path, param): device tensor}
_BUDGET = {}     # device str -> bytes still spendable this run
_HELD = {}       # device str -> bytes currently held
_PLANNED = set()   # device strs budgeted for the CURRENT run; cleared by release_all
_INITIAL = {}      # device str -> the budget as first planned, BEFORE any block was taken
_PEAKS = {}        # device str -> last observed step working set, bytes (survives runs)
# Never spend the last of a card. ComfyUI budgets against get_free_memory, which DOES see these
# allocations even though free_memory can never reclaim them, so leaving it nothing makes it start
# unloading its own models. The node passes a value; H3MS_VRAM_SAFETY_GIB overrides the default for
# anyone who would rather not edit the graph.
SAFETY_GIB = float(os.environ.get("H3MS_VRAM_SAFETY_GIB", "2.0"))
DEFAULT_WORKING_GIB = 4.0   # before any step has run; H3 has been observed at 2.1-3.1 GiB


def _key(device, index):
    return (str(device), int(index))


def note_peak(device, peak_bytes):
    """Record a step's PyTorch peak so the next run budgets against a measured working set.

    The blocks THIS CACHE holds are subtracted first. torch.max_memory_allocated counts them, so
    feeding the raw figure back would treat the cache as part of the working set: each run would
    infer a larger working set, budget less, cache fewer blocks, and converge on zero."""
    if not peak_bytes:
        return
    dev = str(device)
    with _LOCK:
        held = _HELD.get(dev, 0)
    _PEAKS[dev] = max(0, int(peak_bytes) - held)


def plan(device, working_bytes=None, safety_gib=None):
    """Budget `device` for this run: free now, less the step working set and a safety margin.

    Idempotent per run, and it keeps returning the INITIAL budget rather than what is left.
    run_split_stack derives the tier boundary from this on EVERY step: when it returned the remaining
    budget the boundary collapsed to 0 once the first step had spent it, so from step 2 the host
    cache re-pinned all 50 blocks while the VRAM tier sat stranded and unused (measured 2026-09-16).
    release_all() re-arms it.
    `safety_gib` None means the module default (H3MS_VRAM_SAFETY_GIB, else 2.0).
    """
    dev = str(device)
    safety = SAFETY_GIB if safety_gib is None else max(0.0, float(safety_gib))
    with _LOCK:
        if dev in _PLANNED:
            return _INITIAL.get(dev, 0)
        _PLANNED.add(dev)
    if working_bytes is None:
        working_bytes = _PEAKS.get(dev, DEFAULT_WORKING_GIB * 2**30)
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        log.exception("[VRAM cache] cannot read free memory on %s; disabled for this run", dev)
        with _LOCK:
            _BUDGET[dev] = 0
        return 0
    budget = int(free) - int(working_bytes) - int(safety * 2**30)
    budget = max(0, budget)
    with _LOCK:
        _BUDGET[dev] = budget
        _INITIAL[dev] = budget
        held = _HELD.get(dev, 0)
    log.info("[VRAM cache] %s: %s free, working set %s, safety %.1f GiB -> budget %s%s",
             dev, gib(free), gib(working_bytes), safety, gib(budget),
             f" ({gib(held)} already held)" if held else "")
    return budget


def get(device, index):
    with _LOCK:
        return _BLOCKS.get(_key(device, index))


def offer(device, index, staged, nbytes):
    """Keep `staged` for the rest of the run if the budget allows. True when it was kept."""
    if not staged:
        return False
    dev = str(device)
    with _LOCK:
        if _BUDGET.get(dev, 0) < nbytes:
            return False
        _BUDGET[dev] -= nbytes
        _HELD[dev] = _HELD.get(dev, 0) + nbytes
        _BLOCKS[_key(device, index)] = staged
        return True


def staged_bytes(staged):
    """What a staged block really costs. Must agree with cast.staged_size, which the tiering uses to
    pick the boundary: if offer() charges more than the boundary assumed, blocks inside the VRAM tier
    get refused while the host tier has already skipped them, and they end up in neither."""
    from .cast import tensor_bytes
    return sum(tensor_bytes(t) for t in staged.values())


def release_all(reason="sampling finished"):
    """Drop every resident block. Nothing else can: these allocations are invisible to ComfyUI."""
    with _LOCK:
        n, held = len(_BLOCKS), sum(_HELD.values())
        _BLOCKS.clear()
        _BUDGET.clear()
        _INITIAL.clear()
        _HELD.clear()
        _PLANNED.clear()
    if n:
        log.info("[VRAM cache] released %d resident block(s), %s (%s)", n, gib(held), reason)
    return {"blocks": n, "freed_GiB": round(held / 2**30, 2)}


def stats():
    with _LOCK:
        per = {}
        for (dev, _idx) in _BLOCKS:
            per[dev] = per.get(dev, 0) + 1
        return [{"device": d, "blocks": c, "GiB": round(_HELD.get(d, 0) / 2**30, 2)}
                for d, c in sorted(per.items())]
