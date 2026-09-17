# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""The two weight-cache tiers must partition the blocks, not duplicate them.

    python tests/weight_cache_tiering.py

VRAM holds blocks 0..K-1, host RAM holds K..49. A block on the cards must NOT also be pinned in RAM:
before this was wired, cache.get_block() ran for every block and start_prefill() pinned all 50 in the
background, so turning the VRAM tier on added residency without removing anything (2026-09-16).
"""
import ast
import os
import sys
import threading
import time
import types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

PINNED = []


def build_cache():
    """WeightCache with its pinning stubbed, so we can see which blocks it would pin."""
    src = open(os.path.join(PACK, "multistream", "cache.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if getattr(n, "name", None) == "WeightCache")
    import collections, contextlib
    g = {"threading": threading, "time": time, "os": os, "contextlib": contextlib,
         "defaultdict": collections.defaultdict, "torch": types.SimpleNamespace(),
         "log": types.SimpleNamespace(**{k: (lambda *a, **kw: None)
                                         for k in ("info", "warning", "error", "debug", "exception")}),
         "gib": lambda n: f"{n}B", "ram": lambda: "", "ram_available": lambda: (10**12,),
         "_host_params": lambda b: [], "QuantizedTensor": type("QT", (), {}),
         "MIN_HEADROOM_GIB": 0.0, "_GIB": 2**30, "DEFAULT_RESERVE_GIB": 0.0,
         "_pin_param": lambda t, keep: t, "_nbytes": lambda t: 0, "dataclasses": None}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "cache.py", "exec"), g)
    wc = g["WeightCache"](("dit", "/ckpt", 1, 1))
    real_get = wc.get_block

    def spy(idx, block):
        PINNED.append(idx)
        return real_get(idx, block)
    wc.get_block = spy
    return wc


def main():
    blocks = [f"block{i}" for i in range(50)]

    # --- start_prefill must skip the blocks the VRAM tier owns ---------------
    PINNED.clear()
    wc = build_cache()
    wc.start_prefill(blocks, skip_below=20)
    wc._prefill.join(timeout=10)
    assert PINNED and min(PINNED) == 20 and max(PINNED) == 49, (min(PINNED), max(PINNED))
    assert len(PINNED) == 30, len(PINNED)
    print(f"  start_prefill(skip_below=20): pinned blocks {min(PINNED)}-{max(PINNED)} only ({len(PINNED)} of 50)")

    # --- skip_below=0 is the old behaviour, all 50 ---------------------------
    PINNED.clear()
    wc = build_cache()
    wc.start_prefill(blocks, skip_below=0)
    wc._prefill.join(timeout=10)
    assert len(PINNED) == 50 and min(PINNED) == 0, (len(PINNED), min(PINNED))
    print("  skip_below=0: all 50 pinned (host-only mode unchanged)")

    # --- the whole model in VRAM means the host tier does nothing ------------
    PINNED.clear()
    wc = build_cache()
    wc.start_prefill(blocks, skip_below=50)
    if wc._prefill is not None:
        wc._prefill.join(timeout=10)
    assert PINNED == [], PINNED
    print("  skip_below=50: host tier pins nothing")

    # --- the split's per-block path takes exactly one tier -------------------
    src = open(os.path.join(PACK, "multistream", "split.py"), encoding="utf-8").read()
    body = src[src.index("def run_split_stack("):]
    assert "entry = None if tier_vram else (cache.get_block(" in body, \
        "a VRAM-resident block must not be pinned in host RAM as well"
    assert "cache.start_prefill(dm.blocks, skip_below=vram_upto)" in body, \
        "the host prefill must be told where the VRAM tier ends"
    assert "min(fits)" in body, \
        "a block is only host-skippable when EVERY rank has it resident"
    print("  split.py: one tier per block, prefill bounded, boundary is min() across ranks")

    print("PASS: weight cache tiering")


if __name__ == "__main__":
    main()
