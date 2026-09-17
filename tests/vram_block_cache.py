# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""VRAM block cache: budgeting, residency, and the self-eviction it cannot do without.

    python tests/vram_block_cache.py

CPU only; torch.cuda.mem_get_info is stubbed. These allocations are invisible to ComfyUI's model
management (free_memory walks current_loaded_models only), so if release_all ever stops working
nothing else will reclaim them -- hence the emphasis on eviction here.
"""
import os
import sys

import torch

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)          # staged_bytes reaches into multistream.cast, which needs comfy
sys.path.insert(0, PACK)

from multistream import vram_cache as vc   # noqa: E402

G = 2**30
FREE = {}


def fake_mem_get_info(device):
    return FREE[str(device)], 16 * G


def blk(mib):
    """A staged block of about `mib` MiB, shaped like stage_block's output."""
    n = int(mib * 2**20 / 2 / 1024)
    return {("attn.qkv_proj", "weight"): torch.zeros(n, 1024, dtype=torch.float16)}


def main():
    torch.cuda.mem_get_info = fake_mem_get_info
    vc.release_all("reset")

    # --- budget = free - working set - safety --------------------------------
    FREE["cuda:0"] = 14 * G
    budget = vc.plan("cuda:0", working_bytes=3 * G)
    assert budget == 14 * G - 3 * G - int(vc.SAFETY_GIB * G), budget / G
    print(f"  budget on 14 GiB free, 3 GiB working, {vc.SAFETY_GIB} GiB safety = {budget/G:.1f} GiB")

    # --- idempotent per run: a second plan must not re-budget -----------------
    again = vc.plan("cuda:0", working_bytes=3 * G)
    assert again == budget, (again, budget)
    print("  plan() is idempotent within a run (per-step calls do not shrink the budget)")

    # --- REGRESSION: plan() must keep returning the INITIAL budget ------------
    # run_split_stack derives the tier boundary from plan() on EVERY step. When it returned the
    # remaining budget, the boundary collapsed to 0 after the first step had spent it, the host cache
    # re-pinned all 50 blocks, and the VRAM tier sat stranded and unused.
    b = blk(296)
    for i in range(5):
        assert vc.offer("cuda:0", i, b, vc.staged_bytes(b))
    assert vc.plan("cuda:0", working_bytes=3 * G) == budget, \
        "plan() returned the remaining budget; the tier boundary would collapse mid-run"
    print("  plan() still reports the initial budget after blocks have been taken")
    vc.release_all("reset"); vc.plan("cuda:0", working_bytes=3 * G)

    # --- fills in order until the budget runs out ----------------------------
    kept = []
    for i in range(50):
        b = blk(296)                                  # Dasiwa: 296 MiB per block per rank
        if vc.offer("cuda:0", i, b, vc.staged_bytes(b)):
            kept.append(i)
    assert kept == list(range(len(kept))), "blocks must fill in index order"
    assert len(kept) == int(budget / (296 * 2**20)), (len(kept), budget / (296 * 2**20))
    assert vc.get("cuda:0", 0) is not None and vc.get("cuda:0", 49) is None
    print(f"  filled blocks 0..{kept[-1]} of 50, then refused (budget exhausted)")

    # --- a second card budgets independently ---------------------------------
    FREE["cuda:1"] = 6 * G
    b1 = vc.plan("cuda:1", working_bytes=3 * G)
    assert b1 == 6 * G - 3 * G - int(vc.SAFETY_GIB * G), b1 / G
    assert b1 < budget, "the tighter card must get the smaller budget"
    print(f"  cuda:1 with 6 GiB free budgets {b1/G:.1f} GiB, independently of cuda:0")

    # --- release frees everything and re-arms planning -----------------------
    stats = vc.stats()
    assert sum(s["blocks"] for s in stats) == len(kept), stats
    out = vc.release_all("test")
    assert out["blocks"] == len(kept) and out["freed_GiB"] > 0, out
    assert vc.stats() == [] and vc.get("cuda:0", 0) is None
    FREE["cuda:0"] = 2 * G                             # a card that is now nearly full
    assert vc.plan("cuda:0", working_bytes=3 * G) == 0, "must refuse to budget a full card"
    print(f"  release_all freed {out['blocks']} blocks / {out['freed_GiB']} GiB and re-armed planning")

    # --- a measured peak feeds the next run's budget -------------------------
    vc.release_all("reset")
    vc.note_peak("cuda:0", 5 * G)
    FREE["cuda:0"] = 14 * G
    assert vc.plan("cuda:0") == 14 * G - 5 * G - int(vc.SAFETY_GIB * G)
    print("  a recorded step peak is used as the next run's working set")

    # --- an unreadable device disables itself rather than guessing -----------
    vc.release_all("reset")
    torch.cuda.mem_get_info = lambda d: (_ for _ in ()).throw(RuntimeError("no such device"))
    assert vc.plan("cuda:9") == 0
    assert vc.offer("cuda:9", 0, blk(1), 2**20) is False
    print("  a card whose free memory cannot be read budgets 0 and caches nothing")

    # --- REGRESSION: the peak must exclude the cache's own blocks -------------
    # max_memory_allocated counts resident blocks; feeding that back as the working set would make
    # each run budget less than the last and converge on caching nothing.
    torch.cuda.mem_get_info = fake_mem_get_info
    vc.release_all("reset")
    FREE["cuda:0"] = 14 * G
    vc.plan("cuda:0", working_bytes=3 * G)
    b = blk(2048)
    assert vc.offer("cuda:0", 0, b, vc.staged_bytes(b))
    vc.note_peak("cuda:0", 5 * G)                      # 3 GiB of real work + 2 GiB of cache
    vc.release_all("test")
    FREE["cuda:0"] = 14 * G
    budget = vc.plan("cuda:0")
    assert budget == 14 * G - 3 * G - int(vc.SAFETY_GIB * G), \
        f"peak feedback counted the cache itself: budget {budget/G:.2f} GiB"
    print("  note_peak subtracts resident blocks (no shrinking feedback loop)")

    # --- the safety margin is a parameter, and the env var sets its default ---
    torch.cuda.mem_get_info = fake_mem_get_info
    vc.release_all("reset")
    FREE["cuda:0"] = 14 * G
    tight = vc.plan("cuda:0", working_bytes=3 * G, safety_gib=6.0)
    assert tight == 14 * G - 3 * G - 6 * G, tight / G
    vc.release_all("reset")
    assert vc.plan("cuda:0", working_bytes=3 * G, safety_gib=0.0) == 11 * G
    vc.release_all("reset")
    assert vc.plan("cuda:0", working_bytes=3 * G, safety_gib=-5) == 11 * G, "negative clamps to 0"
    vc.release_all("reset")
    assert vc.plan("cuda:0", working_bytes=3 * G) == 14 * G - 3 * G - int(vc.SAFETY_GIB * G), \
        "None must fall back to the module default"
    print(f"  safety margin: 6.0 -> {tight/G:.1f} GiB budget, 0.0 -> 11.0, None -> module default "
          f"({vc.SAFETY_GIB} GiB)")

    vc.release_all("reset")
    print("PASS: vram block cache")


if __name__ == "__main__":
    main()
