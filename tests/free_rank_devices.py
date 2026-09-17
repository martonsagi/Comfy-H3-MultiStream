# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""free_rank_devices: clear every rank GPU, protecting the DiT on the primary.

    python tests/free_secondary.py

CPU only, ComfyUI stubbed. ComfyUI frees a device only when something loads onto it through its
loader; the split's secondary ranks allocate straight through torch, so nothing ever evicts a text
encoder parked there. On 2026-09-16 that left ~8 GiB pinned on GPU0 and OOM'd the VAE split worker.
"""
import ast
import os
import sys
import types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

FREED = []
FREE_MEM = {}
KEPT = []


def build(fail_on=None):
    src = open(os.path.join(PACK, "multistream", "split.py"), encoding="utf-8").read()
    body = [n for n in ast.parse(src).body
            if getattr(n, "name", None) in ("free_rank_devices", "_loaded_entries_for")]

    def free_memory(required, dev, keep_loaded=()):
        KEPT.append(list(keep_loaded))
        if dev == fail_on:
            raise RuntimeError("simulated failure")
        FREED.append((str(dev), required))
        FREE_MEM[dev] = FREE_MEM.get(dev, 0) + 8 * 2**30      # pretend 8 GiB came back

    g = {"comfy": types.SimpleNamespace(model_management=types.SimpleNamespace(
            free_memory=free_memory, get_free_memory=lambda d: FREE_MEM.get(d, 0),
            current_loaded_models=LOADED)),
         "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
            get_device_properties=lambda d: types.SimpleNamespace(total_memory=16 * 2**30))),
         "log": types.SimpleNamespace(**{k: (lambda *a, **kw: None)
                                         for k in ("info", "warning", "error", "exception", "debug")})}
    exec(compile(ast.Module(body=body, type_ignores=[]), "split.py", "exec"), g)
    return g["free_rank_devices"]


class Dev:
    def __init__(self, type_, index): self.type, self.index = type_, index
    def __eq__(self, o): return isinstance(o, Dev) and (self.type, self.index) == (o.type, o.index)
    def __hash__(self): return hash((self.type, self.index))
    def __repr__(self): return f"{self.type}:{self.index}"


LOADED = []


def main():
    cuda0, cuda1 = Dev("cuda", 0), Dev("cuda", 1)

    # --- the primary is never freed -----------------------------------------
    FREED.clear(); FREE_MEM.clear()
    out = build()([cuda1, cuda0], primary=cuda1)
    assert [d for d, _ in FREED] == ["cuda:0"], FREED
    assert out and "cuda:0" in out[0], out
    print(f"  2 ranks, primary cuda:1 -> freed only cuda:0  ({out[0]})")

    # flip which card is primary; the other one gets freed
    FREED.clear(); FREE_MEM.clear()
    build()([cuda0, cuda1], primary=cuda0)
    assert [d for d, _ in FREED] == ["cuda:1"], FREED
    print("  primary cuda:0 -> freed only cuda:1")

    # --- a single-rank plan frees nothing ------------------------------------
    FREED.clear(); FREE_MEM.clear()
    assert build()([cuda1], primary=cuda1) == []
    assert FREED == [], FREED
    print("  1 rank, no dm -> nothing freed")

    # --- duplicates and non-cuda devices are skipped -------------------------
    FREED.clear(); FREE_MEM.clear()
    build()([cuda1, cuda0, cuda0, Dev("cpu", None)], primary=cuda1)
    assert [d for d, _ in FREED] == ["cuda:0"], f"deduplicate and skip cpu: {FREED}"
    print("  repeated rank devices freed once; cpu skipped")

    # --- a failure on one card must not abort the others ---------------------
    FREED.clear(); FREE_MEM.clear()
    cuda2 = Dev("cuda", 2)
    out = build(fail_on=cuda0)([cuda1, cuda0, cuda2], primary=cuda1)
    assert [d for d, _ in FREED] == ["cuda:2"], FREED
    assert len(out) == 1 and "cuda:2" in out[0], out
    print("  a failing card is logged and skipped; the rest still freed")

    # --- REGRESSION 2026-09-16: a modest request frees almost nothing ---------
    # free_memory() stops as soon as the requested amount is available, so asking for 4 GiB on a card
    # with 3.74 GiB free evicted 0.31 GiB and left an 8 GiB text encoder in place. The default must
    # ask for the whole card.
    FREED.clear(); FREE_MEM.clear()
    build()([cuda1, cuda0], primary=cuda1)
    asked = dict(FREED)["cuda:0"] if FREED else 0
    assert asked >= 16 * 2**30, f"must request the whole card, asked for {asked/2**30:.2f} GiB"
    print(f"  default request is the whole card ({asked/2**30:.0f} GiB), not a token amount")

    FREED.clear(); FREE_MEM.clear()
    build()([cuda1, cuda0], primary=cuda1, need_bytes=2 * 2**30)
    assert dict(FREED)["cuda:0"] == 2 * 2**30, FREED
    print("  an explicit need_bytes is still honoured")

    # --- the primary IS cleared when the DiT can be protected ---------------
    class DM: pass
    dm = DM()
    lm_dit = types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm)))
    lm_te = types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=None)))
    LOADED[:] = [lm_dit, lm_te]
    FREED.clear(); FREE_MEM.clear(); KEPT.clear()
    out = build()([cuda1, cuda0], primary=cuda1, dm=dm)
    assert sorted(d for d, _ in FREED) == ["cuda:0", "cuda:1"], FREED
    assert all(lm_dit in k for k in KEPT), "the DiT must be in keep_loaded on every card"
    assert all(lm_te not in k for k in KEPT), "the text encoder must NOT be protected"
    assert any("(primary)" in f for f in out), out
    print("  primary is cleared too, with the DiT in keep_loaded and the encoder evictable")

    # --- but never when the DiT cannot be identified -------------------------
    LOADED[:] = [lm_te]
    FREED.clear(); FREE_MEM.clear(); KEPT.clear()
    build()([cuda1, cuda0], primary=cuda1, dm=dm)
    assert [d for d, _ in FREED] == ["cuda:0"], f"primary must be skipped when unprotectable: {FREED}"
    print("  primary skipped when the DiT's loaded entry cannot be found (never evict our own model)")

    LOADED[:] = []
    print("PASS: free_rank_devices")


if __name__ == "__main__":
    main()
