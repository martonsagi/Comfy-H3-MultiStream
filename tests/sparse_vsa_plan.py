# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""VSA plumbing: cube-plan round trip, per-rank context hoisting, gate row slicing.

    python tests/sparse_vsa_plan.py

CPU only. Uses a REAL PackedLayout and a REAL SparseAttnPatch.vsa_plan, so the permutation under test
is the one that will run. Does not touch the CUDA kernel; tests/sparse_head_split_parity.py covers
the numerics of the head split itself.
"""
import os
import sys
import threading

import torch

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "ComfyUI-H3-MultiStream"))

import comfy_extras.nodes_sparse_attention as nsa            # noqa: E402
from comfy.ldm.minimax.model import PackedLayout             # noqa: E402
from multistream import sparse as ms_sparse                  # noqa: E402


def make_patch(vsa=True):
    return nsa.SparseAttnPatch(tau=1.3, topk_ratio=0.10, vsa=vsa, sigma_start=1.0, sigma_end=0.0,
                               min_tokens=0, dense_blocks=(), sink_conditioning="exact_kv_and_rows",
                               extra_tokens=0, verbose=True)


def make_attention(patch, block_index=0, gate=None):
    block = type("B", (), {"attn": type("A", (), {"heads": 8, "head_dim": 128,
                                                  "to_gate_compress": gate,
                                                  "q_norm": type("N", (), {"eps": 1e-6, "weight": None})()})()})()
    real, nsa.h3_eligible = nsa.h3_eligible, lambda *a, **k: True
    try:
        seen = {}
        nsa.make_h3_block_patch(block, block_index, patch)(
            {"img": None, "rope_freqs": None, "transformer_options": {}},
            {"original_block": lambda a: seen.update(a) or {"img": a["img"]}})
    finally:
        nsa.h3_eligible = real
    return seen["attention"], block


def main():
    layout = PackedLayout(text_len=226, latent_t=8, latent_h=32, latent_w=48, audio_t=0)
    print(f"layout: seq_len={layout.seq_len}, segments={[(k, b - a) for a, b, k in layout.segments]}")

    # --- 1. the cube plan is a lossless permutation of the live rows ----------
    patch = make_patch()
    plan = patch.vsa_plan(layout, torch.device("cpu"))
    src, inv, n = plan["src"], plan["inv"], plan["n"]
    live = src >= 0
    assert n >= layout.seq_len, (n, layout.seq_len)
    assert int(live.sum()) == layout.seq_len, "plan lost or duplicated live rows"
    assert torch.equal(src[live].sort().values, torch.arange(layout.seq_len)), "not a permutation"
    assert torch.equal(src[inv], torch.arange(layout.seq_len)), "src[inv] is not the identity"
    print(f"  plan: {layout.seq_len} rows -> {n} padded ({n - layout.seq_len} pad, "
          f"{plan['n_prefix']} prefix tiles), src[inv] == identity")

    # a value round trip: permute a tagged tensor and bring it back the way rank_attention does
    x = torch.arange(layout.seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    padded = x[src.clamp_min(0)] * (src >= 0).unsqueeze(1).to(x.dtype)
    assert torch.equal(padded[inv], x), "out[plan['inv']] did not restore the model's token order"
    assert float(padded[~live].abs().max()) == 0.0, "pad rows were not zeroed"
    print("  round trip: x[src][inv] == x, pad rows zero")

    # --- 2. block_len counts live rows per tile ------------------------------
    bl = plan["block_len"]
    assert int(bl.sum()) == layout.seq_len, (int(bl.sum()), layout.seq_len)
    assert bl.dtype == torch.int32 and bl.numel() == n // 64, (bl.dtype, bl.numel(), n)
    print(f"  block_len: {bl.numel()} tiles, sum == seq_len, min {int(bl.min())} max {int(bl.max())}")

    # --- 3. vsa_context: built once, cached, and independent of upstream's caches
    attention, _ = make_attention(patch)
    rope = torch.randn(1, layout.seq_len, 1, 64, 2)
    to = {"minimax_h3_layout": layout}
    ms_sparse._PLANS.clear()
    ctx = ms_sparse.vsa_context(attention, rope, to, torch.device("cpu"), allowed=True)
    assert ctx["plan"] is not None and ctx["rope"].shape[1] == n, ctx["rope"].shape
    assert torch.equal(ctx["rope"][0, inv], rope[0]), "padded rope does not carry the live rows"
    assert float(ctx["rope"][0][~live].abs().max()) == 0.0, "padded rope pad rows not zero"
    assert len(ms_sparse._PLANS) == 1
    ctx2 = ms_sparse.vsa_context(attention, rope, to, torch.device("cpu"), allowed=True)
    assert ctx2["plan"] is ctx["plan"], "plan was rebuilt instead of served from our cache"
    assert patch.vsa_rope is None, "upstream's single-slot rope cache was used; it races across ranks"
    print("  vsa_context: plan cached by us, padded rope correct, upstream vsa_rope untouched")

    # --- 4. the rollback switch ---------------------------------------------
    try:
        ms_sparse.vsa_context(attention, rope, to, torch.device("cpu"), allowed=False)
        raise AssertionError("vsa ran with the switch off")
    except ms_sparse.SparseUnsupported as e:
        assert "sparse_vsa" in str(e), e
    print("  allowed=False refuses with a message naming the switch")

    # a non-vsa patch yields no context at all, and never builds a plan
    plain, _ = make_attention(make_patch(vsa=False))
    assert ms_sparse.vsa_context(plain, rope, to, torch.device("cpu"), allowed=False) is None
    print("  sol-attn/sla patches get no vsa context, switch irrelevant")

    # --- 5. concurrent ranks share one plan and do not race ------------------
    ms_sparse._PLANS.clear()
    got, errs = [], []
    def rank():
        try:
            got.append(ms_sparse.vsa_context(attention, rope, to, torch.device("cpu"), allowed=True)["plan"])
        except Exception as e:                                    # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=rank) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs, errs
    assert len({id(p) for p in got}) == 1, "8 ranks built 8 different plans"
    print("  8 concurrent ranks: one shared plan, no errors")

    # --- 6. gate row slicing is the contiguous head range --------------------
    import multistream.split as ms_split
    hd = 128
    idx = ms_split._gate_rows(3, 8, hd, torch.device("cpu"))
    assert torch.equal(idx, torch.arange(3 * hd, 8 * hd)), "gate rows are not the head range"
    qkv = ms_split._head_rows(3, 8, 8, hd, torch.device("cpu"))
    assert qkv.numel() == 3 * idx.numel(), "qkv index should cover q, k and v"
    assert torch.equal(qkv[:idx.numel()], idx), "qkv's q block should match the gate rows"
    print(f"  _gate_rows(3,8): {idx.numel()} contiguous rows; _head_rows covers 3x that (q|k|v)")

    test_gate_must_be_shadowed()

    print("PASS: vsa plumbing")



def test_gate_must_be_shadowed():
    """Regression, 2026-09-16 SEGV: the VSA coarse branch cast to_gate_compress off the ORIGINAL
    block (captured from the sparse node's closure) instead of the rank's shadow. That reads weights
    under a cuMemCreate mapping owned by the other device and segfaults rather than raising."""
    import multistream.split as ms_split

    class Lin:
        def __init__(self, shadowed):
            if shadowed:
                self._multistream_rank = 0

    # the choke point refuses an unshadowed module loudly instead of segfaulting
    try:
        ms_split._rows_linear_fn(Lin(shadowed=False), torch.zeros(4, 4), torch.arange(2))
        raise AssertionError("_rows_linear_fn accepted an unshadowed module")
    except Exception as e:
        assert "shadow" in str(e), e
    print("  _rows_linear_fn refuses an unshadowed module (the SEGV's proximate cause)")

    # and rank_attention no longer reaches for the captured block's gate at all
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "multistream", "sparse.py"), encoding="utf-8").read()
    body = src[src.index("def rank_attention("):]
    assert "gate = attn.to_gate_compress" not in body, \
        "rank_attention reads the gate off the captured (unshadowed) block again"
    assert "gate = gate_group" in body, "the gate should be the caller's shadow-bound closure"
    assert "gate_group(xc)" in body, "gate_group should be called with only the chunk"
    print("  rank_attention takes the gate from the caller's shadow-bound closure only")

    # the caller binds it off blk (the shadow), like qkv_proj
    split_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "multistream", "split.py"), encoding="utf-8").read()
    assert "shadow_gate = getattr(blk.attn, \"to_gate_compress\", None)" in split_src, \
        "_rank_block must bind the gate off the shadow block blk"
    # and both projections are resolved once per block, not inside the producer's chunk loop
    assert "_gate_group_fn(shadow_gate, h_full, g0, g1, hd)" in split_src, \
        "the gate projection must be built by the factory, outside the chunk loop"
    assert "_qkv_group_fn(blk.attn.qkv_proj, h_full, g0, g1, heads, hd)" in split_src, \
        "the qkv projection must be built by the factory, outside the chunk loop"
    print("  _rank_block binds both projections off blk, resolved once per block")

if __name__ == "__main__":
    main()
