# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""capture() against ComfyUI's REAL make_h3_block_patch closure, and _capture_overrides' sparse gate.

Run from the ComfyUI root with its venv. Touches no GPU: it builds the patch closure and inspects it.
"""
import sys, types, os
import torch

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, PACK)

import comfy_extras.nodes_sparse_attention as nsa
from multistream import sparse as ms_sparse

# --- 1. capture() reads the real closure -------------------------------------
patch = nsa.SparseAttnPatch(tau=1.3, topk_ratio=0.0, vsa=False, sigma_start=1.0, sigma_end=0.0,
                            min_tokens=12288, dense_blocks=(), sink_conditioning="exact_kv_and_rows",
                            extra_tokens=256, verbose=False)
block = types.SimpleNamespace(attn=types.SimpleNamespace(heads=56, head_dim=128))
bp = nsa.make_h3_block_patch(block, 7, patch)

captured = {}
def original_block(a):
    captured.update(a)
    return {"img": a["img"]}

img = types.SimpleNamespace(shape=(41068, 7168), dtype=None, device=types.SimpleNamespace(type="cpu"))
args = {"img": img, "t_emb": None, "mod_segments": None, "rope_freqs": None,
        "layout": None, "transformer_options": {"sigmas": [0.5]}}
out = bp(args, {"original_block": original_block})
assert out["img"] is img
# not eligible here (rope_freqs None / cpu), so no attention was installed -- that path must stay clean
assert captured.get("attention") is None, "expected dense for a cpu tensor"

# force the eligible path: install the attention callable the way block_patch would
attention = None
def fake_eligible(*a, **k):
    return True
real, nsa.h3_eligible = nsa.h3_eligible, fake_eligible
try:
    captured.clear()
    bp(args, {"original_block": original_block})
    attention = captured.get("attention")
finally:
    nsa.h3_eligible = real
assert attention is not None, "block_patch did not install an attention callable"

b, idx, p = ms_sparse.capture(attention)
assert b is block and idx == 7 and p is patch, (b, idx, p)
print("capture(): read block/block_index/patch out of the real closure")

# --- 2. a renamed closure must fail loudly, not silently disable sparse -------
def bogus(h, rope_freqs=None, transformer_options={}):
    return renamed_thing                                  # noqa: F821
try:
    ms_sparse.capture(lambda h, rope_freqs=None, transformer_options={}: None)
    raise AssertionError("capture() accepted a closure it cannot read")
except ms_sparse.SparseUnsupported as e:
    assert "nodes_sparse_attention.py has changed shape" in str(e)
print("capture(): refuses an unreadable closure with a clear message")

# --- 3. vsa is refused, with a reason ----------------------------------------
vpatch = nsa.SparseAttnPatch(tau=1.3, topk_ratio=0.1, vsa=True, sigma_start=1.0, sigma_end=0.0,
                             min_tokens=0, dense_blocks=(), sink_conditioning="off",
                             extra_tokens=0, verbose=False)
vbp = nsa.make_h3_block_patch(block, 0, vpatch)
nsa.h3_eligible = fake_eligible
try:
    captured.clear(); vbp(args, {"original_block": original_block})
    vattn = captured["attention"]
finally:
    nsa.h3_eligible = real
assert ms_sparse.wants_vsa(vattn) is True and ms_sparse.wants_vsa(attention) is False
try:
    ms_sparse.vsa_context(vattn, torch.zeros(1, 4, 1, 2), {}, torch.device("cpu"), allowed=False)
    raise AssertionError("vsa ran with the switch off")
except ms_sparse.SparseUnsupported as e:
    assert "sparse_vsa" in str(e), e
print("wants_vsa()/vsa_context(): vsa is gated by the sparse_vsa switch")
print("all sparse-capture checks passed")

# --- 4. _capture_overrides: refuses when the switch is off, captures when on ---
import ast, threading
src = open(os.path.join(PACK, "multistream", "split.py")).read()
tree = ast.parse(src)
nodes = [n for n in tree.body if getattr(n, "name", None) == "_capture_overrides"]
assigns = [n for n in tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", "") in ("_MISSING", "_NONE", "OVERRIDE") for t in n.targets)]
g = {"ms_sparse": ms_sparse, "MultiStreamError": type("MultiStreamError", (RuntimeError,), {})}
exec(compile(ast.Module(body=assigns + nodes, type_ignores=[]), "split.py", "exec"), g)
cap = g["_capture_overrides"]

nsa.h3_eligible = fake_eligible
try:
    user = {("double_block", 0): bp, ("double_block", 1): bp}
    # switch OFF -> the old error, plus the hint pointing at the new switch
    try:
        cap(user, 2, args, sparse=False)
        raise AssertionError("expected a refusal with the switch off")
    except g["MultiStreamError"] as e:
        assert "replaces the block computation" in str(e)
        assert "turn on `sparse_attention`" in str(e), f"no hint toward the switch: {e}"
    # a vsa patch is refused even with sparse on, until sparse_vsa is on too (the rollback)
    vuser = {("double_block", 0): vbp}
    try:
        cap(vuser, 1, args, sparse=True, vsa=False)
        raise AssertionError("vsa captured with sparse_vsa off")
    except g["MultiStreamError"] as e:
        assert "`sparse_vsa`" in str(e), e
    _, vat = cap(vuser, 1, args, sparse=True, vsa=True)
    assert vat[0] is not None and ms_sparse.wants_vsa(vat[0])
    # switch ON -> captured, one attention callable per block
    overrides, sparse_at = cap(user, 2, args, sparse=True)
    assert len(overrides) == 2 and len(sparse_at) == 2
    assert all(a is not None for a in sparse_at), sparse_at
    assert all(ms_sparse.capture(a)[2] is patch for a in sparse_at)
    # blocks the sparse node left dense must come back as None even with the switch on
    nsa.h3_eligible = lambda *a, **k: False
    overrides, sparse_at = cap(user, 2, args, sparse=True)
    assert sparse_at == [None, None], sparse_at
    # a block with no patch at all
    overrides, sparse_at = cap({}, 3, args, sparse=True)
    assert sparse_at == [None, None, None] and overrides == [g["_MISSING"]] * 3
finally:
    nsa.h3_eligible = real
print("_capture_overrides(): refuses when off, captures when on, None when the node says dense")
print("ALL CHECKS PASSED")
