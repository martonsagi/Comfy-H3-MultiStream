# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""_rows_linear_fn resolves the weight ONCE per block, not once per producer chunk.

    python tests/rows_linear_hoist.py

CPU only. The chunk loop calls each projection ~12x per block per rank per step; resolving inside it
cost ~15 s/step for the VSA gate with the DiT weight cache off (32.4 s vs 16.9 s warm, 2026-09-16).
This counts the resolutions so the hoist cannot regress unnoticed.
"""
import ast
import os
import sys
import types

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PACK)

CASTS = []
WARNINGS = []


def build():
    """Extract _rows_linear_fn and the row-index helpers with stubs for comfy/quant."""
    src = open(os.path.join(PACK, "multistream", "split.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    want = {"_rows_linear_fn", "_qkv_group_fn", "_gate_group_fn", "_qkv_group", "_head_rows", "_gate_rows",
            "_int8_kernel_available"}
    body = [n for n in tree.body if getattr(n, "name", None) in want]
    body += [n for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") in ("_HEAD_ROWS", "_INT8_FALLBACK_WARNED") for t in n.targets)]

    def cast_bias_weight(lin, like, **kw):
        # 3 * heads * hd rows: qkv stacks q, k and v, so _head_rows indexes past heads*hd
        CASTS.append(lin)
        return torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4), None, None

    g = {
        "torch": torch,
        "comfy": types.SimpleNamespace(ops=types.SimpleNamespace(cast_bias_weight=cast_bias_weight)),
        "QuantizedTensor": type("QT", (), {}),
        "_dtype_code": lambda d: 0,
        "log": types.SimpleNamespace(warning=lambda *a, **k: WARNINGS.append(a)),
        "MultiStreamError": type("MultiStreamError", (RuntimeError,), {}),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "split.py", "exec"), g)
    return g


def main():
    g = build()

    class Lin:
        _multistream_rank = 0
        weight_function = ()

    like = torch.zeros(64, 4, dtype=torch.float32)

    # --- the factory resolves once, however many chunks follow --------------
    CASTS.clear()
    fn = g["_gate_group_fn"](Lin(), like, 0, 2, 2)
    assert len(CASTS) == 1, f"factory resolved {len(CASTS)} times, expected 1"
    outs = [fn(like[i:i + 8]) for i in range(0, 64, 8)]      # 8 chunks, as the producer would
    assert len(CASTS) == 1, f"{len(CASTS)} resolutions across 8 chunks -- the hoist regressed"
    assert all(o.shape == (8, 4) for o in outs), [o.shape for o in outs]  # 2 heads x hd 2 = 4 rows
    print(f"  _gate_group_fn: 1 resolution, 8 chunks applied  (was 1 per chunk)")

    CASTS.clear()
    qfn = g["_qkv_group_fn"](Lin(), like, 0, 1, 2, 2)
    for i in range(0, 64, 8):
        qfn(like[i:i + 8])
    assert len(CASTS) == 1, f"{len(CASTS)} resolutions for qkv across 8 chunks"
    print("  _qkv_group_fn: 1 resolution, 8 chunks applied")

    # --- the one-shot dense form still resolves exactly once ----------------
    CASTS.clear()
    out = g["_qkv_group"](Lin(), like, 0, 1, 2, 2)
    # 1 head of the group x hd 2, stacked q|k|v = 6 output features
    assert len(CASTS) == 1 and out.shape == (64, 6), (len(CASTS), out.shape)
    print("  _qkv_group (dense one-shot): 1 resolution")

    # --- results must not depend on chunking -------------------------------
    fn = g["_gate_group_fn"](Lin(), like, 0, 2, 2)
    whole = fn(like)
    chunked = torch.cat([fn(like[i:i + 8]) for i in range(0, 64, 8)])
    assert torch.equal(whole, chunked), "chunked application differs from whole"
    print("  chunked application == whole-tensor application")

    # --- the shadow guard survives the refactor ----------------------------
    class Bare:
        weight_function = ()
    try:
        g["_rows_linear_fn"](Bare(), like, torch.arange(2))
        raise AssertionError("accepted an unshadowed module")
    except g["MultiStreamError"] as e:
        assert "shadow" in str(e), e
    print("  unshadowed module still refused (SEGV guard intact)")

    # --- comfy-kitchen without the private dtype helper: dequantize fallback --
    QT = g["QuantizedTensor"]

    class Int8Weight(QT):
        _layout_cls = "TensorWiseINT8Layout"
        _params = types.SimpleNamespace(transposed=False)

        def dequantize(self):
            return torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)

    g["comfy"].ops.cast_bias_weight = lambda lin, like, **kw: (Int8Weight(), None, None)
    g["_dtype_code"] = None
    WARNINGS.clear()
    fb = g["_rows_linear_fn"](Lin(), like, torch.arange(4))
    want = torch.nn.functional.linear(like, torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)[:4])
    assert torch.equal(fb(like), want), "dequantize fallback gave a different result"
    g["_rows_linear_fn"](Lin(), like, torch.arange(4))
    assert len(WARNINGS) == 1, f"fallback warned {len(WARNINGS)} times, expected once"
    print("  missing comfy-kitchen dtype helper: dequantize fallback, warned once")

    print("PASS: rows-linear hoist")


if __name__ == "__main__":
    main()
