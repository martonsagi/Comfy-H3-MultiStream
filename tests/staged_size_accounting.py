# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""The tier boundary and the budget charge must measure a block the same way.

    python tests/staged_size_accounting.py

QuantizedTensor subclasses torch.Tensor without overriding numel()/element_size(), so the obvious
product reports the DEQUANTISED size -- ~2x the truth for H3 int8. cast.staged_size counted the real
qdata while vram_cache.staged_bytes counted the logical size, so offer() charged twice what the
boundary assumed: blocks inside the VRAM tier were refused while the host tier had already skipped
them, leaving them in neither (2026-09-16).
"""
import os
import sys

import torch

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, PACK)

from comfy.quant_ops import QuantizedTensor          # noqa: E402
from multistream import cast as ms_cast              # noqa: E402
from multistream import vram_cache as ms_vram        # noqa: E402


def main():
    assert issubclass(QuantizedTensor, torch.Tensor)
    assert "numel" not in QuantizedTensor.__dict__ and "element_size" not in QuantizedTensor.__dict__, \
        "QuantizedTensor now overrides numel/element_size -- revisit tensor_bytes, it may be redundant"
    print("  QuantizedTensor still inherits numel/element_size (the trap is live)")

    plain = torch.zeros(1024, 512, dtype=torch.bfloat16)
    assert ms_cast.tensor_bytes(plain) == 1024 * 512 * 2
    print(f"  a plain bf16 tensor counts {ms_cast.tensor_bytes(plain) / 2**20:.0f} MiB")

    # a REAL int8 QuantizedTensor, built the way the checkpoint's weights are
    from comfy.quant_ops import TensorWiseINT8Layout as L
    qdata = torch.zeros(4096, 4096, dtype=torch.int8)
    scale = torch.zeros(4096, 1, dtype=torch.float32)
    params = L.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=(4096, 4096),
                      is_weight=True, convrot=False, convrot_groupsize=256, transposed=False)
    q = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    real = ms_cast.tensor_bytes(q)
    expect = qdata.numel() * 1 + scale.numel() * 4
    assert real == expect, (real, expect)
    naive_q = q.numel() * q.element_size()
    assert naive_q > real * 1.5, f"the trap should be visible: naive {naive_q} vs real {real}"
    print(f"  int8 QuantizedTensor: real {real/2**20:.2f} MiB, naive product says "
          f"{naive_q/2**20:.2f} MiB ({naive_q/real:.1f}x)")

    # and both call sites agree
    staged = {("a", "weight"): q, ("a", "bias"): plain}
    assert ms_vram.staged_bytes(staged) == ms_cast.tensor_bytes(q) + ms_cast.tensor_bytes(plain)
    naive = sum(t.numel() * t.element_size() for t in staged.values())
    print(f"  staged_bytes agrees with tensor_bytes: {ms_vram.staged_bytes(staged)/2**20:.2f} MiB"
          f"  (the naive product would have said {naive/2**20:.2f} MiB)")
    assert ms_vram.staged_bytes(staged) <= naive

    print("PASS: staged size accounting")


if __name__ == "__main__":
    main()
