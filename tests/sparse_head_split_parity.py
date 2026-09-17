# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Proof that splitting attention HEADS across ranks is exact, for every sparse method.

This is the load-bearing assumption behind running ComfyUI's Model Sparse Attention node inside the
Ulysses split: a rank holds all tokens for a subset of heads, so it may only run the kernel over its
own heads if nothing in the selection mixes heads. comfy_kitchen ships a pure-PyTorch reference
(backends/eager/sol_attn.py) that runs on CPU, so the claim is checkable without a GPU.

    python tests/sparse_head_split_parity.py

No ComfyUI server, no CUDA. The eager backend is the REFERENCE, not what executes on a GPU -- this
bounds the risk, it does not eliminate it.
"""
import os
import sys

import torch

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, COMFY)

from comfy_kitchen.backends.eager.sol_attn import sol_attn   # noqa: E402

BLOCK = 64
TOL = 1e-7          # fp32 reduction-order noise; a real selection difference is orders larger


def run(label, groups, gated=False, T=512, H=8, D=128, **kw):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, D, dtype=torch.float32) for _ in range(3))
    gate = torch.randn(1, T, H, D, dtype=torch.float32) * 0.1 if gated else None
    full = sol_attn(q, k, v, coarse_gate=gate, **kw)
    parts = [sol_attn(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b],
                      coarse_gate=None if gate is None else gate[:, :, a:b], **kw)
             for a, b in groups]
    split = torch.cat(parts, dim=2)
    err = (full - split).abs().max().item()
    assert split.shape == full.shape, (split.shape, full.shape)
    assert err <= TOL, f"{label}: head split changed the result by {err:.3e} (> {TOL:.0e})"
    print(f"  {label:46} groups={groups}  max|full-split| = {err:.3e}")
    return err


def main():
    # deliberately UNEVEN groups: 56 heads over 3 GPUs is 19/19/18, so evenness must not be assumed
    uneven = [(0, 3), (3, 8)]
    thirds = [(0, 3), (3, 6), (6, 8)]
    blen = torch.full((512 // BLOCK,), BLOCK, dtype=torch.int32)
    blen[-1] = 40                                    # a partly-padded final tile, as VSA produces

    print("sol-attn / sla:")
    run("sol-attn (tau)", uneven, tau=1.3)
    run("sol-attn, 3 ranks", thirds, tau=1.3)
    run("sla (topk_ratio)", uneven, tau=1.0, topk_ratio=0.10)
    run("sol-attn + H3 sinks", uneven, tau=1.3, sink_blocks=[0, 2], sink_q=[0, 2])
    run("token_aug (extra_tokens)", uneven, tau=1.3, token_aug=256)

    print("vsa:")
    run("vsa shape: tail=False + block_len", uneven, tau=1.0, topk_ratio=0.10,
        tail=False, block_len=blen, sink_blocks=[0, 2], sink_q=[0, 2])
    run("vsa + coarse_gate", uneven, gated=True, tau=1.0, topk_ratio=0.10,
        tail=False, block_len=blen, sink_blocks=[0, 2], sink_q=[0, 2])
    run("vsa + coarse_gate, 3 ranks", thirds, gated=True, tau=1.0, topk_ratio=0.10,
        tail=False, block_len=blen, sink_blocks=[0, 2], sink_q=[0, 2])
    run("vsa, one head per rank", [(i, i + 1) for i in range(8)], gated=True, tau=1.0,
        topk_ratio=0.10, tail=False, block_len=blen, sink_blocks=[0, 2], sink_q=[0, 2])

    # a guard on the guard: if the harness compared identical things it would pass vacuously
    print("negative control:")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 512, 8, 128) for _ in range(3))
    a = sol_attn(q, k, v, tau=1.3)
    b = sol_attn(q, k, v, tau=0.1)
    assert (a - b).abs().max().item() > TOL, "tau has no effect -- the harness is not exercising sparsity"
    print(f"  tau 1.3 vs 0.1 differs by {(a - b).abs().max().item():.3e}: the harness does see sparsity")
    print("PASS: head splitting is exact for every method")


if __name__ == "__main__":
    main()
