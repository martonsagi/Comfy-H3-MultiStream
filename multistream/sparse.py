# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Per-rank block-sparse attention: ComfyUI's Model Sparse Attention node inside the Ulysses split.

WHY THIS COMPOSES AT ALL. comfy_extras/nodes_sparse_attention.py replaces a block's whole attention
stage -- qkv projection, rope and attention are fused into ck.sol_attn_chunked, "full Q/K/V never
built" -- so it cannot be handed to MultiStream as an `optimized_attention_override`; that is why
split.py's _capture_overrides refuses it by default. But its block selection is per attention head
and per query block (sol-attn's tau is a threshold on each head's own score distribution), and a
Ulysses rank holds ALL tokens for a SUBSET of heads. Running the kernel over a rank's head group is
therefore the same computation the single-GPU path would do for those heads, not an approximation.
The kernel cooperates: sol_attn_chunked takes the head count as a parameter and returns the attention
output BEFORE out_proj, which is exactly what the rank needs to feed its all-to-all.

VSA. Supported since 2026-09-16, behind its own switch (`sparse_vsa` on the node) so it can be rolled
back on its own. It needs three things the plain path does not: the block's to_gate_compress
row-sliced to the rank's heads (split._gate_group), the cube plan and the padded rope built ONCE per
rank per step rather than per block (see vsa_context -- upstream caches the plan in a 4-entry FIFO
keyed partly on the device, and the padded rope in a SINGLE slot, so across rank threads they thrash
and race), and the padded/permuted token order. That order stays entirely inside this call:
out[plan["inv"]] restores the model's token order before returning, so the caller's token bounds and
all-to-all never interact with VSA's tiling and need no changes.

HEAD SPLITTING IS EXACT, MEASURED. comfy_kitchen's eager reference (backends/eager/sol_attn.py) runs
on CPU; full-vs-split over uneven head groups is bit-identical for sol-attn, sinks, token_aug and the
whole VSA branch including coarse_gate, and ~1 ULP for sla. See tests/sparse_head_split_parity.py.
The eager backend is the reference, not what executes -- the CUDA kernel is still unverified here.

THREAD SAFETY. SparseAttnPatch keeps `pooled`, `vsa_plans` and `_logged` as plain dicts written once
per block per step. The split runs N rank threads through the same patch object at the same time, so
every access here is under _PATCH_LOCK. The pooled statistics are also RE-KEYED per head group: they
are (heads, head_dim) tensors carried between steps, so two ranks sharing one key would race and each
would publish statistics covering only its own heads.
"""
import threading

import torch

_PATCH_LOCK = threading.Lock()
_FREEVARS = ("block", "block_index", "patch")   # make_h3_block_patch's attention() closure
_PLANS = {}          # (device, signature, segments) -> plan; ours, because upstream's holds only 4
_PLANS_LOCK = threading.Lock()


class SparseUnsupported(RuntimeError):
    pass


def _sparse_mod():
    import comfy_extras.nodes_sparse_attention as m
    return m


def capture(attention):
    """(block, block_index, patch) out of the attention closure the Model Sparse Attention node builds.

    It is a closure, not an object, so there is no public way in: make_h3_block_patch() binds block,
    block_index and patch and hands out attention(h, rope_freqs, transformer_options). The freevar
    names are asserted so an upstream rename fails here, loudly, instead of silently disabling sparse.
    """
    names = getattr(attention, "__code__", None) and attention.__code__.co_freevars
    cells = getattr(attention, "__closure__", None)
    if not names or not cells or tuple(sorted(names)) != _FREEVARS:
        raise SparseUnsupported(
            f"cannot read the sparse-attention patch: expected closure over {_FREEVARS}, got {names}. "
            "ComfyUI's nodes_sparse_attention.py has changed shape; H3 MultiStream needs updating.")
    got = {n: c.cell_contents for n, c in zip(names, cells)}
    return got["block"], got["block_index"], got["patch"]


def wants_vsa(attention):
    """True when the sparse node is configured for VSA (needs the `sparse_vsa` switch)."""
    return bool(capture(attention)[2].vsa)


def vsa_context(attention, rope, to, device, allowed):
    """Cube plan + padded rope for one rank, built ONCE per step. None when this is not a VSA patch.

    Everything here depends on (layout, device) only, never on the block, so calling it per block --
    which is what happens if you just let h3_sparse_attention do it -- is wasted work AND unsafe:
    upstream keeps the padded rope in a single slot (self.vsa_rope) that every rank overwrites, and
    the plan in a 4-entry FIFO keyed partly on str(device) that thrashes past 4 ranks. We build the
    rope ourselves and keep our own plan cache.
    """
    patch = capture(attention)[2]
    if not patch.vsa:
        return None
    if not allowed:
        raise SparseUnsupported(
            "the Model Sparse Attention node is set to 'vsa'. Turn on `sparse_vsa` on the H3 "
            "MultiStream node to run it split, or pick 'sol-attn'/'sla' on the sparse node.")
    layout = to.get("minimax_h3_layout")
    if layout is None:
        raise SparseUnsupported(
            "'vsa' needs the MiniMax-H3 packed layout in transformer_options and it is missing; "
            "this is not an H3 model, or H3's forward has changed.")
    key = (str(device), tuple(layout.signature), tuple(layout.segments))
    with _PLANS_LOCK:
        plan = _PLANS.get(key)
    if plan is None:
        with _PATCH_LOCK:                     # patch.vsa_plans is a plain dict shared by every rank
            plan = patch.vsa_plan(layout, device)
        with _PLANS_LOCK:
            _PLANS[key] = plan
    # upstream's vsa_rope_freqs() caches in one slot shared by all ranks; build ours and keep it
    padded = rope.new_zeros((1, plan["n"]) + tuple(rope.shape[2:]))
    padded[0, plan["inv"]] = rope[0]
    return {"plan": plan, "rope": padded}


def rank_attention(attention, h_full, rope, to, qkv_group, qw, kw, g0, g1, hd,
                   vsa=None, gate_group=None):
    """Attention output for head group [g0, g1) over all tokens, as (tokens, (g1-g0)*hd).

    `qkv_group(x)` must return the qkv projection of `x` restricted to this rank's head group --
    split._qkv_group, which already slices the effective (LoRA-applied, possibly int8) weight rows.
    `qw`/`kw` are the q/k RMS-norm weights already resolved onto this rank's device by the caller:
    they may live on another GPU, and split.py stages every such crossing through host RAM because
    aimdo's cuMemCreate mappings grant device access to their owner only.

    `gate_group(xc)` (VSA only) must likewise be bound by the caller to the RANK'S SHADOW block --
    split._gate_group over blk.attn.to_gate_compress. It must NOT come from the block this function
    captures out of the sparse node's closure: that is the original, unshadowed module, whose weights
    carry vbar state and live under a cuMemCreate mapping owned by the primary device. Casting it
    from a rank thread segfaulted (2026-09-16); the presence check below reads an attribute only.
    """
    m = _sparse_mod()
    block, block_index, patch = capture(attention)
    if patch.vsa and vsa is None:
        raise SparseUnsupported("'vsa' patch reached rank_attention without a vsa context; "
                                "vsa_context() must be called once per rank per step")
    attn = block.attn
    ph = g1 - g0
    n_tokens = h_full.shape[0]
    dev = h_full.device
    plan = vsa["plan"] if vsa is not None else None
    n = plan["n"] if plan is not None else n_tokens        # padded/permuted row count under VSA
    freqs = vsa["rope"] if vsa is not None else rope

    # pooled kmean/vscale carry between steps and are shaped (heads, head_dim): key them per head group
    # or two ranks race on one entry and each publishes statistics for only its own heads.
    key = (block_index, n, (g0, g1), tuple(to.get("uuids", ())))
    extra = {}
    gate = None
    with _PATCH_LOCK:
        pooled = patch.pooled.get(key)
        if plan is None:
            sink, sink_q = patch.sinks(to, n_tokens)
        else:
            # VSA: the zero-padded prefix tiles are the sinks, and every tile carries its own live
            # length, so the kernel must not add the pooled tail term (tail=False).
            sink = sink_q = (0, plan["n_prefix"])
            extra = {"tail": False, "block_len": plan["block_len"]}
            # attribute check only -- never touch this module's tensors from a rank thread
            if getattr(attn, "to_gate_compress", None) is not None and gate_group is None:
                raise SparseUnsupported(
                    "VSA: the model has to_gate_compress but the caller supplied no gate_group bound "
                    "to this rank's shadow block; refusing rather than casting the unshadowed module")
            gate = gate_group
    if gate is not None:
        # (1, n, ph, hd): the kernel requires coarse_gate to have q's shape. ~322 MB at 45k rows and
        # 28 heads in bf16 -- the split halves this versus running all 56 heads on one card.
        extra["coarse_gate"] = h_full.new_empty(n, ph * hd).view(1, n, ph, hd)
    elif plan is not None:
        with _PATCH_LOCK:   # log_once mutates patch._logged, a set shared by every rank thread
            patch.log_once(("ms-no-gate",), "VSA: no to_gate_compress on this model; fine stage only")
    first = pooled is None
    if first:
        pooled = (torch.empty((ph, hd), dtype=torch.float32, device=dev),
                  torch.empty((ph, hd), dtype=torch.float32, device=dev))

    def chunks():
        for i in range(0, n, m.PRODUCER_CHUNK):
            if plan is None:
                yield qkv_group(h_full[i:i + m.PRODUCER_CHUNK])
                continue
            idx = plan["src"][i:i + m.PRODUCER_CHUNK]
            xc = h_full[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(h_full.dtype)   # pad rows zero
            if gate is not None:
                extra["coarse_gate"].view(n, ph * hd)[i:i + xc.shape[0]] = gate_group(xc)
            yield qkv_group(xc)

    out, kmean, vscale = m.ck.sol_attn_chunked(
        chunks, n, ph, freqs, (qw, kw),
        kmean=None if first else pooled[0], vscale=None if first else pooled[1],
        tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=0 if plan is not None else patch.extra_tokens,
        sink_blocks=list(sink), sink_q=list(sink_q), rope_eps=attn.q_norm.eps, **extra)
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    with _PATCH_LOCK:
        patch.pooled[key] = pooled
        patch.log_once(("ms-producer", n, ph),
                       f"split sparse producer: {n_tokens} tokens"
                       + (f" -> {n} VSA rows ({plan['n_prefix']} prefix tiles, "
                          f"coarse {'on' if gate is not None else 'off'})" if plan is not None else "")
                       + f", heads {g0}-{g1}, sinks {sink}/{sink_q}")
    out = out.view(n, ph * hd)
    if plan is not None:
        out = out[plan["inv"]]      # back to the model's token order, BEFORE the caller's all-to-all
    return out                      # pre-out_proj: the rank's all-to-all and out_proj follow
