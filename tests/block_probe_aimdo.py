# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Probe: under DynamicVRAM, does a rank shadow of a DiT block compute the same as the original block?

    python tests/block_probe_aimdo.py --ckpt DIT.safetensors [--block 10] [--tokens 3154]

Same GPU, same inputs, no split: isolates the shadow-module cast from the Ulysses split.
Compares each submodule's output, then the whole block.
"""
import argparse
import importlib
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="H3 DiT checkpoint: absolute path or file name under models/diffusion_models")
ap.add_argument("--block", type=int, default=10)
ap.add_argument("--tokens", type=int, default=3154)
ap.add_argument("--no-aimdo", action="store_true")
cli = ap.parse_args()
sys.argv = sys.argv[:1]

from comfy.cli_args import args  # noqa: E402
import comfy_aimdo.control  # noqa: E402

if not cli.no_aimdo:
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not args.disable_nvml_pressure)
import torch  # noqa: E402

import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402

if not cli.no_aimdo:
    assert comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices())
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    comfy_aimdo.control.set_log_info()

import comfy.sd  # noqa: E402
from comfy.ldm.minimax import model as H  # noqa: E402
from comfy.ldm.modules import attention as A  # noqa: E402

ms_cast = importlib.import_module("ComfyUI-H3-MultiStream.multistream.cast")
CKPT = os.path.join(COMFY, "models", "diffusion_models", os.path.expanduser(cli.ckpt))


def main():
    patcher = comfy.sd.load_diffusion_model(CKPT)
    mm.load_models_gpu([patcher])
    dev = patcher.load_device
    dm = patcher.model.diffusion_model
    blk = dm.blocks[cli.block]
    ms_cast.install_cast_hook()
    shadow = ms_cast.make_shadow(blk, 0)
    print("patcher", type(patcher).__name__, "aimdo", comfy.memory_management.aimdo_enabled, "dev", dev, flush=True)

    fn = A.get_attention_function("comfy_kitchen_int8", None)

    def ov(_, *a, **k):
        return fn(*a, **k)
    ov.container_function = fn.container_function
    to = {"optimized_attention_override": ov}

    g = torch.Generator().manual_seed(3)
    S = cli.tokens
    x = (torch.randn(S, 5376, generator=g) * 0.5).to(dev, torch.bfloat16)
    t_emb = torch.randn(3, blk.adaln_proj.linear.in_features, generator=g).to(dev)
    rope = H.rope_rotation_table(torch.rand(S, 96, generator=g) * 100.0, torch.bfloat16).to(dev)
    third = S // 3
    segs = [(0, third, 1), (third, 2 * third, 5), (2 * third, S, 0)]

    def cmp(name, a, b):
        if isinstance(a, (tuple, list)):
            for i, (ai, bi) in enumerate(zip(a, b)):
                cmp(f"{name}[{i}]", ai, bi)
            return
        d = (a.float() - b.float()).abs().max().item()
        print(f"  {name:28s} max|diff|={d:.6g}  dtype {a.dtype}/{b.dtype}  |a|max={a.float().abs().max().item():.5g}",
              flush=True)

    with mm.cuda_device_context(dev), torch.inference_mode():
        cmp("adaln_proj(t_emb)", blk.adaln_proj(t_emb), shadow.adaln_proj(t_emb))
        cmp("norm1(x)", blk.norm1(x), shadow.norm1(x))
        h = blk.norm1(x)
        cmp("attn.qkv_proj(h)", blk.attn.qkv_proj(h), shadow.attn.qkv_proj(h))
        cmp("attn(h) full", blk.attn(h, rope_freqs=rope, transformer_options=to),
            shadow.attn(h, rope_freqs=rope, transformer_options=to))
        a_in = torch.randn(S, 7168, generator=g).to(dev, torch.bfloat16)
        cmp("attn.out_proj", blk.attn.out_proj(a_in), shadow.attn.out_proj(a_in))
        cmp("mlp(h)", blk.mlp(h), shadow.mlp(h))
        cmp("norm2(x)", blk.norm2(x), shadow.norm2(x))
        cmp("BLOCK", blk(x.clone(), t_emb, segs, rope, transformer_options=to),
            shadow(x.clone(), t_emb, segs, rope, transformer_options=to))
        w = blk.attn.qkv_proj
        print("  qkv_proj attrs: layout_type", getattr(w, "layout_type", None), "quant_format", getattr(w, "quant_format", None),
              "weight_function", len(w.weight_function), "lowvram_fn", getattr(w, "weight_lowvram_function", None),
              "_v", hasattr(w, "_v"), "comfy_force_cast_weights", getattr(w, "comfy_force_cast_weights", None), flush=True)
        a_ = blk.adaln_proj.linear
        print("  adaln attrs: weight dtype", a_.weight.dtype, "_comfy_model_dtype",
              getattr(a_, "weight_comfy_model_dtype", None), "bias dtype", a_.bias.dtype,
              "bias_comfy_model_dtype", getattr(a_, "bias_comfy_model_dtype", None), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
