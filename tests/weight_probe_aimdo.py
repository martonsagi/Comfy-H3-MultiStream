# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Probe: under DynamicVRAM, does a plain host->GPU copy of a module weight equal ComfyUI's own vbar cast?

    python tests/weight_probe_aimdo.py --ckpt DIT.safetensors
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
os.chdir(COMFY)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="H3 DiT checkpoint: absolute path or file name under models/diffusion_models")
cli = ap.parse_args()
sys.argv = sys.argv[:1]

from comfy.cli_args import args  # noqa: E402
import comfy_aimdo.control  # noqa: E402

comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not args.disable_nvml_pressure)
import torch  # noqa: E402

import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402

assert comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices())
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
comfy_aimdo.control.set_log_info()

import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

CKPT = os.path.join(COMFY, "models", "diffusion_models", os.path.expanduser(cli.ckpt))


def deq(t):
    t = t.dequantize() if isinstance(t, QuantizedTensor) else t
    return t.float()


def describe(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    inner = t
    extra = ""
    if isinstance(t, QuantizedTensor):
        qd, sc = comfy.ops.TensorWiseINT8Layout.get_plain_tensors(t)
        inner = qd
        extra = f" qdata.dev={qd.device} scale.dev={sc.device}"
    st = inner.untyped_storage()
    print(f"  {name}: type={type(t).__name__} dev={t.device} dtype={t.dtype} meta={getattr(t, 'is_meta', None)}"
          f" file_slice={hasattr(st, '_comfy_tensor_file_slice')}{extra}")


patcher = comfy.sd.load_diffusion_model(CKPT)
dm = patcher.model.diffusion_model
blk = dm.blocks[10]
print("BEFORE load_models_gpu")
describe("qkv.weight", blk.attn.qkv_proj.weight)
describe("norm1.weight", blk.norm1.weight)
mm.load_models_gpu([patcher])
dev = patcher.load_device
print("AFTER load_models_gpu, load_device", dev)
for name, mod in (("qkv_proj", blk.attn.qkv_proj), ("norm1", blk.norm1), ("adaln", blk.adaln_proj.linear)):
    describe(f"{name}.weight", mod.weight)
    print(f"  {name}: has _v={hasattr(mod, '_v')} comfy_cast_weights={getattr(mod, 'comfy_cast_weights', None)}")
    x = torch.zeros(4, mod.weight.shape[-1] if mod.weight.dim() > 1 else mod.weight.shape[0], dtype=torch.bfloat16, device=dev)
    with mm.cuda_device_context(dev), torch.inference_mode():
        w_ref, b_ref, os_ = comfy.ops.cast_bias_weight(mod, x, offloadable=True, compute_dtype=torch.bfloat16, want_requant=True)
        ref = deq(w_ref).cpu()
        comfy.ops.uncast_bias_weight(mod, w_ref, b_ref, os_)
        plain = deq(mod.weight.to(dev)).cpu()
        plain_cpu = deq(mod.weight.to("cpu")) if mod.weight.device.type != "meta" else None
    d = (ref - plain).abs().max().item()
    print(f"  {name}: vbar-cast vs plain .to(dev): max|diff|={d:.6g}  ref.abs.max={ref.abs().max().item():.6g}"
          f" plain.abs.max={plain.abs().max().item():.6g}"
          + (f" plain_cpu.abs.max={plain_cpu.abs().max().item():.6g}" if plain_cpu is not None else ""))
