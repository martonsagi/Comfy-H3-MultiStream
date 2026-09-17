# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Standalone parity + timing: H3 full forward on one GPU vs H3 MultiStream.

    python tests/parity.py --ckpt DIT.safetensors [--lora LORA.safetensors] [--frames N]

Runs outside the ComfyUI service (legacy ModelPatcher, lowvram partial load on cuda:0).
Needs both GPUs idle.
"""
import argparse
import importlib
import json
import os
import sys
import time

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.modules import attention as A  # noqa: E402

ms_split = importlib.import_module("ComfyUI-H3-MultiStream.multistream.split")

def model_path(name, folder):
    """An absolute path, or a file name under ComfyUI/models/<folder>."""
    return os.path.join(COMFY, "models", folder, os.path.expanduser(name))


def kitchen_override():
    fn = A.get_attention_function("comfy_kitchen_int8", None)

    def ov(_, *args, **kwargs):
        return fn(*args, **kwargs)
    ov.container_function = fn.container_function
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="H3 DiT checkpoint: absolute path or file name under models/diffusion_models")
    ap.add_argument("--lora", help="optional LoRA: absolute path or file name under models/loras")
    ap.add_argument("--frames", type=int, default=37)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--text", type=int, default=300)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--skip-single", action="store_true")
    a = ap.parse_args()

    patcher = comfy.sd.load_diffusion_model(model_path(a.ckpt, "diffusion_models"))
    if a.lora:
        lora_sd = comfy.utils.load_torch_file(model_path(a.lora, "loras"))
        patcher, _ = comfy.sd.load_lora_for_models(patcher, None, lora_sd, 1.0, 0.0)
    dm = patcher.model.diffusion_model
    mm.load_models_gpu([patcher])
    print("loaded; cuda:0 alloc GiB", torch.cuda.memory_allocated(0) / 2**30, flush=True)

    latent_t = 2 if a.frames <= 5 else ((a.frames - 5) // 17) * 5 + 2
    audio_t = round(a.frames / 24 * 40)
    g = torch.Generator().manual_seed(7)
    dev = mm.get_torch_device()
    video = torch.randn(1, 24, latent_t, a.height // 16, a.width // 16, generator=g).to(dev)
    audio = torch.randn(1, 32, 2, audio_t, generator=g).to(dev)
    context = (torch.randn(1, a.text, 5120, generator=g) * 0.5).to(dev, torch.bfloat16)
    timestep = torch.tensor([700.0], device=dev)
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0], device=dev)
    S = a.text + latent_t * (a.height // 32) * (a.width // 32) + 2 * audio_t
    print(f"latent_t={latent_t} audio_t={audio_t} S~{S}", flush=True)

    def to_base():
        return {"optimized_attention_override": kitchen_override(), "sample_sigmas": sigmas}

    def run(to):
        with torch.inference_mode():
            torch.cuda.synchronize(0); torch.cuda.synchronize(1)
            t = time.perf_counter()
            out = patcher.model.diffusion_model.forward([video, audio], timestep, context, transformer_options=to,
                                                        minimax_payload={})
            torch.cuda.synchronize(0); torch.cuda.synchronize(1)
            return out, time.perf_counter() - t

    res = {"S": S, "lora": a.lora, "frames": a.frames, "hw": [a.height, a.width]}
    if not a.skip_single:
        ref, _ = run(to_base())
        ts = [run(to_base())[1] for _ in range(a.trials)]
        res["single_s"] = min(ts)
        ref = [r.float().cpu() for r in ref]
    ds_to = ms_split.add_to_transformer_options(to_base())
    got, _ = run(dict(ds_to))
    ts = [run(dict(ds_to))[1] for _ in range(a.trials)]
    res["dual_s"] = min(ts)
    got = [r.float().cpu() for r in got]
    if not a.skip_single:
        for name, r, g_ in zip(("video", "audio"), ref, got):
            d = (r - g_).abs()
            res[f"{name}_max_abs"] = d.max().item()
            res[f"{name}_rel"] = (d.max() / r.abs().max()).item()
        res["ratio"] = res["dual_s"] / res["single_s"]
    res["dual_video_abs_max"] = got[0].abs().max().item()
    res["dual_video_mean"] = got[0].mean().item()
    if not a.skip_single:
        res["single_video_abs_max"] = ref[0].abs().max().item()
        res["single_video_mean"] = ref[0].mean().item()
    res["peak_GiB"] = [torch.cuda.max_memory_allocated(i) / 2**30 for i in (0, 1)]
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
