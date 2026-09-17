# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Parity + timing under DynamicVRAM (comfy-aimdo), initialised the way ComfyUI's main.py does.

    python tests/parity_aimdo.py --ckpt DIT.safetensors [--frames N --height H --width W] [--modes single,host,p2p]
    H3MS_UNSAFE=1 python tests/parity_aimdo.py --ckpt DIT.safetensors --modes unsafe   # pre-fix behaviour: expected to crash

DiT on gpu:1 via SelectModelDevice with the comfy kitchen int8 attention override. --block-patch applies it per block
through the MiniMaxH3BlockAttentionSplit node instead (a separate custom node, h3_block_attention, must be installed).
Outputs of each mode are compared against the single-GPU run in the same process.
"""
import argparse
import contextlib
import faulthandler
import importlib
import json
import logging
import os
import signal
import sys
import time

faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps every thread's stack
print("pid", os.getpid(), flush=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=37)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--text", type=int, default=300)
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--modes", default="single,host,p2p")
ap.add_argument("--gpu", default="gpu:1")
ap.add_argument("--ckpt", required=True, help="H3 DiT checkpoint: absolute path or file name under models/diffusion_models")
ap.add_argument("--lora", help="optional LoRA (LoraLoaderModelOnly order): absolute path or file name under models/loras")
ap.add_argument("--lora-strength", type=float, default=1.0)
ap.add_argument("--block-patch", action="store_true",
                help="per-block kitchen int8 patches via the MiniMaxH3BlockAttentionSplit custom node")
cli = ap.parse_args()
sys.argv = sys.argv[:1]  # keep ComfyUI's own arg parser away from ours

from comfy.cli_args import args  # noqa: E402
import comfy_aimdo.control  # noqa: E402

comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not args.disable_nvml_pressure)

import torch  # noqa: E402

import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402

ok = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices())
assert ok, "aimdo init_devices failed"
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
comfy_aimdo.control.set_log_info()  # main.py sets the native level from the console level (INFO in the service)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

import comfy.patcher_extension  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_multigpu import SelectModelDeviceNode  # noqa: E402

ms_split = importlib.import_module("ComfyUI-H3-MultiStream.multistream.split")
ms_gpus = importlib.import_module("ComfyUI-H3-MultiStream.multistream.gpus")

CKPT = os.path.join(COMFY, "models", "diffusion_models", os.path.expanduser(cli.ckpt))
LORA = os.path.join(COMFY, "models", "loras", os.path.expanduser(cli.lora)) if cli.lora else None


def main():
    patcher = comfy.sd.load_diffusion_model(CKPT)
    print("patcher", type(patcher).__name__, "aimdo", comfy.memory_management.aimdo_enabled, flush=True)
    if cli.lora:
        lora_sd = comfy.utils.load_torch_file(LORA)
        patcher, _ = comfy.sd.load_lora_for_models(patcher, None, lora_sd, cli.lora_strength, 0.0)
        print("lora", LORA, "strength", cli.lora_strength, "patches", len(patcher.patches), flush=True)
    out = SelectModelDeviceNode.execute(patcher, cli.gpu)
    patcher = out.result[0] if hasattr(out, "result") else out[0]
    if cli.block_patch:
        h3_block_attention = importlib.import_module("h3_block_attention")
        node = h3_block_attention.NODE_CLASS_MAPPINGS["MiniMaxH3BlockAttentionSplit"]()
        patcher = node.patch(patcher, "comfy kitchen attention", "comfy kitchen attention", 0, 0)[0]
    else:
        import comfy.ldm.modules.attention as attn_mod
        patcher = patcher.clone()
        patcher.set_model_optimized_attention(attn_mod.get_attention_function("comfy_kitchen_int8", None))
    print("block_patch", cli.block_patch, flush=True)
    dev = patcher.load_device
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    print("load_device", dev, flush=True)

    latent_t = 2 if cli.frames <= 5 else ((cli.frames - 5) // 17) * 5 + 2
    audio_t = round(cli.frames / 24 * 40)
    g = torch.Generator().manual_seed(7)
    video = torch.randn(1, 24, latent_t, cli.height // 16, cli.width // 16, generator=g).to(dev)
    audio = torch.randn(1, 32, 2, audio_t, generator=g).to(dev)
    context = (torch.randn(1, cli.text, 5120, generator=g) * 0.5).to(dev, torch.bfloat16)
    timestep = torch.tensor([700.0], device=dev)
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0], device=dev)
    S = cli.text + latent_t * (cli.height // 32) * (cli.width // 32) + 2 * audio_t
    res = {"S": S, "frames": cli.frames, "hw": [cli.height, cli.width], "load_device": str(dev)}
    print(f"latent_t={latent_t} audio_t={audio_t} S~{S}", flush=True)

    def base_to():
        to = comfy.patcher_extension.copy_nested_dicts(patcher.model_options.get("transformer_options", {}))
        to["sample_sigmas"] = sigmas
        return to

    def run(to, split=False):
        # the sampler wraps sampling in cuda_device_context(load_device) (comfy/samplers.py); aimdo's malloc
        # graph records on the current device's stream, so a model on cuda:1 must run with cuda:1 current.
        # The node's APPLY_MODEL wrapper disables the comfy compiler; this harness calls the diffusion model
        # directly (below apply_model), so it applies the same switch itself for split runs.
        compiler = ms_split.comfy_compiler_disabled() if split else contextlib.nullcontext()
        with mm.cuda_device_context(dev), compiler, torch.inference_mode():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            t = time.perf_counter()
            o = dm([video, audio], timestep, context, transformer_options=to, minimax_payload={})
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            return [x.float().cpu() for x in o], time.perf_counter() - t

    # modes: single (compiler on, as the sampler runs it), single_nc (compiler off), host, p2p, unsafe.
    # Every run's output is kept and compared pairwise, so nondeterminism shows up as single vs single.
    runs = []
    for idx, mode in enumerate(cli.modes.split(",")):
        # mode = base[@rank map[~shares]], e.g. host_cache@1+0+1+0 (4 logical ranks on 2 GPUs) or
        # host_cache@1+0+1~1+1+0.5. Rank 0 of a rank map must be the model's device (--gpu).
        base, _, spec = mode.partition("@")
        rank_map, _, share_spec = spec.partition("~")
        ranks = [int(i) for i in rank_map.split("+")] if rank_map else None
        shares = [float(v) for v in share_spec.split("+")] if share_spec else None
        parts = base.split("_")
        if base in ("single", "single_nc"):
            to_fn = base_to
        else:
            # host / p2p, plus any of: _cache (pinned weight cache), _nopf (no side-stream prefetch),
            # _1gpu (GPU set limited to the model's GPU: must run unsplit and match single)
            exch = "p2p" if base == "unsafe" else parts[0]
            key = (os.path.realpath(CKPT), "[]") if "cache" in parts else None
            pf = "nopf" not in parts
            gs = ms_gpus.GPUSet(max_gpus=1) if "1gpu" in parts else None
            to_fn = lambda e=exch, k=key, p=pf, s=gs, rd=ranks, rs=shares: ms_split.add_to_transformer_options(
                base_to(), exchange=e, cache_key=k, prefetch=p, gpu_set=s, rank_devices=rd, rank_shares=rs)
        # the node installs no wrappers for a 1-GPU plan, so the model runs with ComfyUI's compiler like "single"
        wrap_compiler = base != "single" and "1gpu" not in parts
        label = f"{idx}:{mode}"
        try:
            outs, first = run(to_fn(), wrap_compiler)
            times = [first] + [run(to_fn(), wrap_compiler)[1] for _ in range(cli.trials)]
        except Exception as e:
            # keep going: one failing mode must not hide the other modes' parity results
            msg = str(e).splitlines()[0] if str(e) else ""
            print(f"{label} FAILED: {type(e).__name__}: {msg}", flush=True)
            res[label] = {"failed": f"{type(e).__name__}: {msg}"}
            mm.soft_empty_cache(force=True)
            continue
        entry = {"first_s": round(first, 3), "best_s": round(min(times[1:]) if len(times) > 1 else first, 3),
                 "video_abs_max": outs[0].abs().max().item(), "video_mean": outs[0].mean().item()}
        runs.append((label, outs))
        # release this mode's cached allocator memory, as ComfyUI does between prompts, so the next mode
        # (e.g. an unsplit 1-GPU run after a split) starts with the VRAM a real prompt would have
        outs = None
        mm.soft_empty_cache(force=True)
        res[label] = entry
        print(label, json.dumps(entry), flush=True)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            (la, a), (lb, b) = runs[i], runs[j]
            dv = (a[0] - b[0]).abs().max().item()
            da = (a[1] - b[1]).abs().max().item()
            print(f"DIFF {la} vs {lb}: video {dv:.6g} audio {da:.6g}", flush=True)
    ms_cache = importlib.import_module("ComfyUI-H3-MultiStream.multistream.cache")
    print("CACHE", json.dumps(ms_cache.all_stats()), flush=True)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
