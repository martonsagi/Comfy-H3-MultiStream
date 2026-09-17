# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Text-encoder timing and cache parity under DynamicVRAM, initialised like ComfyUI's main.py.

    python tests/te_bench_aimdo.py --prompt-json WORKFLOW_API.json --image IMAGE.png [--te TE.safetensors] [--baseline-only]

Text encoder (default qwen3vl_32b nvfp4 awq, type minimax); prompt, width and height from the MiniMaxH3ImageToVideo node
of an API-format workflow; the image resized to that canvas exactly as the node does. Every case after the first runs
after mm.unload_all_models(), which is what loading the DiT does to the encoder between scenes.

  U1  uncached, prompt 1 (cold)
  U2  uncached, prompt 2                             <- reference output for prompt 2
  --- install H3 MS Text Encoder Cache: weight cache on, output cache off
  W1  prompt 3 (fills the weight cache)
  W2  prompt 2 (weight cache warm)                    == U2 ?
  --- output cache on
  W3  prompt 2 (miss, stored)
  W4  prompt 2 (hit)                                  == U2 ?
"""
import argparse
import importlib
import json
import logging
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
ap = argparse.ArgumentParser()
ap.add_argument("--prompt-json", required=True, help="API-format workflow containing a MiniMaxH3ImageToVideo node")
ap.add_argument("--image", required=True, help="start image: file name under ComfyUI/input")
ap.add_argument("--te", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                help="text encoder file name under models/text_encoders")
ap.add_argument("--baseline-only", action="store_true")
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import nodes  # noqa: E402
from comfy_extras.nodes_minimax_h3 import _resize  # noqa: E402


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def main():
    with open(os.path.expanduser(cli.prompt_json)) as f:
        doc = json.load(f)
    doc = doc.get("prompt", doc)
    node = next(n["inputs"] for n in doc.values() if n.get("class_type") == "MiniMaxH3ImageToVideo")
    base = node["prompt"]
    clip = nodes.CLIPLoader().load_clip(cli.te, "minimax", "default")[0]
    raw = nodes.LoadImage().load_image(cli.image)[0]
    image = _resize(raw[:1], node["width"], node["height"], "disabled")
    prompts = [base, base + " She smiles.", base + " She waves."]

    def case(label, text, unload=True):
        if unload:
            mm.unload_all_models()
            mm.soft_empty_cache()
        sync()
        with torch.inference_mode():
            t0 = time.perf_counter()
            tokens = clip.tokenize(text, images=[image])
            cond = clip.encode_from_tokens_scheduled(tokens)
            sync()
            dt = time.perf_counter() - t0
        c = cond[0][0].float().cpu()
        print(f"CASE {label}: {dt:6.2f}s tokens {c.shape[1]}", flush=True)
        return c, dt

    res = {}
    _, res["U1_cold"] = case("U1 uncached cold, prompt 1", prompts[0], unload=False)
    ref, res["U2_uncached_after_unload"] = case("U2 uncached after unload, prompt 2", prompts[1])
    if cli.baseline_only:
        print("RESULT", json.dumps(res))
        os._exit(0)

    te_cache = importlib.import_module("ComfyUI-H3-MultiStream.multistream.te_cache")
    cache = importlib.import_module("ComfyUI-H3-MultiStream.multistream.cache")
    te_cache.install(clip, weight_cache=True, cond_cache_entries=0, reserve_gib=0.0)
    _, res["W1_fill"] = case("W1 weight cache fill, prompt 3", prompts[2])
    w2, res["W2_weightcache_after_unload"] = case("W2 weight cache warm, prompt 2", prompts[1])
    print(f"PARITY W2 vs U2: max|diff| {(w2 - ref).abs().max().item():.6g}", flush=True)
    te_cache.install(clip, weight_cache=True, cond_cache_entries=64, reserve_gib=0.0)
    _, res["W3_outcache_miss"] = case("W3 output cache miss, prompt 2", prompts[1])
    w4, res["W4_outcache_hit"] = case("W4 output cache hit, prompt 2", prompts[1])
    print(f"PARITY W4 vs U2: max|diff| {(w4 - ref).abs().max().item():.6g}", flush=True)
    print("CACHE", json.dumps(cache.all_stats()), json.dumps(te_cache.COND_CACHE.stats()), flush=True)
    print("RESULT", json.dumps({k: round(v, 2) for k, v in res.items()}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
