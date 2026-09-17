# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""VAE load-vs-compute timing and weight-cache parity under DynamicVRAM, initialised like ComfyUI's main.py.

    python tests/vae_bench_aimdo.py --image IMAGE.png [--latent-t 12]

VAEs (defaults): minimax_h3_video_vae_int8_convrot (on cuda:1 like the workflow's SelectVAEDevice gpu:1) and
minimax_h3_audio_vae_fp32. Every case after the first runs after mm.unload_all_models(), which is what loading the
DiT does to the VAEs between scenes. Load cost does not depend on clip length, so a short latent keeps it quick.

  video: encode(first frame) + decode(latent)   U cold, U after unload, [install cache] W fill, W warm (== U ?)
  audio: decode(latent)                          same
"""
import argparse
import importlib
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
ap.add_argument("--latent-t", type=int, default=12)
ap.add_argument("--gpu", default="gpu:1")
ap.add_argument("--image", required=True, help="image to encode: file name under ComfyUI/input")
ap.add_argument("--video-vae", default="minimax_h3_video_vae_int8_convrot.safetensors")
ap.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
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
from comfy_extras.nodes_multigpu import SelectVAEDeviceNode  # noqa: E402



def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def select(vae):
    out = SelectVAEDeviceNode.execute(vae, cli.gpu)
    return out.result[0] if hasattr(out, "result") else out[0]


def main():
    vae_cache = importlib.import_module("ComfyUI-H3-MultiStream.multistream.vae_cache")
    cache = importlib.import_module("ComfyUI-H3-MultiStream.multistream.cache")
    video = select(nodes.VAELoader().load_vae(cli.video_vae)[0])
    audio = select(nodes.VAELoader().load_vae(cli.audio_vae)[0])
    image = _resize(nodes.LoadImage().load_image(cli.image)[0][:1], 768, 1344, "disabled")
    g = torch.Generator().manual_seed(11)
    v_latent = torch.randn(1, 24, cli.latent_t, 84, 48, generator=g)
    a_latent = torch.randn(1, 32, 2, 80, generator=g)
    print(f"video vae device {video.device}, audio vae device {audio.device}, latent_t {cli.latent_t}", flush=True)

    def run(label, unload):
        if unload:
            mm.unload_all_models()
            mm.soft_empty_cache()
        sync()
        with torch.inference_mode():
            t0 = time.perf_counter()
            enc = video.encode(image)
            sync()
            t1 = time.perf_counter()
            dec = video.decode(v_latent)
            sync()
            t2 = time.perf_counter()
            aud = audio.decode(a_latent)
            sync()
            t3 = time.perf_counter()
        print(f"CASE {label}: video encode {t1 - t0:6.2f}s  video decode {t2 - t1:6.2f}s  audio decode {t3 - t2:6.2f}s  "
              f"total {t3 - t0:6.2f}s", flush=True)
        return [enc.float().cpu(), dec.float().cpu(), aud.float().cpu()]

    run("U1 uncached cold", unload=False)
    ref = run("U2 uncached after unload", unload=True)
    vae_cache.install(video, True, 0.0)
    vae_cache.install(audio, True, 0.0)
    run("W1 cache fill after unload", unload=True)
    got = run("W2 cache warm after unload", unload=True)
    for name, r, o in zip(("video encode", "video decode", "audio decode"), ref, got):
        print(f"PARITY {name}: max|diff| {(r - o).abs().max().item():.6g}", flush=True)
    print("CACHE", cache.all_stats(), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
