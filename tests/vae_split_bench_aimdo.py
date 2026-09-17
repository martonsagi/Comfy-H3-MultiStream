# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Split VAE decode vs single-GPU cached decode under DynamicVRAM, initialised like ComfyUI's main.py.

    python tests/vae_split_bench_aimdo.py [--latent-t 57] [--cases 0,0+1,0+0+1,bytes@0+1]

Video VAE (int8 convrot) on cuda:1 via SelectVAEDevice gpu:1, like the workflow. Caches warm first, then a single-GPU
reference decode, then every case: a list of worker GPUs (`+`-separated, may repeat a GPU or name cuda:1 itself, so more
GPU ranks than physical cards can be tested) with an optional transport prefix (`shm@` / `bytes@`). Each case decodes
twice (the first includes worker start and VAE load); both must equal the reference exactly.
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
ap.add_argument("--latent-t", type=int, default=57, help="latent frames (57 = 192 video frames, 11 chunks)")
ap.add_argument("--cases", default="0,0+1,0+0+1,bytes@0+1",
                help="comma-separated worker GPU lists, e.g. 0 (2 GPU ranks) or 0+0+1 (4 ranks); prefix bytes@ or shm@")
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

import nodes  # noqa: E402
from comfy_extras.nodes_multigpu import SelectVAEDeviceNode  # noqa: E402

VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
PACK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def main():
    vae_cache = importlib.import_module(f"{PACK}.multistream.vae_cache")
    vae_split = importlib.import_module(f"{PACK}.multistream.vae_split")
    out = SelectVAEDeviceNode.execute(nodes.VAELoader().load_vae(VIDEO_VAE)[0], "gpu:1")
    vae = out.result[0] if hasattr(out, "result") else out[0]
    vae_cache.install(vae, True, 0.0)
    g = torch.Generator().manual_seed(13)
    latent = torch.randn(1, 24, cli.latent_t, 84, 48, generator=g)
    frames = vae.first_stage_model.decode_output_shape(latent.shape)[2]
    print(f"vae device {vae.device}, latent_t {cli.latent_t} -> {frames} frames", flush=True)

    def run(label):
        sync()
        with torch.inference_mode():
            t0 = time.perf_counter()
            dec = vae.decode(latent)
            sync()
            dt = time.perf_counter() - t0
        print(f"CASE {label}: decode {dt:6.2f}s  shape {tuple(dec.shape)}", flush=True)
        return dec.float().cpu(), dt

    run("warmup single (cache fill)")
    ref, _ = run("S single reference")
    failures = []
    for case in cli.cases.split(","):
        transport, _, spec = case.rpartition("@")
        workers = [int(i) for i in spec.split("+")]
        vae_split.TRANSPORT = transport or "auto"
        vae_split.install(vae, True, None, 0.0, worker_devices=workers)
        name = f"{len(workers) + 1} ranks, workers {workers}, transport {vae_split.TRANSPORT}"
        for attempt in (1, 2):
            try:
                got, _ = run(f"{name} #{attempt}")
            except Exception as e:
                print(f"CASE {name} #{attempt} FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append(f"{name} #{attempt}")
                break
            d = (got - ref).abs().max().item()
            print(f"PARITY {name} #{attempt} vs single: max|diff| {d:.6g}", flush=True)
            if d != 0:
                failures.append(f"{name} #{attempt}")
        print(vae_split.release_workers("next case")["message"], flush=True)
    print("RESULT", "all bit-identical" if not failures else "FAIL: " + "; ".join(failures), flush=True)
    os._exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
