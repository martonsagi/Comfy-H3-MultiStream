# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
# Contains logic derived from ComfyUI (GPL-3.0): the MiniMax H3 video VAE tiled-decode blend logic (comfy/ldm/minimax/vae.py).
"""Experiment: batch the H3 video VAE's spatial tiles to cut kernel-launch (GIL) overhead on ONE GPU.

    python tests/tile_batch_bench_aimdo.py [--latent-t 27] [--groups 1,4,8,24]

MiniMaxH3VideoVAE.tiled_decode decodes every 256 px tile of a temporal chunk one at a time. split_tiles gives all tiles
the same size (only overlaps vary), so they can go through _decode_pixels as one batch. This replaces tiled_decode with a
version that decodes tiles in groups of G, then runs the identical blend/canvas logic on the precomputed tiles.

Reports single-GPU decode time per group size and max |diff| vs the upstream sequential decode (batching can change
kernel selection, so exactness is not guaranteed and is exactly what this measures).
"""
import argparse
import logging
import os
import sys
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
os.chdir(COMFY)
ap = argparse.ArgumentParser()
ap.add_argument("--latent-t", type=int, default=27)
ap.add_argument("--groups", default="1,4,8,24")
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
from comfy_extras.nodes_multigpu import SelectVAEDeviceNode  # noqa: E402

VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"


def batched_tiled_decode(group):
    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        coords = []
        for i_pos, i_len in zip(y_idx, y_len):
            for j_pos, j_len in zip(x_idx, x_len):
                zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                coords.append((zi, zl, zj, zw))
        decoded = []
        for g0 in range(0, len(coords), group):
            part = coords[g0:g0 + group]
            batch = torch.cat([z[..., zi:zi + zl, zj:zj + zw] for zi, zl, zj, zw in part], dim=0)
            out = self._decode_pixels(batch)
            decoded.extend(out.split(z.shape[0], dim=0))
        # identical blend/canvas logic to upstream tiled_decode, on precomputed tiles
        canvas = None
        row_tails = []
        out_y = 0
        k = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            new_tails = []
            left_tail = None
            out_x = 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                tile = decoded[k]
                k += 1
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i]:, :].clone())
                next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas
    return tiled_decode


def main():
    out = SelectVAEDeviceNode.execute(nodes.VAELoader().load_vae(VIDEO_VAE)[0], "gpu:1")
    vae = out.result[0] if hasattr(out, "result") else out[0]
    model = vae.first_stage_model
    upstream = type(model).tiled_decode
    g = torch.Generator().manual_seed(13)
    latent = torch.randn(1, 24, cli.latent_t, 84, 48, generator=g)
    ty, _, _ = model.split_tiles(84 * model.vae_ratio)
    tx, _, _ = model.split_tiles(48 * model.vae_ratio)
    print(f"vae device {vae.device}; tiles per chunk {len(ty)}x{len(tx)} = {len(ty) * len(tx)}", flush=True)

    def run(label):
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(vae.device)
        with torch.inference_mode():
            t0 = time.perf_counter()
            dec = vae.decode(latent)
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(vae.device) / 2**30
        print(f"CASE {label}: decode {dt:6.2f}s peak {peak:.2f} GiB", flush=True)
        return dec.float().cpu(), dt

    model.tiled_decode = types.MethodType(upstream, model)
    run("warmup upstream")
    ref, t_ref = run("upstream sequential")
    for grp in [int(x) for x in cli.groups.split(",")]:
        model.tiled_decode = types.MethodType(batched_tiled_decode(grp), model)
        run(f"batched G={grp} warmup")
        got, dt = run(f"batched G={grp}")
        d = (got - ref).abs()
        print(f"RESULT G={grp}: {dt:.2f}s ({t_ref / dt:.2f}x) max|diff| {d.max().item():.6g} mean|diff| {d.mean().item():.3g}",
              flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
