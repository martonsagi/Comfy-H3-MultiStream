# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Minimal repro: two threads allocate/compute/free on two GPUs concurrently, with aimdo set up like main.py.

    python tests/alloc_thread_repro.py [--no-aimdo] [--native-alloc] [--seconds 30] [--graph]

Prints iterations per thread every 2 s; a thread whose count stops moving is hung.
`kill -USR1 <pid>` dumps Python stacks.
"""
import argparse
import faulthandler
import os
import signal
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--no-aimdo", action="store_true")
ap.add_argument("--native-alloc", action="store_true")
ap.add_argument("--seconds", type=int, default=30)
ap.add_argument("--graph", action="store_true", help="main thread holds an aimdo malloc graph (paused) meanwhile")
ap.add_argument("--mib", type=int, default=512)
cli = ap.parse_args()
sys.argv = sys.argv[:1]

if not cli.native_alloc:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
os.chdir(COMFY)
faulthandler.register(signal.SIGUSR1, all_threads=True)
print("pid", os.getpid(), "aimdo", not cli.no_aimdo, "alloc", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), flush=True)

from comfy.cli_args import args  # noqa: E402
import comfy_aimdo.control  # noqa: E402

if not cli.no_aimdo:
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not args.disable_nvml_pressure)

import torch  # noqa: E402

import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402

if not cli.no_aimdo:
    assert comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices())
    comfy.memory_management.aimdo_enabled = True
import comfy.model_prefetch  # noqa: E402

counts = [0, 0]
stop = threading.Event()
errors = []


def worker(r):
    dev = torch.device("cuda", r)
    n = cli.mib * 2**20 // 4
    try:
        with torch.cuda.device(dev), torch.inference_mode():
            while not stop.is_set():
                a = torch.empty(n, dtype=torch.float32, device=dev)
                a.normal_()
                b = a * 2.0
                c = torch.empty(n // 3, dtype=torch.int8, device=dev)
                del a
                c.fill_(1)
                del b, c
                torch.cuda.current_stream(dev).synchronize()
                counts[r] += 1
    except BaseException as e:
        errors.append((r, repr(e)))


ctx = mm.cuda_device_context(torch.device("cuda", 1))
with ctx, torch.inference_mode():
    if cli.graph:
        comfy.model_prefetch.malloc_graph_begin(torch.device("cuda", 1))
        _ = torch.empty(1024, device="cuda:1")
    pause = comfy.model_prefetch.pause_malloc_graph(sync=True) if cli.graph else None
    if pause:
        pause.__enter__()
    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in (0, 1)]
    for t in threads:
        t.start()
    last = [0, 0]
    t0 = time.time()
    while time.time() - t0 < cli.seconds:
        time.sleep(2)
        print(f"t={time.time() - t0:4.0f}s counts={counts} delta={[counts[i] - last[i] for i in (0, 1)]} errors={errors}",
              flush=True)
        last = list(counts)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    hung = [t.is_alive() for t in threads]
    if pause:
        pause.__exit__(None, None, None)
    print("DONE hung_threads", hung, "errors", errors, flush=True)
    os._exit(1 if any(hung) else 0)
