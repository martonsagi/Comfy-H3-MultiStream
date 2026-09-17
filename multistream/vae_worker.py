# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""VAE decode worker process for H3 MS VAE Split Decode (one per additional GPU).

Started by multistream/vae_split.py as a plain subprocess (not multiprocessing spawn, which would re-run ComfyUI's
main.py) with CUDA_VISIBLE_DEVICES set to its GPU, so its cuda:0 is that GPU. No comfy-aimdo here: ComfyUI's
legacy loading, the VAE fully loaded onto the GPU for a job and unloaded afterwards so it holds no VRAM between scenes.

Protocol over an authenticated AF_UNIX multiprocessing.connection:
  ("load", vae_path)                    -> ("ready", info) | ("load_error", traceback)
  ("job", job_id, [(i, shape, dtype, raw_bytes), ...], transport)
      transport "shm" (default when omitted): ("chunk", job_id, i, shm_name, shape, dtype, seconds) per chunk; the
          decoded chunk sits in POSIX shared memory created here (untracked), the parent unlinks it after reading
      transport "bytes": ("chunkb", job_id, i, nbytes, shape, dtype, seconds) followed by the raw chunk bytes as one
          send_bytes message (for hosts where /dev/shm is too small)
      then ("done", job_id, stats) | ("error", job_id, traceback)
  ("shutdown",)
Everything is logged with timestamps to this process's stderr, which the parent redirects to
vae-worker-gpu<N>.log in its private temporary directory (mode 0700).
"""
import argparse
import inspect
import logging
import os
import sys
import threading
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--address", required=True)
ap.add_argument("--parent-pid", type=int, required=True)
ap.add_argument("--comfy-root", required=True)
ap.add_argument("--gpu", default="?")
cli = ap.parse_args()
sys.argv = sys.argv[:1]  # keep ComfyUI's argument parser on defaults
sys.path.insert(0, cli.comfy_root)
os.chdir(cli.comfy_root)

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format=f"%(asctime)s [H3 MultiStream] [VAE worker gpu{cli.gpu} pid{os.getpid()}] %(levelname)s %(message)s")
wlog = logging.getLogger("h3ms_vae_worker")


def _watch_parent():
    while True:
        if os.getppid() != cli.parent_pid:
            wlog.warning("parent %d is gone (ppid now %d): exiting", cli.parent_pid, os.getppid())
            os._exit(0)
        time.sleep(2)


threading.Thread(target=_watch_parent, daemon=True).start()
t_boot = time.perf_counter()

from multiprocessing import shared_memory  # noqa: E402
_SHM_HAS_TRACK = "track" in inspect.signature(shared_memory.SharedMemory.__init__).parameters  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402

import torch  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

wlog.info("imports done in %.1fs: torch %s, device %s (%s)", time.perf_counter() - t_boot, torch.__version__,
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA",
          os.environ.get("CUDA_VISIBLE_DEVICES"))


def _shm_open(name=None, create=False, size=0):
    """SharedMemory that no resource tracker owns: the creating worker and the reading parent hand segments over
    explicitly. Python 3.13+ has track=False. On 3.12 (the Runpod ComfyUI image) the segment is opened normally and
    unregistered at once, so the tracker neither unlinks it when the process exits nor warns about a leak."""
    if _SHM_HAS_TRACK:
        return shared_memory.SharedMemory(name=name, create=create, size=size, track=False)
    shm = shared_memory.SharedMemory(name=name, create=create, size=size)
    from multiprocessing import resource_tracker
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def _shm_unlink(shm):
    """Remove a segment opened with _shm_open. On 3.12, unlink() would unregister it a second time, which makes the
    resource tracker print a KeyError, so the segment is removed directly."""
    if _SHM_HAS_TRACK:
        shm.unlink()
    else:
        import _posixshmem
        _posixshmem.shm_unlink(shm._name)


def _send_chunk(conn, job_id, i, host, dt, transport):
    n = host.numel() * host.element_size()
    dtype = str(host.dtype).replace("torch.", "")
    flat = host.reshape(-1).view(torch.uint8)
    if transport == "bytes":
        conn.send(("chunkb", job_id, i, n, tuple(host.shape), dtype, dt))
        if n:
            conn.send_bytes(flat.numpy())
        return n
    shm = _shm_open(create=True, size=max(n, 1))
    view = torch.frombuffer(shm.buf, dtype=torch.uint8, count=n)
    view.copy_(flat)
    del view
    shm.close()
    conn.send(("chunk", job_id, i, shm.name, tuple(host.shape), dtype, dt))
    return n


def main():
    conn = Client(cli.address, family="AF_UNIX", authkey=bytes.fromhex(os.environ["H3MS_AUTHKEY"]))
    wlog.info("connected to parent %d", cli.parent_pid)
    vae = None
    loaded_path = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            wlog.info("connection closed by parent: exiting")
            break
        op = msg[0]
        if op == "shutdown":
            wlog.info("shutdown requested")
            break
        if op == "load":
            path = msg[1]
            try:
                if path != loaded_path:
                    t0 = time.perf_counter()
                    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                    vae.throw_exception_if_invalid()
                    loaded_path = path
                    wlog.info("loaded %s in %.1fs: device %s, dtype %s", os.path.basename(path),
                              time.perf_counter() - t0, vae.device, vae.vae_dtype)
                conn.send(("ready", {"device": str(vae.device), "vae_dtype": str(vae.vae_dtype),
                                     "torch": torch.__version__, "pid": os.getpid()}))
            except Exception:
                tb = traceback.format_exc()
                wlog.error("load failed for %s:\n%s", path, tb)
                conn.send(("load_error", tb))
            continue
        if op == "job":
            job_id, chunks = msg[1], msg[2]
            transport = msg[3] if len(msg) > 3 else "shm"
            t_job = time.perf_counter()
            sent = 0
            try:
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    mm.load_models_gpu([vae.patcher], force_full_load=True)
                    load_s = time.perf_counter() - t0
                    torch.cuda.reset_peak_memory_stats()
                    wlog.info("job %d: %d chunk(s), transport %s, VAE on GPU in %.2fs", job_id, len(chunks), transport,
                              load_s)
                    model = vae.first_stage_model
                    for i, shape, dtype, raw in chunks:
                        t_chunk = time.perf_counter()
                        clip = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(getattr(torch, dtype)).view(shape)
                        out = model._adaptive_decode(clip.to(vae.device))
                        host = out.to("cpu")
                        del out
                        dt = time.perf_counter() - t_chunk
                        n = _send_chunk(conn, job_id, i, host, dt, transport)
                        sent += n
                        wlog.info("job %d: chunk %d latent %s -> %s in %.2fs (%.0f MiB)", job_id, i, tuple(shape),
                                  tuple(host.shape), dt, n / 2**20)
                        del host
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                stats = {"seconds": round(time.perf_counter() - t_job, 2), "load_seconds": round(load_s, 2),
                         "peak_vram_gib": round(peak, 2), "sent_mib": round(sent / 2**20, 1), "transport": transport}
                wlog.info("job %d done: %s", job_id, stats)
                conn.send(("done", job_id, stats))
            except Exception:
                tb = traceback.format_exc()
                wlog.error("job %d failed:\n%s", job_id, tb)
                conn.send(("error", job_id, tb))
                try:
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                except Exception:
                    pass
    os._exit(0)


if __name__ == "__main__":
    main()
