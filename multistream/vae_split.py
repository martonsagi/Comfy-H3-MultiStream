# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
# Contains logic derived from ComfyUI (GPL-3.0): the MiniMax H3 video VAE temporal chunk plan and blend/write loop (comfy/ldm/minimax/vae.py).
"""Split the MiniMax H3 video VAE's temporal chunk decode across GPUs: ComfyUI's process plus one worker process per
additional GPU.

decode_temporal decodes the latent in chunks of tokens_chunk_size + token_overlap tokens; each chunk goes through
_adaptive_decode independently (spatial tiling inside, no instance state), and the only coupling between chunks is the
ordered overlap blend + write into the host output buffer afterwards.

Threads do not work for this model (measured 19.3 s single vs ~24 s split on 2 GPUs): Python threads on one GIL starve
each other. Every additional GPU therefore gets its own process (multistream/vae_worker.py, own GIL, own CUDA context,
CUDA_VISIBLE_DEVICES = that GPU, no aimdo). With N GPU ranks, chunk i belongs to rank i % N: rank 0 is ComfyUI's process
on the VAE's device, ranks 1..N-1 are the workers. Each worker gets its chunks as one job up front; ComfyUI's process
decodes its own chunks and replays upstream's blend + write loop in chunk order, taking each worker chunk as it arrives.

Decoded chunks come back through POSIX shared memory, or as raw bytes over the worker's socket when /dev/shm is too small
for them (containers often mount only 64 MB) or H3MS_VAE_TRANSPORT=bytes.
"""
import inspect
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

import torch

import comfy.model_management as mm

from . import gpus as ms_gpus
from . import hooks as ms_hooks
from .log import log, vram

_CFG_ATTR = "_h3ms_vae_split_cfg"
_ORIG_ATTR = "_h3ms_orig_decode_temporal"
_WRAP_ATTR = "_h3ms_wrap_decode_temporal"
_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_worker.py")
START_TIMEOUT = float(os.environ.get("H3MS_VAE_WORKER_START_TIMEOUT", "300"))
TRANSPORT = os.environ.get("H3MS_VAE_TRANSPORT", "auto").strip().lower()   # auto | shm | bytes
DEFAULT_MAX_GPUS = 4   # the gain flattens after 3-4 GPUs: ~11 chunks at 192 frames plus the serial blend/write
_SHM_MARGIN = 256 * 2**20
_SHM_HAS_TRACK = "track" in inspect.signature(shared_memory.SharedMemory.__init__).parameters
_RUN_DIR = None
_RUN_DIR_LOCK = threading.Lock()


def _run_dir():
    """Private per-process directory (mode 0700) for worker sockets and logs, removed when ComfyUI exits."""
    global _RUN_DIR
    with _RUN_DIR_LOCK:
        if _RUN_DIR is None:
            _RUN_DIR = tempfile.mkdtemp(prefix=f"h3ms-{os.getpid()}-")
        return _RUN_DIR


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


class WorkerError(RuntimeError):
    pass


def _log_tail(path, lines=15):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-lines:]).rstrip()
    except OSError:
        return "(no worker log)"


class _Worker:
    """One worker process on one GPU. slot > 0 only when a test runs several workers on the same GPU."""

    def __init__(self, gpu_index, slot=0):
        self.gpu_index = gpu_index
        self.slot = slot
        self.name = f"GPU {gpu_index}" + (f" slot {slot}" if slot else "")
        suffix = f"gpu{gpu_index}" + (f"-{slot}" if slot else "")
        self.lock = threading.Lock()
        self.proc = None
        self.conn = None
        self.loaded_path = None
        self.busy = 0
        self.jobs = itertools.count(1)
        self.started_at = None
        self.jobs_done = 0
        self._suffix = suffix
        self.log_path = os.path.join(_run_dir(), f"vae-worker-{suffix}.log")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    def state(self):
        if self.proc is None:
            return "stopped"
        code = self.proc.poll()
        if code is None:
            return "running" if self.conn is not None else "starting"
        return f"exited({code})"

    def status(self):
        return {"gpu": self.gpu_index, "slot": self.slot, "state": self.state(),
                "pid": self.proc.pid if self.proc else None, "busy": self.busy > 0,
                "loaded": os.path.basename(self.loaded_path) if self.loaded_path else None,
                "jobs_done": self.jobs_done,
                "uptime_s": round(time.monotonic() - self.started_at, 1) if self.started_at and self.alive() else None,
                "log": self.log_path}

    def _start(self):
        if self.proc is not None and self.proc.poll() is not None:
            log.warning("[VAE split] worker on %s had exited with %s; restarting. Last log lines:\n%s", self.name,
                        self.proc.returncode, _log_tail(self.log_path))
        authkey = os.urandom(16)
        address = os.path.join(_run_dir(), f"vae-{self._suffix}-{time.time_ns()}.sock")
        listener = Listener(address, family="AF_UNIX", authkey=authkey)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env["H3MS_AUTHKEY"] = authkey.hex()
        env["PYTHONPATH"] = _COMFY_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        log_file = open(self.log_path, "w")
        t0 = time.perf_counter()
        log.info("[VAE split] starting worker process for %s (log %s)", self.name, self.log_path)
        self.proc = subprocess.Popen([sys.executable, _WORKER_SCRIPT, "--address", address,
                                      "--parent-pid", str(os.getpid()), "--comfy-root", _COMFY_ROOT,
                                      "--gpu", self._suffix[3:]],
                                     env=env, cwd=_COMFY_ROOT, stdout=log_file, stderr=log_file)
        log_file.close()
        accepted = {}

        def accept():
            try:
                accepted["conn"] = listener.accept()
            except Exception as e:
                accepted["error"] = e

        th = threading.Thread(target=accept, daemon=True)
        th.start()
        th.join(START_TIMEOUT)
        listener.close()
        if "conn" not in accepted:
            self.proc.kill()
            tail = _log_tail(self.log_path)
            log.error("[VAE split] worker on %s did not connect within %.0fs: %s\n%s", self.name, START_TIMEOUT,
                      accepted.get("error", "timeout"), tail)
            raise WorkerError(f"VAE worker on {self.name} did not connect within {START_TIMEOUT:.0f}s "
                              f"(log: {self.log_path})")
        self.conn = accepted["conn"]
        self.loaded_path = None
        self.started_at = time.monotonic()
        log.info("[VAE split] worker for %s connected in %.1fs (pid %d)", self.name, time.perf_counter() - t0,
                 self.proc.pid)

    def _recv(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.conn.poll(1.0):
                return self.conn.recv()
            if self.proc.poll() is not None:
                tail = _log_tail(self.log_path)
                log.error("[VAE split] worker on %s died with exit code %s. Last log lines:\n%s", self.name,
                          self.proc.returncode, tail)
                raise WorkerError(f"VAE worker on {self.name} exited with {self.proc.returncode} "
                                  f"(log: {self.log_path})")
            if deadline is not None and time.monotonic() > deadline:
                raise WorkerError(f"VAE worker on {self.name} timed out (log: {self.log_path})")

    def ensure(self, vae_path):
        """Start the worker if needed and make sure it has vae_path loaded."""
        with self.lock:
            if not self.alive():
                self._start()
            if self.loaded_path != vae_path:
                t0 = time.perf_counter()
                self.conn.send(("load", vae_path))
                msg = self._recv(START_TIMEOUT)
                if msg[0] != "ready":
                    log.error("[VAE split] worker %s failed to load %s:\n%s", self.name,
                              os.path.basename(vae_path), msg[1] if len(msg) > 1 else msg)
                    raise WorkerError(f"VAE worker failed to load {vae_path}:\n{msg[1] if len(msg) > 1 else msg}")
                self.loaded_path = vae_path
                log.info("[VAE split] worker %s ready with %s in %.1fs: %s", self.name, os.path.basename(vae_path),
                         time.perf_counter() - t0, msg[1])

    def shutdown(self, reason="shutdown"):
        pid = self.proc.pid if self.proc else None
        try:
            if self.alive():
                self.conn.send(("shutdown",))
                self.proc.wait(timeout=10)
        except Exception:
            pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        if pid is not None:
            log.info("[VAE split] worker on %s (pid %s) stopped: %s, %d job(s) served", self.name, pid, reason,
                     self.jobs_done)
        self.conn = None
        self.loaded_path = None


_WORKERS = {}   # (gpu_index, slot) -> _Worker
_WORKERS_LOCK = threading.Lock()


def _worker(gpu_index, slot=0):
    with _WORKERS_LOCK:
        w = _WORKERS.get((gpu_index, slot))
        if w is None:
            w = _Worker(gpu_index, slot)
            _WORKERS[(gpu_index, slot)] = w
        return w


def worker_status():
    with _WORKERS_LOCK:
        return [w.status() for w in _WORKERS.values()]


def release_workers(reason="released on request"):
    """Stop every idle worker process (they restart on the next split decode). Busy workers are left alone."""
    with _WORKERS_LOCK:
        workers = list(_WORKERS.items())
        busy = [w.name for _, w in workers if w.busy]
        if busy:
            msg = f"VAE worker(s) on {', '.join(busy)} are decoding right now; nothing released"
            log.warning("[VAE split] release refused: %s", msg)
            return {"released": 0, "busy": True, "message": msg}
        released = []
        for key, w in workers:
            if w.proc is not None:
                w.shutdown(reason)
                released.append(w.name)
            del _WORKERS[key]
    msg = (f"released VAE worker(s) on {', '.join(released)}; they restart on the next split decode"
           if released else "no VAE workers were running")
    log.info("[VAE split] %s", msg)
    return {"released": len(released), "busy": False, "message": msg}


def shutdown_workers():
    with _WORKERS_LOCK:
        for w in _WORKERS.values():
            if w.proc is not None:
                w.shutdown("ComfyUI exiting")
    if _RUN_DIR is not None:
        shutil.rmtree(_RUN_DIR, ignore_errors=True)


import atexit  # noqa: E402

atexit.register(shutdown_workers)


def _vae_path(vae):
    init = getattr(getattr(vae, "patcher", None), "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    path = init[1][0]
    return os.path.realpath(path) if isinstance(path, str) and os.path.isfile(path) else None


def _slots(indices):
    """[(gpu, slot)] for a list of worker GPU indices; a repeated GPU (tests only) gets increasing slots."""
    seen = {}
    out = []
    for i in indices:
        out.append((int(i), seen.get(int(i), 0)))
        seen[int(i)] = seen.get(int(i), 0) + 1
    return out


def _worker_keys(primary_device, cfg):
    """(worker keys, full plan, used plan) for a decode on primary_device. No keys = decode unsplit."""
    if cfg.get("worker_devices") is not None:
        return _slots(cfg["worker_devices"]), None, None
    full = ms_gpus.resolve(cfg["gpu_set"], primary_device)
    limit = max(1, min(int(cfg.get("max_gpus", DEFAULT_MAX_GPUS)), ms_gpus.VAE_MAX_RANKS))
    plan = full.limited(limit)
    return _slots(d.index for d in plan.devices[1:]), full, plan


def _ensure_all(workers, path):
    """Start / load every worker in parallel (each cold start is seconds of process start plus a VAE load)."""
    errors = []

    def one(w):
        try:
            w.ensure(path)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=one, args=(w,), name=f"h3ms-vae-ensure-{w._suffix}", daemon=True)
               for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def uninstall(model):
    """Put the original decode_temporal back."""
    return ms_hooks.uninstall(model, _ORIG_ATTR, _WRAP_ATTR, ("decode_temporal",), "VAE split")


def install(vae, enabled=True, second_gpu_index=None, reserve_gib=None, gpu_set=None, max_gpus=DEFAULT_MAX_GPUS,
            worker_devices=None):
    """Hook the H3 video VAE's decode_temporal and pre-start the workers in the background.

    GPUs: gpu_set (H3 MS GPU Set) if given, else the legacy second_gpu_index; at most max_gpus GPU ranks (ComfyUI's
    process counts as one). worker_devices (tests only): explicit worker GPU indices, which may repeat a GPU or name the
    VAE's own GPU. reserve_gib: unused, kept for call compatibility."""
    gpu_set = gpu_set if gpu_set is not None else ms_gpus.GPUSet.legacy(second_gpu_index)
    model = vae.first_stage_model
    if not hasattr(model, "decode_temporal") or not hasattr(model, "_decode_temporal_chunks"):
        log.warning("[VAE split] not a MiniMax H3 video VAE (%s): left unchanged", type(model).__name__)
        return
    if not enabled:
        # the hook lives on the shared first_stage_model, not on this node: leaving it attached would
        # keep splitting decodes (and logging) after the node is bypassed. Remove it for real.
        if uninstall(model):
            release_workers("VAE split node disabled")
        else:
            log.info("[VAE split] node disabled: nothing was hooked")
        return
    path = _vae_path(vae)
    if enabled and path is None:
        log.warning("[VAE split] the VAE loader has no reload factory: decode left unsplit")
    cfg = {"enabled": bool(enabled) and path is not None, "gpu_set": gpu_set, "max_gpus": int(max_gpus),
           "worker_devices": list(worker_devices) if worker_devices is not None else None, "path": path}
    setattr(model, _CFG_ATTR, cfg)
    dev = getattr(vae, "device", None)
    primary = dev if isinstance(dev, torch.device) else torch.device("cuda", 0)
    keys, full, plan = _worker_keys(primary, cfg) if cfg["enabled"] else ([], None, None)
    if full is not None:
        ms_gpus.log_plan("vae", gpu_set, full)
        if plan is not full:
            log.info("[VAE split] %s (max_gpus %d)", plan.notes[-1], cfg["max_gpus"])
    if cfg["enabled"] and keys:
        workers = [_worker(*k) for k in keys]
        log.info("[VAE split] node active: %s on %s, %d GPU ranks (workers on %s), decode will be split",
                 os.path.basename(path), primary, len(workers) + 1, ", ".join(w.name for w in workers))

        def prestart():
            try:
                _ensure_all(workers, path)
            except Exception:
                log.exception("[VAE split] worker pre-start failed (will retry at decode)")

        threading.Thread(target=prestart, name="h3ms-vae-worker-prestart", daemon=True).start()
    elif enabled:
        log.warning("[VAE split] not splitting: %s, reload factory %s",
                    "1 usable GPU" if full is not None else f"{torch.cuda.device_count()} CUDA device(s)",
                    "found" if path else "missing")
    if getattr(model, _ORIG_ATTR, None) is not None:
        return
    original = model.decode_temporal
    setattr(model, _ORIG_ATTR, original)

    def decode_temporal(z, output_buffer=None):
        c = getattr(model, _CFG_ATTR)
        if not c["enabled"] or z.device.type != "cuda":
            log.info("[VAE split] decode unsplit (enabled=%s, device %s)", c["enabled"], z.device)
            return original(z, output_buffer)
        keys, _, _ = _worker_keys(z.device, c)
        if not keys:
            log.info("[VAE split] decode unsplit: 1 usable GPU (%s)", z.device)
            return original(z, output_buffer)
        workers = [_worker(*k) for k in keys]
        t0 = time.perf_counter()
        _ensure_all(workers, c["path"])
        wait_s = time.perf_counter() - t0
        if wait_s > 0.5:
            log.info("[VAE split] waited %.1fs for %d worker(s)", wait_s, len(workers))
        for w in workers:
            w.busy += 1
        try:
            return _split_decode_temporal(model, workers, z, output_buffer)
        finally:
            for w in workers:
                w.busy -= 1

    model.decode_temporal = decode_temporal
    setattr(model, _WRAP_ATTR, {"decode_temporal": decode_temporal})
    ms_hooks.remember(model, "VAE split", _ORIG_ATTR, _WRAP_ATTR, ("decode_temporal",))


def _choose_transport(m0, z, spans, n_remote):
    """'shm' when /dev/shm can hold every worker chunk at once (workers may run ahead of the ordered blend), else
    'bytes' over the socket. H3MS_VAE_TRANSPORT=shm|bytes forces a mode."""
    if TRANSPORT in ("shm", "bytes"):
        return TRANSPORT, f"H3MS_VAE_TRANSPORT={TRANSPORT}"
    s, e = spans[0]
    need = 4   # bytes per element: float32 upper bound
    for d in m0.decode_output_shape((z.shape[0], z.shape[1], e - s) + tuple(z.shape[3:])):
        need *= d
    try:
        st = os.statvfs("/dev/shm")
        free = st.f_bavail * st.f_frsize
    except OSError:
        return "bytes", "/dev/shm is not available"
    total = need * n_remote + _SHM_MARGIN
    if free < total:
        return "bytes", f"/dev/shm has {free / 2**30:.2f} GiB free, {n_remote} worker chunk(s) need ~{total / 2**30:.2f} GiB"
    return "shm", None


def _discard(worker, msg):
    """Free what a stale chunk message (from an abandoned job) still holds."""
    log.warning("[VAE split] discarding stale chunk %s from job %s (worker %s)", msg[2], msg[1], worker.name)
    if msg[0] == "chunkb":
        worker.conn.recv_bytes()
        return
    try:
        stale = _shm_open(name=msg[3])
        stale.close()
        _shm_unlink(stale)
    except FileNotFoundError:
        pass


def _fetch_chunk(worker, msg, device):
    """msg: ("chunk", job, i, shm_name, shape, dtype, s) or ("chunkb", job, i, nbytes, shape, dtype, s) + payload."""
    shape, dtype = msg[4], getattr(torch, msg[5])
    if msg[0] == "chunkb":
        buf = bytearray(msg[3])
        worker.conn.recv_bytes_into(buf)
        host = torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(shape) if msg[3] else torch.empty(shape, dtype=dtype)
        return host.to(device)
    n = 1
    for s in shape:
        n *= s
    n *= torch.empty((), dtype=dtype).element_size()
    shm = _shm_open(name=msg[3])
    try:
        host = torch.frombuffer(shm.buf, dtype=torch.uint8, count=n).view(dtype).view(shape)
        out = host.to(device)
        del host
    finally:
        shm.close()
        _shm_unlink(shm)
    return out


def _split_decode_temporal(m0, workers, z, output_buffer):
    dev0 = z.device
    n_ranks = 1 + len(workers)
    t_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(dev0)
    # mirror of MiniMaxH3VideoVAE.decode_temporal (comfy/ldm/minimax/vae.py): chunk plan and padding
    chunk_dec = m0.tokens_chunk_size * m0.vae_ratio_t
    split_count = int(m0.token_drop > 0) + 1
    if output_buffer is None:
        output_buffer = torch.empty(m0.decode_output_shape(z.shape), dtype=torch.float32,
                                    device=mm.intermediate_device())
    latent_t = z.shape[2]
    pad_tokens, num_chunks = m0._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)
    spans = [(i * m0.tokens_chunk_size, i * m0.tokens_chunk_size + m0.tokens_chunk_size + m0.token_overlap)
             for i in range(num_chunks)]

    # raw bytes, never torch tensors: pickling a tensor over a Connection uses torch's fd-sharing reduction, which calls
    # back into multiprocessing's resource_sharer with the parent's authkey and fails for a plain subprocess
    remote = [[] for _ in workers]
    for i, (s, e) in enumerate(spans):
        owner = i % n_ranks
        if owner:
            clip = z[:, :, s:e, :, :].to("cpu").contiguous()
            remote[owner - 1].append((i, tuple(clip.shape), str(clip.dtype).replace("torch.", ""),
                                      clip.reshape(-1).view(torch.uint8).numpy().tobytes()))
    n_remote = sum(len(r) for r in remote)
    transport, why = _choose_transport(m0, z, spans, n_remote)
    if transport == "bytes":
        log.warning("[VAE split] chunks return over the worker sockets instead of shared memory: %s", why)
    job_ids = [next(w.jobs) for w in workers]
    log.info("[VAE split] decode jobs %s: latent %s (%d frames out), %d chunks over %d GPU ranks: %d on %s, %s | "
             "transport %s | %s", job_ids, tuple(z.shape[:2]) + (latent_t,) + tuple(z.shape[3:]),
             output_buffer.shape[2], num_chunks, n_ranks, num_chunks - n_remote, dev0,
             ", ".join(f"{len(r)} on {w.name}" for w, r in zip(workers, remote)), transport, vram(dev0))
    for w, jid, chunks in zip(workers, job_ids, remote):
        if chunks:
            with w.lock:
                w.conn.send(("job", jid, chunks, transport))
    results = [{} for _ in workers]
    done = [not r for r in remote]
    stats = [{} for _ in workers]
    remote_wait = 0.0

    def pump(k):
        """Receive one message from worker k and file it."""
        w, jid = workers[k], job_ids[k]
        msg = w._recv()
        if msg[0] in ("chunk", "chunkb") and msg[1] == jid:
            results[k][msg[2]] = _fetch_chunk(w, msg, dev0)
            log.debug("[VAE split] job %d: chunk %d received from %s (%.2fs on worker)", jid, msg[2], w.name, msg[6])
        elif msg[0] == "done" and msg[1] == jid:
            done[k] = True
            stats[k] = msg[2] if len(msg) > 2 else {}
        elif msg[0] == "error":
            log.error("[VAE split] job %s: worker %s failed:\n%s", msg[1], w.name, msg[2])
            raise WorkerError(f"VAE worker {w.name} failed:\n{msg[2]}")
        elif msg[0] in ("chunk", "chunkb"):
            _discard(w, msg)

    def next_remote(i, k):
        nonlocal remote_wait
        tw = time.perf_counter()
        while i not in results[k]:
            if done[k]:
                raise WorkerError(f"VAE worker {workers[k].name} finished job {job_ids[k]} without chunk {i}")
            pump(k)
        remote_wait += time.perf_counter() - tw
        return results[k].pop(i)

    # mirror of the upstream blend + write loop, in chunk order on the primary device
    dec = output_buffer
    dec_overlap = None
    write_pos = 0
    local_seconds = 0.0

    def write_part(part):
        nonlocal write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        part = m0._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(part[:, :, :copy_frames, :, :])
            write_pos += copy_frames

    for i in range(num_chunks):
        mm.throw_exception_if_processing_interrupted()
        owner = i % n_ranks
        if owner == 0:
            tl = time.perf_counter()
            s, e = spans[i]
            clip_dec = m0._adaptive_decode(z[:, :, s:e, :, :])
            dt = time.perf_counter() - tl
            local_seconds += dt
            log.debug("[VAE split] local chunk %d decoded in %.2fs", i, dt)
        else:
            clip_dec = next_remote(i, owner - 1)
        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, m0.frame_pre_padding:, :, :]
            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = m0.blend(dec_overlap, clip_dec_chunk, m0.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.contiguous()
        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None
        del clip_dec

    for k in range(len(workers)):
        while not done[k]:
            pump(k)
        if remote[k]:
            workers[k].jobs_done += 1

    log.info("[VAE split] decode jobs %s done in %.1fs: local %.1fs on %s (peak %.1f GiB), waited %.1fs for workers | %s",
             job_ids, time.perf_counter() - t_start, local_seconds, dev0, torch.cuda.max_memory_allocated(dev0) / 2**30,
             remote_wait, "; ".join(f"{w.name} {st or 'n/a'}" for w, st in zip(workers, stats)))
    return dec
