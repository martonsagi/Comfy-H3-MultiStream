# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Shared logger and formatting helpers for H3 MultiStream.

Everything logs through the "h3_multistream" logger, which propagates to ComfyUI's console handler. INFO carries one line
per meaningful event (node activation, sampling step, cache fill/evict/hit, VAE decode, worker lifecycle); DEBUG carries
per-block and per-chunk detail. Set H3MS_DEBUG=1 to lower this logger to DEBUG without touching ComfyUI's own level.
"""
import logging
import os
import sys
import time

import psutil
import torch

try:
    import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

PREFIX = "[H3 MultiStream]"


_last_emit = [0.0]


def _break_progress_line():
    """tqdm progress bars (sampler, decode) redraw with carriage returns and leave their line unterminated, so a record
    emitted while one is active would be glued onto the bar's text (in the journal and in ComfyUI's web log alike).
    If any bar has printed since this logger's last line, end that line first. The formatter prefixes [INFO] to the
    message, so the break has to go to the stream before the record is emitted, not into the message."""
    if _tqdm is None:
        return
    try:
        bars = list(getattr(_tqdm.tqdm, "_instances", None) or ())
        if not bars:
            return
        latest = max((getattr(b, "last_print_t", 0.0) or 0.0) for b in bars)
        if latest >= _last_emit[0]:
            sys.stderr.write("\n")
            sys.stderr.flush()
    except Exception:
        pass


class _PrefixFilter(logging.Filter):
    """Prepend the main prefix to every record of this logger (subsystem tags like [VAE split] stay after it), and start
    the record on a fresh line when a progress bar left the current one open."""

    def filter(self, record):
        if isinstance(record.msg, str) and not record.msg.startswith(PREFIX):
            record.msg = f"{PREFIX} {record.msg}"
        _break_progress_line()
        _last_emit[0] = time.time()
        return True


log = logging.getLogger("h3_multistream")
if not any(isinstance(f, _PrefixFilter) for f in log.filters):
    log.addFilter(_PrefixFilter())
if os.environ.get("H3MS_DEBUG") == "1":
    log.setLevel(logging.DEBUG)

GIB = 2**30


def gib(n):
    return f"{n / GIB:.2f} GiB"


_CGROUP_ROOT = "/sys/fs/cgroup"


def _read_int(path):
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _stat_values(path):
    values = {}
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(" ")
                try:
                    values[name] = int(value)
                except ValueError:
                    pass
    except OSError:
        pass
    return values


def _reclaimable(stat, prefix=""):
    """File page cache the kernel can evict: active + inactive, less pages waiting to be written.

    Counting only inactive_file under-reports badly right after model loading: the freshly read safetensors
    mmaps sit on the ACTIVE list (50 GiB active against 0.4 GiB inactive was seen on a pod), and a cache
    fill checked at that moment disabled itself although the pages were evictable. Shared memory is on the
    anon LRU in both cgroup versions, so it is not included here."""
    get = lambda key: stat.get(prefix + key, 0)
    cache = get("active_file") + get("inactive_file")
    return max(0, cache - get("dirty" if prefix else "file_dirty") - get("writeback" if prefix else "file_writeback"))


# Headroom kept out of the reclaimable figure, as a fraction of the container limit: evicting every file page
# would push the mmap'd model files ComfyUI still reads back to disk.
RECLAIM_MARGIN = 0.05


def memory_limit(root=_CGROUP_ROOT):
    """(limit, available) in bytes inside a container memory limit (cgroup v2, else v1), or (None, None) without one.

    psutil reports the host's memory inside containers (a 93 GB Runpod pod showed "126 GiB"), so the weight caches
    would fill past the pod's limit. Available = limit - usage + reclaimable file cache - a margin."""
    limit = _read_int(os.path.join(root, "memory.max"))
    usage = _read_int(os.path.join(root, "memory.current"))
    reclaimable = _reclaimable(_stat_values(os.path.join(root, "memory.stat")))
    if limit is None or usage is None:
        v1 = os.path.join(root, "memory")
        limit = _read_int(os.path.join(v1, "memory.limit_in_bytes"))
        usage = _read_int(os.path.join(v1, "memory.usage_in_bytes"))
        reclaimable = _reclaimable(_stat_values(os.path.join(v1, "memory.stat")), "total_")
        if limit is None or usage is None or limit >= 1 << 60:   # v1 reports "no limit" as a huge number
            return None, None
    reclaimable = max(0, reclaimable - int(limit * RECLAIM_MARGIN))
    return limit, max(0, limit - usage + reclaimable)


def ram_available(root=_CGROUP_ROOT):
    """(available, total) system RAM in bytes, bounded by the container memory limit when there is one."""
    vm = psutil.virtual_memory()
    limit, available = memory_limit(root)
    if limit is None:
        return vm.available, vm.total
    return min(vm.available, available), min(vm.total, limit)


def ram():
    available, total = ram_available()
    limited = memory_limit()[0] is not None
    return f"RAM available {available / GIB:.1f} of {total / GIB:.0f} GiB{' (container limit)' if limited else ''}"


def vram(device):
    try:
        if device is None or getattr(device, "type", None) != "cuda":
            return "n/a"
        free, total = torch.cuda.mem_get_info(device)
        return f"cuda:{device.index} free {free / GIB:.1f}/{total / GIB:.1f} GiB"
    except Exception:
        return f"{device}: n/a"


def used(device):
    """Driver-level VRAM in use on the device (every process and allocator: ComfyUI, staged weights, the VAE worker)."""
    try:
        free, total = torch.cuda.mem_get_info(device)
        return f"{(total - free) / GIB:.1f}/{total / GIB:.0f} GiB"
    except Exception:
        return "n/a"


def peak(device):
    """Peak PyTorch allocation on the device since the last reset_peak_memory_stats (the split resets it every step)."""
    try:
        return f"{torch.cuda.max_memory_allocated(device) / GIB:.1f} GiB"
    except Exception:
        return "n/a"
