# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""GPU selection for H3 MultiStream: which CUDA devices take part, in which rank order, and with which shares.

A GPUSet is what the H3 MS GPU Set node outputs (or what the legacy `second_gpu` input maps to). It is resolved against
the model's primary device at first use into a RankPlan:
  * rank 0 is always the primary device (the one the model and its H3 forward live on);
  * the other ranks follow in the order given (explicit list) or by CUDA index (auto);
  * excluded GPUs, GPUs with too little free VRAM (checked once, at resolve time) and GPUs beyond max_gpus are dropped;
  * one usable GPU means the model runs unsplit; the caches keep working.
Shares are relative speeds per GPU (1.0 = normal). The transformer split sizes each rank's attention heads and tokens by
its share; the VAE split divides its chunks evenly.
"""
import dataclasses

import torch

from .log import GIB, log

MAX_GPUS = 8
DIT_MAX_RANKS = MAX_GPUS   # transformer split: up to 8 ranks
VAE_MAX_RANKS = MAX_GPUS   # VAE split decode: hard cap; the node's max_gpus (default 4) limits it further

LAST_PLANS = {}   # consumer ("dit" / "vae") -> RankPlan.describe(), for the status route


class GPUSelectionError(ValueError):
    pass


def _parse_indices(text, what):
    text = (text or "").strip().lower()
    if text in ("", "auto", "all"):
        return None
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("cuda:"):
            part = part[5:]
        try:
            idx = int(part)
        except ValueError:
            raise GPUSelectionError(f"{what}: '{part}' is not a CUDA index (use e.g. 0,1,3)") from None
        if idx < 0:
            raise GPUSelectionError(f"{what}: negative index {idx}")
        if idx in out:
            raise GPUSelectionError(f"{what}: cuda:{idx} listed twice")
        out.append(idx)
    return tuple(out)


def _parse_shares(text):
    text = (text or "").strip()
    if not text:
        return ()
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            raise GPUSelectionError(f"shares: '{part}' is not a number (use e.g. 1,1,0.7)") from None
        if not v > 0:
            raise GPUSelectionError(f"shares: {v} must be greater than 0")
        out.append(v)
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class GPUSet:
    gpus: tuple = None             # explicit CUDA indices in rank order after the primary; None = all visible
    exclude: tuple = ()
    shares: tuple = ()             # ((cuda_index, share), ...); missing GPUs get 1.0
    min_free_vram_gb: float = 0.0
    max_gpus: int = MAX_GPUS
    implicit_primary: bool = False  # legacy second_gpu: adding the primary is expected, not worth a warning

    @classmethod
    def from_inputs(cls, gpus="auto", exclude="", shares="", min_free_vram_gb=0.0, max_gpus=MAX_GPUS, device_count=None):
        """Validate the node inputs. Index checks need the visible device count (torch.cuda.device_count())."""
        n = torch.cuda.device_count() if device_count is None else device_count
        listed = _parse_indices(gpus, "gpus")
        excluded = _parse_indices(exclude, "exclude") or ()
        for idx in (listed or ()) + tuple(excluded):
            if idx >= n:
                raise GPUSelectionError(f"cuda:{idx} does not exist: {n} CUDA device(s) visible "
                                        f"(CUDA_VISIBLE_DEVICES / --cuda-device limit what ComfyUI sees)")
        share_values = _parse_shares(shares)
        order = [i for i in (listed if listed is not None else range(n)) if i not in excluded]
        if share_values and len(share_values) != len(order):
            raise GPUSelectionError(f"shares: {len(share_values)} value(s) for {len(order)} GPU(s) "
                                    f"({', '.join(f'cuda:{i}' for i in order)})")
        max_gpus = int(max_gpus)
        if not 1 <= max_gpus <= MAX_GPUS:
            raise GPUSelectionError(f"max_gpus must be between 1 and {MAX_GPUS}")
        return cls(gpus=listed, exclude=tuple(excluded), shares=tuple(zip(order, share_values)),
                   min_free_vram_gb=float(min_free_vram_gb), max_gpus=max_gpus)

    @classmethod
    def legacy(cls, second_gpu_index=None):
        """The pre-GPU-Set behaviour: second_gpu -1 = auto, otherwise the primary plus that GPU."""
        if second_gpu_index is None or second_gpu_index < 0:
            return cls()
        return cls(gpus=(int(second_gpu_index),), implicit_primary=True)

    def describe(self):
        parts = [f"gpus {','.join(map(str, self.gpus)) if self.gpus is not None else 'auto'}"]
        if self.exclude:
            parts.append(f"exclude {','.join(map(str, self.exclude))}")
        if self.shares:
            parts.append("shares " + ",".join(f"cuda:{i}={s:g}" for i, s in self.shares))
        if self.min_free_vram_gb:
            parts.append(f"min free {self.min_free_vram_gb:g} GiB")
        if self.max_gpus != MAX_GPUS:
            parts.append(f"max {self.max_gpus}")
        return ", ".join(parts)


@dataclasses.dataclass
class RankPlan:
    devices: list          # torch.device per rank; rank 0 = primary
    shares: list           # float per rank
    notes: list            # human-readable decisions (dropped GPUs, added primary, caps)

    @property
    def n(self):
        return len(self.devices)

    def describe(self):
        ranks = ", ".join(f"{d}{' (primary)' if r == 0 else ''}{'' if s == 1.0 else f' share {s:g}'}"
                          for r, (d, s) in enumerate(zip(self.devices, self.shares)))
        return f"{self.n} rank(s): {ranks}" + (f" | {'; '.join(self.notes)}" if self.notes else "")

    def limited(self, n_max):
        """The first n_max ranks (primary first), for splits that support fewer ranks than selected."""
        if self.n <= n_max:
            return self
        dropped = ", ".join(str(d) for d in self.devices[n_max:])
        return RankPlan(self.devices[:n_max], self.shares[:n_max],
                        self.notes + [f"{self.n} GPUs selected, this split uses {n_max} for now (unused: {dropped})"])


def _free_gib(index):
    try:
        free, _ = torch.cuda.mem_get_info(index)
        return free / GIB
    except Exception:
        return None


def resolve(gpu_set, primary, device_count=None, free_gib=_free_gib):
    """RankPlan for this GPUSet with `primary` (torch.device or CUDA index) as rank 0."""
    gpu_set = gpu_set or GPUSet()
    n_visible = torch.cuda.device_count() if device_count is None else device_count
    p = primary.index if isinstance(primary, torch.device) else primary
    if p is None:
        p = 0
    notes = []
    if p in gpu_set.exclude:
        raise GPUSelectionError(f"the model's device cuda:{p} is excluded; it must take part (it runs the H3 forward)")
    candidates = list(gpu_set.gpus) if gpu_set.gpus is not None else list(range(n_visible))
    candidates = [i for i in candidates if i not in gpu_set.exclude and i < n_visible]
    if p in candidates:
        candidates.remove(p)
    elif gpu_set.gpus is not None and not gpu_set.implicit_primary:
        notes.append(f"added the model's device cuda:{p} as rank 0 (not in the gpus list)")
    ranks = [p]
    for i in candidates:
        if gpu_set.min_free_vram_gb > 0:
            free = free_gib(i)
            if free is not None and free < gpu_set.min_free_vram_gb:
                notes.append(f"skipped cuda:{i}: {free:.1f} GiB free < {gpu_set.min_free_vram_gb:g} GiB")
                continue
        ranks.append(i)
    if len(ranks) > gpu_set.max_gpus:
        notes.append(f"capped at max_gpus {gpu_set.max_gpus} (dropped {', '.join(f'cuda:{i}' for i in ranks[gpu_set.max_gpus:])})")
        ranks = ranks[:gpu_set.max_gpus]
    share_map = dict(gpu_set.shares)
    return RankPlan([torch.device("cuda", i) for i in ranks], [float(share_map.get(i, 1.0)) for i in ranks], notes)


def log_plan(consumer, gpu_set, plan):
    LAST_PLANS[consumer] = {"selection": gpu_set.describe(), "plan": plan.describe()}
    free = ", ".join(f"{d} {f:.1f} GiB free" for d in plan.devices if (f := _free_gib(d.index)) is not None)
    log.info("[GPUs] %s: %s | selection: %s | %s", consumer, plan.describe(), gpu_set.describe(), free or "free VRAM n/a")
    if consumer == "vae" and plan.n > 1 and any(s != 1.0 for s in plan.shares):
        log.info("[GPUs] vae: shares do not apply to the VAE split decode; chunks are divided evenly")
