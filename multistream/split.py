# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
# Contains logic derived from ComfyUI (GPL-3.0): the MiniMax H3 transformer block and attention computation (comfy/ldm/minimax/model.py), reimplemented per rank.
"""Ulysses-style split of the MiniMax H3 DiT block stack across N GPU ranks, one thread per rank.

Hooks in as a DIFFUSION_MODEL wrapper: H3's own _forward still does embedding, layout,
rope and the final layer on the primary GPU (rank 0). The wrapper installs a blocks_replace
patch on block 0 that runs all blocks split across the ranks, and identity patches on the rest,
so the upstream block loop never casts a block weight on its own.

Rank r owns a contiguous token range and a contiguous attention head group, both sized by its
share (equal shares: even split; uneven counts such as 56 heads on 3 GPUs are fine). Per block:
  adaln + norm1 on local tokens -> all-gather hidden states -> qkv rows of head group r over
  all tokens -> rope + attention -> all-to-all of attention outputs (each rank gets every head
  group's output for its own tokens) -> out_proj + MLP on local tokens.
Every op is ComfyUI's own kernel; slicing int8 qkv rows per head group and running attention
heads independently keeps outputs identical to the single-GPU block.

Memory safety under DynamicVRAM (comfy-aimdo): H3's forward records the primary thread's
allocations into an aimdo malloc graph, backed by cuMemCreate mappings whose access is granted
to the owning GPU only. A peer-to-peer copy touching such memory is an illegal address. So:
  * the aimdo malloc graph is paused for the whole split stack;
  * every tensor that crosses between the H3 forward and the ranks goes through host RAM;
  * the per-block exchange is either host-staged (pinned buffers, default) or P2P, where P2P
    only ever touches tensors the rank threads allocated themselves.
"""
import contextlib
import logging
import math
import os
import threading
import time

import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.minimax import model as H
from comfy.ldm.modules import attention as A
from comfy.quant_ops import QuantizedTensor
try:
    # Private comfy-kitchen helper (output dtype -> int8_linear kernel code). If a comfy-kitchen update removes
    # it, the sliced int8 projection falls back to dequantize + linear instead of failing the whole pack.
    from comfy_kitchen.tensor.int8 import _dtype_code
except ImportError:
    _dtype_code = None

from . import cache as ms_cache
from . import gpus as ms_gpus
from . import cast as ms_cast
from . import sparse as ms_sparse
from . import vram_cache as ms_vram
from .log import gib, log, peak, used, vram

WRAPPER_KEY = "h3_multistream"
OVERRIDE = "optimized_attention_override"
EXCHANGE_MODES = ("host", "p2p")
MIN_PIPELINE_BYTES = 8 * 2**20   # below this an exchange goes single-shot: the barriers cost more
_XFER_STREAMS = {}               # device str -> (send, recv) stream pair, per device per process
_XFER_LOCK = threading.Lock()
_MISSING = object()   # block has no user patch: keep the model-level override
_NONE = object()      # user patch removed the override for this block
UNSAFE = os.environ.get("H3MS_UNSAFE") == "1"   # test only: pre-fix behaviour (no graph pause, direct P2P)
PREFETCH = os.environ.get("H3MS_PREFETCH", "1") != "0"   # copy the next block's weights on a side CUDA stream
LAST_STEP = {}   # summary of the most recent split step, for /h3multistream/status


class MultiStreamError(RuntimeError):
    pass


@contextlib.contextmanager
def comfy_compiler_disabled():
    """Switch off ComfyUI's model compiler (aimdo malloc graph) for one model call.

    H3's forward records allocations into a malloc graph with per-block scopes. The split stack allocates
    across two threads and two GPUs, which that planner cannot follow (pausing it mid-scope still breaks the
    next block's scope pop: 'aimdo memory compile error'). Same approach as ComfyUI-VDN-H3: flip
    --disable-comfy-compiler around the APPLY_MODEL call only, restore afterwards."""
    from comfy.cli_args import args
    if UNSAFE or not hasattr(args, "disable_comfy_compiler") or args.disable_comfy_compiler:
        yield
        return
    args.disable_comfy_compiler = True
    try:
        yield
    finally:
        args.disable_comfy_compiler = False


def make_apply_model_wrapper():
    def apply_model_wrapper(executor, *args, **kwargs):
        with comfy_compiler_disabled():
            return executor(*args, **kwargs)
    return apply_model_wrapper


def _set_aimdo_log_level_like_main():
    """Restore comfy-aimdo's native log level the way ComfyUI's main.py sets it."""
    import comfy_aimdo.control as control
    from comfy.cli_args import args, get_console_log_level
    level = get_console_log_level(args.verbose)
    if level == "DEBUG":
        control.set_log_debug()
    elif level == "DETAIL":
        try:
            control.set_log_detail()
        except AttributeError:
            control.set_log_info()
    elif level == "CRITICAL":
        control.set_log_critical()
    elif level == "ERROR":
        control.set_log_error()
    elif level == "WARNING":
        control.set_log_warning()
    else:
        control.set_log_info()


@contextlib.contextmanager
def aimdo_logging_silenced():
    """comfy-aimdo's allocator hook logs through a Python callback while PyTorch's CudaMallocAsync mutex is held.
    With two rank threads, one waits for the GIL inside the allocator while the other holds the GIL and waits for
    the allocator mutex: a deadlock (native stacks: aimdo_cuda_malloc_async -> aimdo_log -> PyGILState_Ensure vs
    THPVariable_dealloc -> freeAsync -> pthread_mutex_lock). Silence native logging while both ranks run."""
    if not comfy.memory_management.aimdo_enabled or UNSAFE:
        yield
        return
    import comfy_aimdo.control as control
    control.set_log_none()
    try:
        yield
    finally:
        _set_aimdo_log_level_like_main()


def to_device(t, device):
    """Move t to device; a cross-GPU move goes through host RAM (never P2P on memory we did not allocate)."""
    if t is None or not isinstance(t, torch.Tensor):
        return t
    if t.device == device:
        return t
    if not UNSAFE and t.device.type == "cuda" and device.type == "cuda":
        t = t.to("cpu")
    return t.to(device)


def _local_segments(mod_segments, lo, hi, device):
    out = []
    for a, b, row in mod_segments:
        s, e = max(a, lo), min(b, hi)
        if s < e:
            if isinstance(row, torch.Tensor):
                row = to_device(row[s - a:e - a], device)
            out.append((s - lo, e - lo, row))
    return out


_PEER_READY = set()
_PEER_LOCK = threading.Lock()


def _enable_peer_access(devices):
    """Copy between every ordered GPU pair once, from one thread, before the rank threads start.

    The first peer copy between two devices enables peer access (and pool access for PyTorch's cudaMallocAsync
    allocator). Doing that from several rank threads at once failed with 'CUDA error: invalid argument' (4 logical ranks
    on 2 GPUs, p2p exchange)."""
    unique = list(dict.fromkeys(d for d in devices if d.type == "cuda"))
    with _PEER_LOCK:
        for a in unique:
            for b in unique:
                if a == b or (a, b) in _PEER_READY:
                    continue
                with torch.cuda.device(a):
                    torch.ones(1, device=a).to(b)
                torch.cuda.synchronize(a)
                torch.cuda.synchronize(b)
                _PEER_READY.add((a, b))


class _Collective:
    """Exchanges between the N rank threads: all-gather of hidden states, all-to-all of attention outputs.

    Every call has two phases behind an N-party barrier. First each rank stages what it sends (host mode: its device
    tensors into its own pinned buffers; p2p mode: the device tensors as they are). Then each rank copies what it
    receives to its GPU and synchronizes; the second barrier keeps every sender's buffers alive until all receivers are
    done. An error on any rank aborts the barrier, which unblocks the others."""

    def __init__(self, mode, n, devices=(), chunks=0, min_chunk_bytes=MIN_PIPELINE_BYTES):
        if mode not in EXCHANGE_MODES:
            raise MultiStreamError(f"unknown exchange mode {mode!r}")
        if mode == "p2p":
            _enable_peer_access(devices)
        self.mode = mode
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n
        self.seconds = [0.0] * n
        self.sent = [0] * n
        self._pinned = [{} for _ in range(n)]
        # pipelining: chunk along dim 0 so chunk j's upload overlaps chunk j+1's download. Only in
        # host mode -- p2p hands the device tensors over untouched, so there are no two directions to
        # overlap. chunks <= 1 takes the original single-shot path unchanged.
        self.chunks = int(chunks) if mode == "host" else 0
        self.min_chunk_bytes = int(min_chunk_bytes)
        self._bars = [threading.Barrier(n) for _ in range(max(0, self.chunks))] if self.chunks > 1 else []
        # PER-CHUNK publish slots, not one slot per rank. With a single slot a rank that clears the
        # chunk-j barrier first overwrites its own entry with chunk j+1 before a slower rank has read
        # chunk j -- silently wrong results, and only for some rank counts. Per-chunk slots mean
        # writers and readers never touch the same entry, so one barrier per chunk is enough.
        self._pipe = [[None] * max(0, self.chunks) for _ in range(n)]
        self._props = [(0, None)] * n

    def _agree(self, rank, proposal, shapes):
        """Settle on ONE chunk count for every rank, and publish what each rank will send.

        `_plan` works from the rank's own tensor, and the token shares are not equal -- so two ranks
        can propose different counts, and then they wait on a different number of per-chunk barriers
        and the exchange deadlocks. The same barrier carries every rank's send shapes, which is what
        lets a receiver allocate ONE destination for the whole transfer instead of one per chunk.
        Costs one extra barrier, and only when pipelining is switched on at all."""
        self._props[rank] = (proposal, shapes)
        self.barrier.wait()
        return min(p for p, _ in self._props), [sh for _, sh in self._props]

    def _pinned_buffer(self, rank, key, tensor):
        k = (key, tensor.numel(), tensor.dtype)
        buf = self._pinned[rank].get(k)
        if buf is None:
            buf = torch.empty(tensor.numel(), dtype=tensor.dtype, pin_memory=True)
            self._pinned[rank][k] = buf
        return buf

    def _stage(self, rank, parts, stream):
        if self.mode != "host":
            return parts
        staged = {}
        for key, tensor in parts.items():
            buf = self._pinned_buffer(rank, key, tensor)
            buf.copy_(tensor.reshape(-1), non_blocking=True)
            staged[key] = (buf, tensor.shape, tensor.dtype)
        stream.synchronize()
        return staged

    @staticmethod
    def _keep(t, stream):
        """Tell the caching allocator `stream` may still be reading `t` when it is freed.

        A tensor allocated on one stream and touched on another must say so, or the allocator can
        hand the block to a later allocation while the copy is still in flight. Not every allocator
        under this process implements it, so a refusal is not fatal -- the side streams are joined
        before the tensors are used either way."""
        try:
            t.record_stream(stream)
        except (RuntimeError, TypeError, AttributeError, NotImplementedError):
            pass

    def _recv_into(self, srcs, device, stream):
        """H2D of several staged chunks on `stream`, with the destinations allocated on the CURRENT
        stream. Allocating inside a `torch.cuda.stream()` block binds the block to that stream's pool
        while every consumer downstream runs on the current one -- which is how this first shipped,
        and it aborted the process with an illegal access on the second exchange."""
        out = [torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in srcs]
        with torch.cuda.stream(stream):
            for dst, (buf, _, _) in zip(out, srcs):
                dst.view(-1).copy_(buf, non_blocking=True)
                self._keep(dst, stream)
        return out

    def _fetch(self, src, device):
        if self.mode != "host":
            return src.to(device)
        buf, shape, dtype = src
        recv = torch.empty(shape, dtype=dtype, device=device)
        recv.view(-1).copy_(buf, non_blocking=True)
        return recv

    def _finish(self, rank, stream, t0, nbytes):
        stream.synchronize()
        self.barrier.wait()
        self.slots[rank] = None
        self.seconds[rank] += time.perf_counter() - t0
        self.sent[rank] += nbytes

    def all_gather(self, rank, tensor, device):
        """Every rank's `tensor` (its token range), concatenated in rank order on this rank's device."""
        if self.chunks > 1:
            c, shapes = self._agree(rank, self._plan(tensor), (tuple(tensor.shape), tensor.dtype))
            if c:
                return self.all_gather_pipelined(rank, tensor, device, c, shapes)
        stream = torch.cuda.current_stream(device)
        stream.synchronize()
        t0 = time.perf_counter()
        self.slots[rank] = self._stage(rank, {"all": tensor}, stream)["all"]
        self.barrier.wait()
        parts = [tensor if q == rank else self._fetch(self.slots[q], device) for q in range(self.n)]
        self._finish(rank, stream, t0, tensor.numel() * tensor.element_size() * (self.n - 1))
        return torch.cat(parts, dim=0)

    def all_to_all(self, rank, sends, device):
        """sends[q] goes to rank q (sends[rank] stays local). Returns recv, where recv[q] is what rank q sent here."""
        if self.chunks > 1:
            c, shapes = self._agree(rank, min((self._plan(t) for t in sends), default=0),
                                    ([tuple(t.shape) for t in sends], sends[0].dtype))
            if c:
                return self.all_to_all_pipelined(rank, sends, device, c, shapes)
        stream = torch.cuda.current_stream(device)
        stream.synchronize()
        t0 = time.perf_counter()
        self.slots[rank] = self._stage(rank, {q: s for q, s in enumerate(sends) if q != rank}, stream)
        self.barrier.wait()
        recv = [sends[q] if q == rank else self._fetch(self.slots[q][rank], device) for q in range(self.n)]
        self._finish(rank, stream, t0, sum(s.numel() * s.element_size() for q, s in enumerate(sends) if q != rank))
        return recv

    def _plan(self, tensor):
        """How many chunks to split `tensor` into along dim 0, or 0 for the single-shot path."""
        if self.chunks <= 1 or tensor.dim() == 0:
            return 0
        nbytes = tensor.numel() * tensor.element_size()
        rows = tensor.shape[0]
        # pipelining a small transfer is a loss: the barriers and launches cost more than the overlap
        c = min(self.chunks, rows, max(1, nbytes // max(1, self.min_chunk_bytes)))
        return c if c > 1 else 0

    @staticmethod
    def _slices(rows, c):
        step = max(1, (rows + c - 1) // c)
        # exactly c spans, trailing ones empty if rows < c: the agreed count is global, this rank's
        # row count is not, and len(spans) must equal the number of per-chunk barriers.
        return [(min(i * step, rows), min((i + 1) * step, rows)) for i in range(c)]

    def _streams(self, device):
        key = str(device)
        st = _XFER_STREAMS.get(key)
        if st is None:
            with _XFER_LOCK:
                st = _XFER_STREAMS.get(key)
                if st is None:
                    st = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
                    _XFER_STREAMS[key] = st
        return st

    def _h2d(self, dsts, srcs, stream):
        """H2D of several staged chunks on `stream`, into slices of destinations allocated elsewhere.

        Nothing is allocated here. The first version allocated a fresh device tensor per chunk per
        peer INSIDE `with torch.cuda.stream(recv)`, which was wrong twice over: the caching allocator
        bound each block to the side stream's pool while every consumer ran on the current stream
        (an illegal access under a stream-ordered allocator), and at 4 chunks it multiplied the
        allocation count per step by four, on an allocator that maps VBAR pages.
        """
        with torch.cuda.stream(stream):
            for dst, buf in zip(dsts, srcs):
                dst.view(-1).copy_(buf, non_blocking=True)

    def all_gather_pipelined(self, rank, tensor, device, c, shapes):
        """all_gather with the two PCIe directions overlapped.

        Each card has its own full-duplex link to the host, so the D2H of chunk j+1 can run while the
        H2D of chunk j is in flight. The serial path pays D2H + H2D; this approaches max(D2H, H2D).

        The gathered result is allocated once, up front -- every rank's send shape came over the
        agreement barrier -- and each chunk lands directly in its slice. That removes both the
        per-chunk allocations and the concatenation the first version needed.
        """
        send, recv = self._streams(device)
        cur = torch.cuda.current_stream(device)
        cur.synchronize()
        t0 = time.perf_counter()
        rows = [sh[0][0] for sh in shapes]
        offs, acc = [], 0
        for r in rows:
            offs.append(acc)
            acc += r
        out = torch.empty((acc,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=device)
        self._keep(out, recv)
        self._keep(tensor, send)
        out[offs[rank]:offs[rank] + rows[rank]].copy_(tensor)
        # every rank slices with the same function over the same row count, so a receiver's spans for
        # peer q are exactly the spans q used when sending.
        spans = [self._slices(r, c) for r in rows]
        staged, evts = [None] * c, [None] * c

        def issue_send(j):
            a, b = spans[rank][j]
            piece = tensor[a:b]
            buf = self._pinned_buffer(rank, ("g", j), piece)
            with torch.cuda.stream(send):
                buf.copy_(piece.reshape(-1), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(send)
            staged[j], evts[j] = (buf, piece.shape, piece.dtype), ev

        issue_send(0)
        for j in range(c):
            if j + 1 < c:
                issue_send(j + 1)          # queued behind chunk j; overlaps the fetch below
            evts[j].synchronize()
            self._pipe[rank][j] = staged[j]
            self._bars[j].wait()
            dsts, srcs = [], []
            for q in range(self.n):
                if q == rank:
                    continue
                a, b = spans[q][j]
                dsts.append(out[offs[q] + a:offs[q] + b])
                srcs.append(self._pipe[q][j][0])
            self._h2d(dsts, srcs, recv)
        recv.synchronize()
        self.barrier.wait()                # every sender's pinned buffers stay alive until here
        for j in range(c):
            self._pipe[rank][j] = None
        self.seconds[rank] += time.perf_counter() - t0
        self.sent[rank] += tensor.numel() * tensor.element_size() * (self.n - 1)
        return out

    def all_to_all_pipelined(self, rank, sends, device, c, shapes):
        """all_to_all with the two PCIe directions overlapped; see all_gather_pipelined."""
        send, recv_s = self._streams(device)
        cur = torch.cuda.current_stream(device)
        cur.synchronize()
        t0 = time.perf_counter()
        # shapes[q][0][rank] is what rank q sends HERE, so each destination is allocated once.
        out = []
        for q in range(self.n):
            if q == rank:
                out.append(sends[rank])
            else:
                t = torch.empty(shapes[q][0][rank], dtype=shapes[q][1], device=device)
                self._keep(t, recv_s)
                out.append(t)
        # send spans come from this rank's own tensors, receive spans from what the peer will send;
        # both sides call _slices on the same row count, so they agree.
        send_spans = [self._slices(t.shape[0], c) for t in sends]
        recv_spans = [self._slices(shapes[q][0][rank][0], c) for q in range(self.n)]
        staged, evts = [None] * c, [None] * c
        for q, t in enumerate(sends):
            if q != rank:
                self._keep(t, send)

        def issue_send(j):
            out_j, pieces = {}, []
            for q, t in enumerate(sends):
                if q == rank:
                    continue
                a, b = send_spans[q][j]
                piece = t[a:b]
                buf = self._pinned_buffer(rank, ("a", q, j), piece)
                pieces.append((buf, piece))
                out_j[q] = (buf, piece.shape, piece.dtype)
            with torch.cuda.stream(send):
                for buf, piece in pieces:
                    buf.copy_(piece.reshape(-1), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(send)
            staged[j], evts[j] = out_j, ev

        issue_send(0)
        for j in range(c):
            if j + 1 < c:
                issue_send(j + 1)
            evts[j].synchronize()
            self._pipe[rank][j] = staged[j]
            self._bars[j].wait()
            dsts, srcs = [], []
            for q in range(self.n):
                if q == rank:
                    continue
                a, b = recv_spans[q][j]
                dsts.append(out[q][a:b])
                srcs.append(self._pipe[q][j][rank][0])
            self._h2d(dsts, srcs, recv_s)
        recv_s.synchronize()
        self.barrier.wait()
        for j in range(c):
            self._pipe[rank][j] = None
        self.seconds[rank] += time.perf_counter() - t0
        self.sent[rank] += sum(t.numel() * t.element_size() for q, t in enumerate(sends) if q != rank)
        return out

    def abort(self):
        self.barrier.abort()
        for b in self._bars:
            b.abort()


def _partition(total, shares, what):
    """Bounds [0, ..., total] of a contiguous split into len(shares) parts proportional to shares, each at least 1.

    Largest remainder; ties go to the later rank, so an odd count with equal shares splits like total // 2 | rest
    (the 2-GPU split's original token boundary)."""
    n = len(shares)
    if total < n:
        raise MultiStreamError(f"cannot split {total} {what} across {n} GPU ranks")
    s = float(sum(shares))
    raw = [total * x / s for x in shares]
    sizes = [max(1, math.floor(r)) for r in raw]
    while sum(sizes) < total:
        i = max(range(n), key=lambda k: (raw[k] - sizes[k], k))
        sizes[i] += 1
    while sum(sizes) > total:
        i = max((k for k in range(n) if sizes[k] > 1), key=lambda k: (sizes[k] - raw[k], k))
        sizes[i] -= 1
    bounds = [0]
    for z in sizes:
        bounds.append(bounds[-1] + z)
    return bounds


_HEAD_ROWS = {}


def _head_rows(g0, g1, heads, hd, device):
    """Output rows of qkv_proj that belong to attention heads [g0, g1): the q, k and v blocks of those heads."""
    key = (g0, g1, heads, hd, device)
    idx = _HEAD_ROWS.get(key)
    if idx is None:
        hs = torch.arange(g0, g1)
        cols = (hs[:, None] * hd + torch.arange(hd)[None]).reshape(-1)
        inner = heads * hd
        idx = torch.cat([cols, cols + inner, cols + 2 * inner]).to(device)
        _HEAD_ROWS[key] = idx
    return idx


def _gate_rows(g0, g1, hd, device):
    """Output rows of a single per-head projection (to_gate_compress) for heads [g0, g1).

    Unlike qkv_proj -- three stacked heads*hd blocks, hence _head_rows' strided index -- the gate has
    one block, so its rows are the contiguous range [g0*hd, g1*hd). Verified against the FastH3
    checkpoint: to_gate_compress.weight is I8 [7168, 5376] = (56 heads * 128) rows, head-major."""
    key = ("gate", g0, g1, hd, device)
    idx = _HEAD_ROWS.get(key)
    if idx is None:
        idx = torch.arange(g0 * hd, g1 * hd, device=device)
        _HEAD_ROWS[key] = idx
    return idx


def _rows_linear_fn(lin, like, idx):
    """Resolve `lin`'s effective weight ONCE, slice output rows `idx`, return a callable (x) -> x @ W'.

    Resolving is not free: cast_bias_weight walks the LoRA/quant path and index_select materialises a
    copy of the slice. The sparse producer calls its projections once per PRODUCER_CHUNK, so doing
    this inside the chunk loop repeated the work 12x per block per rank per step -- measured at
    ~15 s/step for the VSA gate with the DiT weight cache off (32.4 s vs 16.9 s warm, 2026-09-16).
    Hoisting it to once per block removes that entirely. `like` only supplies dtype and device, which
    every chunk of a block shares.

    `lin` must come from the rank's shadow (ms_cast.make_shadow tags those with _multistream_rank).
    Casting an unshadowed module here touches vbar state and memory mapped to another device, which
    segfaults instead of raising -- so check, cheaply, rather than trust."""
    if not hasattr(lin, "_multistream_rank"):
        raise MultiStreamError(
            f"{type(lin).__name__} was not taken from this rank's shadow block; casting it here would "
            "read weights mapped to another device. This is a bug in the caller, not in the model.")
    weight, bias, _ = comfy.ops.cast_bias_weight(lin, like, offloadable=True, compute_dtype=like.dtype,
                                                 want_requant=len(lin.weight_function) == 0)
    b = bias.index_select(0, idx) if bias is not None else None
    if (isinstance(weight, QuantizedTensor) and weight._layout_cls == "TensorWiseINT8Layout"
            and not getattr(weight._params, "transposed", False) and _int8_kernel_available()):
        qd, sc = comfy.ops.TensorWiseINT8Layout.get_plain_tensors(weight)
        qd = qd.index_select(0, idx).contiguous()
        sc = sc.index_select(0, idx).contiguous()
        code = _dtype_code(like.dtype)
        convrot = bool(getattr(weight._params, "convrot", False))
        groupsize = int(getattr(weight._params, "convrot_groupsize", 256))

        def int8_apply(x):
            return torch.ops.comfy_kitchen.int8_linear(x.contiguous(), qd, sc, b, code, convrot, groupsize)
        return int8_apply
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    w = weight.index_select(0, idx).to(like.dtype)

    def apply(x):
        return torch.nn.functional.linear(x, w, b)
    return apply


_INT8_FALLBACK_WARNED = []


def _int8_kernel_available():
    """True when comfy-kitchen still provides both the private dtype helper and the int8_linear op."""
    ok = _dtype_code is not None and hasattr(torch.ops.comfy_kitchen, "int8_linear")
    if not ok and not _INT8_FALLBACK_WARNED:
        _INT8_FALLBACK_WARNED.append(True)
        log.warning("[MultiStream] comfy-kitchen's int8_linear kernel or its dtype helper is unavailable in this "
                    "version; sliced int8 projections use dequantize + linear (slower, and output may differ "
                    "slightly from a single GPU)")
    return ok


def _qkv_group_fn(lin, like, g0, g1, heads, hd):
    """qkv_proj restricted to attention heads [g0, g1), resolved once."""
    return _rows_linear_fn(lin, like, _head_rows(g0, g1, heads, hd, like.device))


def _gate_group_fn(lin, like, g0, g1, hd):
    """to_gate_compress restricted to attention heads [g0, g1), resolved once (VSA coarse branch)."""
    return _rows_linear_fn(lin, like, _gate_rows(g0, g1, hd, like.device))


def _qkv_group(lin, h_full, g0, g1, heads, hd):
    """One-shot form for the dense path, which projects the block's tokens exactly once."""
    return _qkv_group_fn(lin, h_full, g0, g1, heads, hd)(h_full)


def _rank_block(rank, blk, x, t_emb, segs, rope, to, ex, bounds, groups, sparse_attention=None, vsa=None):
    """One block on one rank. bounds: token range bounds per rank; groups: attention head group bounds per rank."""
    dev = x.device
    heads, hd = blk.attn.heads, blk.attn.head_dim
    g0, g1 = groups[rank], groups[rank + 1]
    ph = g1 - g0
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(t_emb)
    h = H._mod_scale_shift(blk.norm1(x), shift_msa, scale_msa, segs)
    h_full = ex.all_gather(rank, h, dev)
    del h
    qw = to_device(blk.attn.q_norm.weight, dev)
    kw = to_device(blk.attn.k_norm.weight, dev)
    if sparse_attention is not None:
        # block-sparse attention over this rank's head group; qkv, rope and attention are fused in the
        # kernel and it returns the pre-out_proj output, which is exactly what the all-to-all wants.
        # BOTH projections must come off `blk`, the rank's shadow -- not off the original block the
        # sparse node closed over. Casting the unshadowed to_gate_compress from a rank thread reads
        # weights under a cuMemCreate mapping owned by the other device and segfaults (2026-09-16).
        # Both are resolved HERE, once per block, not inside the producer's chunk loop.
        shadow_gate = getattr(blk.attn, "to_gate_compress", None)
        a_all = ms_sparse.rank_attention(
            sparse_attention, h_full, rope, to,
            _qkv_group_fn(blk.attn.qkv_proj, h_full, g0, g1, heads, hd), qw, kw, g0, g1, hd,
            vsa=vsa,
            gate_group=(None if shadow_gate is None
                        else _gate_group_fn(shadow_gate, h_full, g0, g1, hd)))
        del h_full
        recv = ex.all_to_all(rank, [a_all[bounds[p]:bounds[p + 1]] for p in range(ex.n)], dev)
        del a_all
        attn_in = torch.cat(recv, dim=-1)
        del recv
        x = H._mod_gate(x, gate_msa, blk.attn.out_proj(attn_in), segs)
        del attn_in
        h = H._mod_scale_shift(blk.norm2(x), shift_mlp, scale_mlp, segs)
        return H._mod_gate(x, gate_mlp, blk.mlp(h), segs)
    qkv = _qkv_group(blk.attn.qkv_proj, h_full, g0, g1, heads, hd)
    del h_full
    S = qkv.shape[0]
    q, k, v = qkv.split(ph * hd, dim=-1)
    v = v.view(S, ph, hd)
    q = q.view(1, S, ph, hd)
    k = k.view(1, S, ph, hd)
    comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, qw, kw, epsilon=blk.attn.q_norm.eps,
                                            rot_dim=rope.shape[-3] * 2)
    q = A.AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
    k = A.AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
    v = A.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = A.optimized_attention(q, k, v, ph, mask=None, skip_reshape=True, transformer_options=to)
    del q, k, v, qkv
    a_all = out.squeeze(0)
    recv = ex.all_to_all(rank, [a_all[bounds[p]:bounds[p + 1]] for p in range(ex.n)], dev)
    del a_all, out
    attn_in = torch.cat(recv, dim=-1)   # head groups in rank order = the model's head order
    del recv
    x = H._mod_gate(x, gate_msa, blk.attn.out_proj(attn_in), segs)
    del attn_in
    h = H._mod_scale_shift(blk.norm2(x), shift_mlp, scale_mlp, segs)
    return H._mod_gate(x, gate_mlp, blk.mlp(h), segs)


def _capture_overrides(user_patches, n_blocks, args, sparse=False, vsa=False):
    """Run wrapper-style block patches against a capturing original_block.

    Returns (overrides, sparse_attentions). A patch may only adjust transformer_options -- except that
    with `sparse` on, a patch that installs an `attention` callable (ComfyUI's Model Sparse Attention
    node) has that callable captured and run per rank instead; see multistream/sparse.py for why the
    head split is exact rather than an approximation.
    """
    overrides, sparse_at = [], []
    for i in range(n_blocks):
        patch = user_patches.get(("double_block", i))
        if patch is None:
            overrides.append(_MISSING)
            sparse_at.append(None)
            continue
        seen = {}

        def original_block(a, _seen=seen):
            _seen["to"] = dict(a["transformer_options"])
            _seen["attention"] = a.get("attention")
            return {"img": a["img"]}

        img = args["img"]
        out = patch({"img": img, "t_emb": args["t_emb"], "mod_segments": args["mod_segments"],
                     "rope_freqs": args["rope_freqs"], "layout": args.get("layout"),
                     "transformer_options": args["transformer_options"]},
                    {"original_block": original_block})
        attention = seen.get("attention")
        # the patch must still only have read the block and adjusted transformer_options, sparse or not
        if "to" not in seen or out.get("img") is not img or (attention is not None and not sparse):
            hint = ("" if sparse or attention is None else
                    " If this is ComfyUI's Model Sparse Attention node, turn on `sparse_attention` on the "
                    "H3 MultiStream node to run it per rank.")
            raise MultiStreamError(
                f"block {i} carries a patch that replaces the block computation; H3 MultiStream only "
                f"supports patches that adjust transformer_options (e.g. per-block attention backend).{hint}")
        if attention is not None:
            ms_sparse.capture(attention)   # fail here, with a clear message, not inside a rank thread
            if ms_sparse.wants_vsa(attention) and not vsa:
                raise MultiStreamError(
                    f"block {i}: the Model Sparse Attention node is set to 'vsa'. Turn on `sparse_vsa` "
                    "on the H3 MultiStream node to run it split, or pick 'sol-attn'/'sla' instead.")
        sparse_at.append(attention)
        overrides.append(seen["to"].get(OVERRIDE, _NONE))
    return overrides, sparse_at


def _loaded_entries_for(dm):
    """ComfyUI's LoadedModel entries that own this diffusion model, for free_memory(keep_loaded=...)."""
    keep = []
    for lm in list(getattr(comfy.model_management, "current_loaded_models", [])):
        base = getattr(getattr(lm, "model", None), "model", None)
        if base is not None and getattr(base, "diffusion_model", None) is dm:
            keep.append(lm)
    return keep


DYNAMIC_VRAM_MODES = ("keep", "off for this model")


def non_dynamic_delegate(model):
    """ComfyUI's per-model DynamicVRAM opt-out. Returns (model, what happened).

    `ModelPatcherDynamic.get_non_dynamic_delegate()` clones the patcher with disable_dynamic=True,
    giving a plain legacy ModelPatcher for THIS model while every other model in the process keeps
    DynamicVRAM. ComfyUI uses it itself (comfy/samplers.py, when conditioning carries hooks, because
    hooks are not implemented in ModelPatcherDynamic), and custom loaders that subclass the plain
    ModelPatcher -- GGUF ones, for instance -- already coexist with dynamic models, so a mixed regime
    is ordinary rather than novel.

    Why the split wants it: aimdo's per-call regime is what makes device tensors allocated by this
    pack unusable on the next sampler step (see docs/vram-block-residency.md -- an illegal access that
    aborts the process). Under the legacy patcher they behave like ordinary torch allocations.

    NOT free: the clone re-invokes the loader from cached_patcher_init, building a second model
    instance. ComfyUI memoises it in non_dynamic_delegate_model, so it happens once per process.

    Never raises. A model that cannot be delegated is returned unchanged with a reason, because
    running dynamic is always a valid fallback."""
    if not callable(getattr(model, "is_dynamic", None)):
        return model, "no is_dynamic(): not a DynamicVRAM build"
    if not model.is_dynamic():
        return model, "already non-dynamic"
    fn = getattr(model, "get_non_dynamic_delegate", None)
    if not callable(fn):
        log.warning("[MultiStream] this ComfyUI has no get_non_dynamic_delegate(); staying dynamic")
        return model, "get_non_dynamic_delegate() unavailable"
    try:
        delegate = fn()
    except Exception as e:                     # cached_patcher_init missing, loader failure, ...
        log.warning("[MultiStream] could not take the non-dynamic delegate (%s: %s); staying dynamic",
                    type(e).__name__, e)
        return model, f"failed: {type(e).__name__}"
    if delegate is None or (callable(getattr(delegate, "is_dynamic", None)) and delegate.is_dynamic()):
        log.warning("[MultiStream] the delegate came back dynamic; staying on the original")
        return model, "delegate still dynamic"
    return delegate, "delegated"


def free_rank_devices(devices, primary, dm=None, need_bytes=None):
    """Ask ComfyUI to evict its models from the rank GPUs that are NOT the model's own device.

    ComfyUI only frees a device when something is loaded onto it THROUGH its loader. The split's
    non-primary ranks allocate straight through torch, so that never happens: a text encoder or VAE
    the user assigned to such a card is never asked to leave, however tight the card gets. That is
    not hypothetical -- on 2026-09-16 a text encoder pinned ~8 GiB on the second GPU for a whole run
    and the VAE split worker (a separate process, needing 5.18 GiB) then OOM'd there.

    The primary is included too, but only when the DiT's own LoadedModel entries can be identified
    and passed as keep_loaded -- otherwise freeing there could evict the model we are about to run,
    so it is skipped. This matters whenever the text encoder shares a card with the DiT: ComfyUI's
    own pressure at DiT-load time is often not enough to move it, and it is then dead weight for the
    whole sampling run (measured 2026-09-16: 4 blocks fit on that card against 12 on the other).

    `need_bytes` defaults to the WHOLE CARD, and that is deliberate. free_memory() unloads only until
    the requested amount is free and then stops, so a modest request quietly does nothing: asking for
    4 GiB on a card that already had 3.74 GiB free evicted 0.31 GiB and left an 8 GiB text encoder
    untouched (measured 2026-09-16). Nothing of ComfyUI's is needed on a secondary rank while the
    split runs, and whatever is unloaded reloads from host when a later node wants it.
    """
    keep = _loaded_entries_for(dm) if dm is not None else []
    freed = []
    for dev in dict.fromkeys(devices):
        if dev.type != "cuda":
            continue
        if dev == primary and not keep:
            log.debug("[MultiStream] not freeing the primary %s: the DiT's loaded entry was not found", dev)
            continue
        before = comfy.model_management.get_free_memory(dev)
        try:
            want = need_bytes
            if want is None:
                want = torch.cuda.get_device_properties(dev).total_memory
            comfy.model_management.free_memory(want, dev, keep_loaded=keep)
        except Exception:
            log.exception("[MultiStream] could not free %s; continuing", dev)
            continue
        after = comfy.model_management.get_free_memory(dev)
        if after - before > 64 * 2**20:
            freed.append(f"{dev}{' (primary)' if dev == primary else ''} +{(after - before) / 2**30:.2f} GiB")
    if freed:
        log.info("[MultiStream] evicted ComfyUI models from rank GPU(s): %s", ", ".join(freed))
    return freed


_SIDE_STREAMS = {}
_SIDE_STREAMS_LOCK = threading.Lock()


def _side_stream(device):
    with _SIDE_STREAMS_LOCK:
        s = _SIDE_STREAMS.get(device)
        if s is None:
            s = torch.cuda.Stream(device=device)
            _SIDE_STREAMS[device] = s
        return s


def stream_status():
    with _SIDE_STREAMS_LOCK:
        return [str(d) for d in _SIDE_STREAMS]


def release_streams(reason="released on request"):
    """Drop the cached prefetch side streams; the next split step creates them again.

    A stream keeps its device's CUDA context referenced for the life of the process, so on a secondary GPU it
    pins memory nothing here uses between video jobs. Synchronizing first is what makes this safe: a stream
    must not be destroyed with work still queued on it. Rank threads hold their own reference for the duration
    of a step, so dropping the dict entry can only take effect once they are done with it."""
    released, kept = [], []
    with _SIDE_STREAMS_LOCK:
        for device in list(_SIDE_STREAMS):
            try:
                torch.cuda.synchronize(device)
            except Exception:
                log.exception("[MultiStream] could not synchronize %s; its side stream is left in place", device)
                kept.append(str(device))
                continue
            _SIDE_STREAMS.pop(device, None)
            released.append(str(device))
    msg = (f"released prefetch side stream(s) on {', '.join(released)}; the next split step creates them again"
           if released else "no prefetch side streams were held")
    if kept:
        msg += f" (kept on {', '.join(kept)}: could not synchronize)"
    log.info("[MultiStream] %s (%s)", msg, reason)
    return {"released": len(released), "kept": len(kept), "message": msg}


def run_split_stack(dm, args, user_patches, devices, shares=None, exchange="host", cache_key=None,
                    cache_reserve_gib=0.0, prefetch=None, sparse=False, vsa=False, vram_blocks=False,
                    vram_reserve_gib=None, exchange_chunks=0):
    """Run all blocks split across `devices` (rank 0 = the model's device). Ranks may share a GPU (logical-rank tests)."""
    h = args["img"]
    t_emb = args["t_emb"]
    segs = args["mod_segments"]
    rope = args["rope_freqs"]
    base_to = args["transformer_options"]
    dev0 = h.device
    devs = list(devices)
    n = len(devs)
    if devs[0] != dev0:
        raise MultiStreamError(f"rank 0 must be the model's device {dev0}, got {devs[0]}")
    shares = list(shares) if shares else [1.0] * n
    n_blocks = len(dm.blocks)
    heads = dm.blocks[0].attn.heads

    overrides, sparse_at = _capture_overrides(user_patches, n_blocks, args, sparse, vsa)
    ms_cast.install_cast_hook()
    # VRAM is the FIRST tier of one cache: blocks 0..vram_upto-1 live on the cards, the rest are
    # pinned in host RAM by ms_cache. A block is only skippable on the host when EVERY rank has it
    # resident -- rank r reads it from r's own device, so one rank short means the host still needs it.
    vram_upto = 0
    if vram_blocks:
        per_block = ms_cast.staged_size(dm.blocks[0]) if n_blocks else 0
        fits = []
        for d in dict.fromkeys(devices):
            budget = ms_vram.plan(d, safety_gib=vram_reserve_gib)
            fits.append(int(budget // per_block) if per_block else 0)
        vram_upto = max(0, min(min(fits) if fits else 0, n_blocks))
        log.info("[VRAM cache] tiering: %s/block staged, blocks 0-%d on the cards, %d-%d from host RAM",
                 gib(per_block), vram_upto - 1, vram_upto, n_blocks - 1)

    cache = ms_cache.acquire(cache_key, cache_reserve_gib, "dit") if cache_key is not None else None
    if cache is not None:
        cache.start_prefill(dm.blocks, skip_below=vram_upto)
    cache_blocks_before = len(cache.blocks) if cache is not None else 0

    S = h.shape[0]
    bounds = _partition(S, shares, "tokens")
    groups = _partition(heads, shares, "attention heads")
    if prefetch is None:
        prefetch = PREFETCH
    prefetch = prefetch and cache is not None   # prefetch stages from the pinned cache

    def owned(t, dev):
        # memory the rank owns (blocks update x in place): a clone on the model's GPU, a host-staged copy elsewhere
        return t.clone() if t.device == dev else to_device(t, dev)

    xs = [owned(h[bounds[r]:bounds[r + 1]], devs[r]) for r in range(n)]
    t_embs = [owned(t_emb, devs[r]) for r in range(n)]
    ropes = [owned(rope, devs[r]) for r in range(n)]
    seg_parts = [_local_segments(segs, bounds[r], bounds[r + 1], devs[r]) for r in range(n)]
    base_override = base_to.get(OVERRIDE, _NONE)
    tos = [dict(base_to) for _ in range(n)]

    if vram_blocks:
        # idempotent per run: budgets against the previous step's measured peak once the ranks are free
        for d in dict.fromkeys(devices):
            ms_vram.plan(d, safety_gib=vram_reserve_gib)

    inference = torch.is_inference_mode_enabled()
    grad = torch.is_grad_enabled()
    ex = _Collective("p2p" if UNSAFE else exchange, n, devs, chunks=exchange_chunks)
    outs = [None] * n
    errors = []
    unique = list(dict.fromkeys(devs))
    t0 = time.perf_counter()
    for d in unique:
        try:
            torch.cuda.reset_peak_memory_stats(d)
        except Exception:
            pass

    def worker(r):
        dev = devs[r]
        try:
            with torch.cuda.device(dev), torch.inference_mode(inference), torch.set_grad_enabled(grad):
                cur = torch.cuda.current_stream(dev)
                side = _side_stream(dev) if prefetch else None
                pending = {}    # block -> (staged weights, copy-done event), copied on the side stream
                held = None     # staged weights of the previous block, see below
                x = xs[r]
                to = tos[r]
                # VSA's cube plan and padded rope depend on (layout, device) only, never on the block:
                # build them ONCE here. Doing it per block would rebuild an O(seq_len) plan 50x per
                # rank per step and race on upstream's single-slot rope cache.
                vsa_ctx = None
                first_sparse = next((a for a in sparse_at if a is not None), None)
                if first_sparse is not None:
                    vsa_ctx = ms_sparse.vsa_context(first_sparse, ropes[r], to, dev, vsa)
                for i in range(n_blocks):
                    if r == 0:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                    ov = overrides[i]
                    if ov is _MISSING:
                        ov = base_override
                    if ov is _NONE:
                        to.pop(OVERRIDE, None)
                    else:
                        to[OVERRIDE] = ov
                    to["block_index"] = i
                    tier_vram = vram_blocks and i < vram_upto
                    staged = ms_vram.get(dev, i) if tier_vram else None
                    resident = staged is not None
                    # host tier only: a block the cards own must not also be pinned in RAM
                    entry = None if tier_vram else (cache.get_block(i, dm.blocks[i]) if cache is not None else None)
                    if resident and i in pending:
                        pending.pop(i)          # prefetched needlessly; drop it
                    elif i in pending:
                        staged, done = pending.pop(i)
                        cur.wait_event(done)
                    if tier_vram and not resident:
                        fresh = ms_cast.stage_block_resident(dm.blocks[i], dev)
                        if fresh and ms_vram.offer(dev, i, fresh, ms_vram.staged_bytes(fresh)):
                            staged, resident = fresh, True
                    if side is not None and i + 1 < n_blocks and not (vram_blocks and i + 1 < vram_upto):
                        # copy the next block's weights while this block computes (H2D and compute overlap)
                        nxt = cache.get_block(i + 1, dm.blocks[i + 1])
                        if nxt is not None:
                            with torch.cuda.stream(side):
                                ahead = ms_cast.stage_block(dm.blocks[i + 1], nxt, dev)
                                done = torch.cuda.Event()
                                done.record(side)
                            pending[i + 1] = (ahead, done)
                    blk = ms_cast.make_shadow(dm.blocks[i], r, entry, staged=staged)
                    x = _rank_block(r, blk, x, t_embs[r], seg_parts[r], ropes[r], to, ex, bounds, groups,
                                    sparse_at[i], vsa_ctx)
                    del blk
                    # Lifetime of side-stream memory: this block's MLP kernels may still be queued when _rank_block
                    # returns, so its staged weights stay referenced until the next block has run. The next block's
                    # first exchange synchronizes this device's current stream (_Collective, both modes; ranks that
                    # share a GPU share that stream), after which every kernel of this block is done and the side
                    # stream may reuse the memory.
                    held = None if resident else staged   # resident staging outlives the step
                del held
                cur.synchronize()
                outs[r] = x
        except BaseException as e:
            errors.append((r, e))
            ex.abort()

    threads = [threading.Thread(target=worker, args=(r,), name=f"h3ms-rank{r}") for r in range(n)]
    with cache.in_use() if cache is not None else contextlib.nullcontext():
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for d in unique:
            # the prefetch's side-stream copies out of pinned cache memory may still be queued after the
            # rank threads have joined; they must land before in_use() is released, or a concurrent
            # release() could cudaHostUnregister the source underneath them.
            try:
                torch.cuda.synchronize(d)
            except Exception:
                log.exception("[MultiStream] could not synchronize %s after the split step", d)
    if errors:
        real = [(r, e) for r, e in errors if not isinstance(e, threading.BrokenBarrierError)]
        rank, err = (real or errors)[0]
        if isinstance(err, comfy.model_management.InterruptProcessingException):
            log.info("[MultiStream] step interrupted")
        else:
            log.error("[MultiStream] split step failed on rank %d (%s): %s: %s | %s", rank, devs[rank],
                      type(err).__name__, err, " | ".join(vram(d) for d in unique))
        raise err

    result = torch.cat([outs[r] if devs[r] == dev0 else to_device(outs[r], dev0) for r in range(n)], dim=0)
    wall = time.perf_counter() - t0
    if cache is None:
        cstat = "off"
    elif cache.disabled:
        cstat = f"DISABLED (RAM reserve {cache.reserve_gib:.1f} GiB)"
    elif cache_blocks_before < n_blocks:
        cstat = f"filling {cache_blocks_before}->{len(cache.blocks)}/{n_blocks} blocks, {gib(cache.bytes)}"
    else:
        cstat = f"warm, {gib(cache.bytes)}"
    for d in unique:
        try:
            ms_vram.note_peak(d, torch.cuda.max_memory_allocated(d))
        except Exception:
            pass
    if cache is not None:
        cstat += f", prefetch {'on' if prefetch else 'off'}"
    if vram_blocks:
        res = ms_vram.stats()
        cstat += ((f", tiered: {vram_upto} block(s) on each card"
                   f" ({sum(r['GiB'] for r in res):.1f} GiB), {n_blocks - vram_upto} from host")
                  if res else ", vram tier empty (nothing fit)")
    n_sparse = sum(a is not None for a in sparse_at)
    if sparse:
        method = ""
        first_sparse = next((a for a in sparse_at if a is not None), None)
        if first_sparse is not None:
            method = " (vsa)" if ms_sparse.wants_vsa(first_sparse) else " (sol-attn/sla)"
        cstat += (f", sparse attention {n_sparse}/{n_blocks} blocks{method}" if n_sparse
                  else ", sparse attention on but 0 blocks eligible")
    sigma = base_to.get("sigmas")
    sigma_s = f"sigma {float(sigma.flatten()[0]):.4f} " if isinstance(sigma, torch.Tensor) and sigma.numel() else ""
    # step split peak: PyTorch allocations of this step only (reset at step start), constant for a fixed shape;
    # GPU used: driver level, includes staged weights, other models and the VAE worker
    # the COLLECTIVE's effective chunk count, not the node's requested one: p2p forces chunks to 0
    # (it hands device tensors over untouched, so there are no two directions to overlap). Logging the
    # request instead printed "p2p x8" for an exchange that was running unchunked, which is exactly
    # the wrong thing to show someone benchmarking p2p against host.
    eff_chunks = getattr(ex, "chunks", 0)
    ex_label = exchange if not eff_chunks else f"{exchange} x{eff_chunks}"
    log.info("[MultiStream] %sstep: %d tokens x %d blocks on %s, %.1fs (exchange %s: %s, %s moved), "
             "weight cache %s, step split peak %s, GPU used %s",
             sigma_s, S, n_blocks, "+".join(str(d) for d in devs), wall, ex_label,
             "/".join(f"{s:.1f}s" for s in ex.seconds), gib(sum(ex.sent)), cstat,
             " / ".join(peak(d) for d in unique), " + ".join(used(d) for d in unique))
    LAST_STEP.clear()
    LAST_STEP.update({"ranks": [str(d) for d in devs], "heads_per_rank": [groups[r + 1] - groups[r] for r in range(n)],
                      "tokens": S, "seconds": round(wall, 2), "exchange": ex.mode,
                      "exchange_seconds_per_rank": [round(s, 2) for s in ex.seconds],
                      "moved_gib": round(sum(ex.sent) / 2**30, 2), "prefetch": bool(prefetch),
                      "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log.debug("[MultiStream] %d ranks: heads per rank %s, tokens per rank %s, overrides captured %d/%d", n,
              [groups[r + 1] - groups[r] for r in range(n)], [bounds[r + 1] - bounds[r] for r in range(n)],
              sum(o is not _MISSING for o in overrides), n_blocks)
    return result


def make_wrapper(second_gpu_index=None, exchange="host", cache_key=None, cache_reserve_gib=0.0, prefetch=None,
                 gpu_set=None, rank_devices=None, rank_shares=None, sparse=False, vsa=False,
                 vram_blocks=False, vram_reserve_gib=None, exchange_chunks=0):
    """DIFFUSION_MODEL wrapper. GPUs: gpu_set (H3 MS GPU Set) if given, else the legacy second_gpu_index. The rank plan
    is resolved against the model's device at first use and kept for this wrapper's lifetime.

    rank_devices / rank_shares (tests only): an explicit rank -> device list that may repeat a GPU, to run several
    logical ranks on one card; rank 0 must be the model's device."""
    gpu_set = gpu_set if gpu_set is not None else ms_gpus.GPUSet.legacy(second_gpu_index)
    warned = []
    calls = [0]
    plans = {}

    def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
        dm = executor.class_obj
        dev0 = x[0].device
        if not isinstance(dm, H.MiniMaxH3Model) or dev0.type != "cuda":
            if not warned:
                log.warning("[MultiStream] running UNSPLIT: model %s on %s (needs MiniMax H3 on CUDA)",
                            type(dm).__name__, dev0)
                warned.append(True)
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        plan = plans.get(dev0)
        if plan is None:
            if rank_devices is not None:
                devices = [d if isinstance(d, torch.device) else torch.device("cuda", int(d)) for d in rank_devices]
                plan = ms_gpus.RankPlan(devices, [float(s) for s in rank_shares] if rank_shares else [1.0] * len(devices),
                                        ["explicit rank map (test)"])
                log.info("[GPUs] dit: %s", plan.describe())
            else:
                full = ms_gpus.resolve(gpu_set, dev0)
                ms_gpus.log_plan("dit", gpu_set, full)
                plan = full.limited(ms_gpus.DIT_MAX_RANKS)
                if plan is not full:
                    log.warning("[MultiStream] %s", plan.notes[-1])
            plans[dev0] = plan
        if plan.n < 2:
            if not warned:
                log.info("[MultiStream] running unsplit on 1 GPU (%s): caches still apply", dev0)
                warned.append(True)
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        heads = dm.blocks[0].attn.heads
        if plan.n > heads:
            raise MultiStreamError(f"{plan.n} GPU ranks but the model has only {heads} attention heads")

        to = dict(transformer_options)
        patches_replace = dict(to.get("patches_replace", {}))
        dit = dict(patches_replace.get("dit", {}))
        user = {}
        for key in list(dit.keys()):
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "double_block":
                user[key] = dit.pop(key)
        if dit:
            log.error("[MultiStream] refusing unsupported dit patches: %s", sorted(map(str, dit.keys())))
            raise MultiStreamError(f"unsupported dit patches: {sorted(map(str, dit.keys()))}")
        calls[0] += 1
        if calls[0] == 1:
            # once per sampling run, before the first split step allocates anything
            free_rank_devices(plan.devices, plan.devices[0], dm)   # rank 0 is the model's own device
            groups = _partition(heads, plan.shares, "attention heads")
            log.info("[MultiStream] active: %d ranks %s, exchange %s, %d blocks, attention heads per rank %s, "
                     "%d per-block patch(es) captured, sparse attention %s (vsa %s), weight cache %s | %s", plan.n,
                     "+".join(str(d) for d in plan.devices), exchange, len(dm.blocks),
                     "/".join(str(groups[r + 1] - groups[r]) for r in range(plan.n)), len(user),
                     "on" if sparse else "off", "on" if vsa else "off",
                     os.path.basename(cache_key[0]) if cache_key else "off",
                     " | ".join(vram(d) for d in dict.fromkeys(plan.devices)))

        def first_block(a, extra):
            pause = contextlib.nullcontext() if UNSAFE else comfy.model_prefetch.pause_malloc_graph(sync=True)
            with pause, aimdo_logging_silenced():
                return {"img": run_split_stack(dm, a, user, plan.devices, plan.shares, exchange, cache_key,
                                               cache_reserve_gib, prefetch, sparse, vsa, vram_blocks,
                                               vram_reserve_gib, exchange_chunks)}

        def identity_block(a, extra):
            return {"img": a["img"]}

        for i in range(len(dm.blocks)):
            dit[("double_block", i)] = first_block if i == 0 else identity_block
        patches_replace["dit"] = dit
        to["patches_replace"] = patches_replace
        to["prefetch_dynamic_vbars"] = False
        return executor(x, timestep, context, to, *args, **kwargs)

    return wrapper


def cache_key_for(model_patcher):
    """Base cache key (checkpoint realpath, model options) from the loader's reload factory, or None."""
    init = getattr(model_patcher, "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    path = init[1][0]
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    opts = init[1][1] if len(init[1]) > 1 and isinstance(init[1][1], dict) else {}
    return (os.path.realpath(path), repr(sorted((str(k), str(v)) for k, v in opts.items())))


def add_to_transformer_options(transformer_options, second_gpu_index=None, exchange="host", cache_key=None,
                               prefetch=None, gpu_set=None, rank_devices=None, rank_shares=None):
    comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                                                 make_wrapper(second_gpu_index, exchange, cache_key, prefetch=prefetch,
                                                              gpu_set=gpu_set, rank_devices=rank_devices,
                                                              rank_shares=rank_shares),
                                                 transformer_options)
    return transformer_options
