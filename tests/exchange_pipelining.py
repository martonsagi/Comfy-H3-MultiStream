# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""The pipelined exchange must move the same bytes, and must actually overlap the two directions.

    python tests/exchange_pipelining.py

CPU only: torch.cuda's streams and events are stubbed, so the ordering assertions test the SCHEDULE
this code issues rather than what a GPU does with it. The exchange is ~36% of a sparse step and both
PCIe directions are idle half the time, which is what the pipeline is for -- but a version that
silently ran serially would still be correct, so ordering is asserted explicitly.
"""
import ast
import contextlib
import math
import os
import sys
import threading
import time
import types

import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

ORDER = []          # ("D2H"|"H2D", chunk) in issue order, per test


class FakeStream:
    def __init__(self, device=None): self.device = device
    def synchronize(self): pass
    def wait_stream(self, other): pass


class FakeEvent:
    def record(self, stream=None): pass
    def synchronize(self): pass
    def wait(self, stream=None): pass


def build():
    """_Collective plus its helpers, with torch.cuda stubbed."""
    src = open(os.path.join(PACK, "multistream", "split.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    want = {"_Collective"}
    body = [n for n in tree.body if getattr(n, "name", None) in want]
    body += [n for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") in ("MIN_PIPELINE_BYTES", "_XFER_STREAMS", "_XFER_LOCK",
                                              "EXCHANGE_MODES") for t in n.targets)]
    cuda = types.SimpleNamespace(Stream=FakeStream, Event=FakeEvent,
                                 current_stream=lambda d=None: FakeStream(d),
                                 stream=lambda s: contextlib.nullcontext())
    g = {"torch": types.SimpleNamespace(empty=torch.empty, cat=torch.cat, cuda=cuda),
         "threading": threading, "time": time, "math": math,
         "MultiStreamError": type("MultiStreamError", (RuntimeError,), {}),
         "_enable_peer_access": lambda d: None}
    exec(compile(ast.Module(body=sorted(body, key=lambda n: n.lineno), type_ignores=[]),
                 "split.py", "exec"), g)
    C = g["_Collective"]

    # record the transfer schedule: staging is D2H, fetching is H2D
    real_stage_buf, real_fetch, real_recv = C._pinned_buffer, C._fetch, C._h2d
    def spy_buf(self, rank, key, tensor):
        ORDER.append(("D2H", key[1] if isinstance(key, tuple) else key))
        return real_stage_buf(self, rank, key, tensor)
    def spy_fetch(self, src, device):
        ORDER.append(("H2D", None))
        return real_fetch(self, src, device)
    def spy_recv(self, dsts, srcs, stream):            # the pipelined path receives through this
        ORDER.extend(("H2D", None) for _ in srcs)
        return real_recv(self, dsts, srcs, stream)
    C._pinned_buffer, C._fetch, C._h2d = spy_buf, spy_fetch, spy_recv
    return C, g


def run_ranks(coll, fn, n):
    """Run fn(rank) on n threads and return results in rank order."""
    out, errs = [None] * n, []
    def worker(r):
        try:
            out[r] = fn(r)
        except BaseException as e:                      # noqa: BLE001
            errs.append(e); coll.abort()
    ts = [threading.Thread(target=worker, args=(r,)) for r in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errs:
        raise errs[0]
    return out


def main():
    C, g = build()
    torch.manual_seed(0)

    # --- identical results, chunked vs not, over several rank counts and shapes ---
    for n in (2, 3, 8):
        rows, width = 4096, 64
        data = [torch.arange(r * 10_000, r * 10_000 + rows * width, dtype=torch.float32).reshape(rows, width)
                for r in range(n)]
        plain = C("host", n)
        ref = run_ranks(plain, lambda r: plain.all_gather(r, data[r], "cpu"), n)
        for c in (2, 3, 7):
            ORDER.clear()
            piped = C("host", n, chunks=c, min_chunk_bytes=1)
            got = run_ranks(piped, lambda r: piped.all_gather(r, data[r], "cpu"), n)
            for r in range(n):
                assert torch.equal(ref[r], got[r]), f"n={n} c={c} rank {r}: pipelined result differs"
        print(f"  {n} ranks: chunked (c=2,3,7) == unchunked, {tuple(ref[0].shape)} gathered")

    # --- the gathered result is what every rank sent, in rank order ---
    n = 3
    data = [torch.full((300, 4), float(r)) for r in range(n)]
    piped = C("host", n, chunks=4, min_chunk_bytes=1)
    got = run_ranks(piped, lambda r: piped.all_gather(r, data[r], "cpu"), n)
    assert torch.equal(got[0], torch.cat(data, dim=0)), "rank order not preserved by the pipeline"
    print("  rank order preserved: cat(data) reproduced exactly")

    # --- all_to_all: identical results, chunked vs not ----------------------
    for n in (2, 3, 5):
        # uneven per-destination lengths, as real token bounds produce
        sends = [[torch.arange(r * 1000 + q * 7, r * 1000 + q * 7 + (600 + 37 * q) * 5,
                               dtype=torch.float32).reshape(-1, 5) for q in range(n)]
                 for r in range(n)]
        plain = C("host", n)
        ref = run_ranks(plain, lambda r: plain.all_to_all(r, sends[r], "cpu"), n)
        for c in (2, 5):
            piped = C("host", n, chunks=c, min_chunk_bytes=1)
            got = run_ranks(piped, lambda r: piped.all_to_all(r, sends[r], "cpu"), n)
            for r in range(n):
                assert len(got[r]) == n
                for q in range(n):
                    assert torch.equal(ref[r][q], got[r][q]), f"all_to_all n={n} c={c} r={r} q={q}"
        print(f"  all_to_all, {n} ranks, uneven lengths: chunked (c=2,5) == unchunked")

    # routing: what rank q sent to rank r must arrive at r in slot q
    n = 4
    sends = [[torch.full((120 + 11 * q, 3), float(r * 10 + q)) for q in range(n)] for r in range(n)]
    piped = C("host", n, chunks=3, min_chunk_bytes=1)
    got = run_ranks(piped, lambda r: piped.all_to_all(r, sends[r], "cpu"), n)
    for r in range(n):
        for q in range(n):
            assert torch.equal(got[r][q], sends[q][r]), f"routing wrong: r={r} q={q}"
    print("  all_to_all routing: recv[q] on rank r is exactly what rank q sent to r")

    # --- IT MUST ACTUALLY OVERLAP: D2H(j+1) issued before H2D(j) ------------
    ORDER.clear()
    piped = C("host", 2, chunks=4, min_chunk_bytes=1)
    run_ranks(piped, lambda r: piped.all_gather(r, torch.zeros(800, 8), "cpu"), 2)
    d2h = [i for i, (k, _) in enumerate(ORDER) if k == "D2H"]
    h2d = [i for i, (k, _) in enumerate(ORDER) if k == "H2D"]
    assert d2h and h2d, ORDER[:8]
    assert d2h[1] < h2d[0], (
        "the second chunk's upload was not issued before the first chunk's download -- "
        f"this is running serially: {ORDER[:8]}")
    print(f"  overlap: D2H(1) issued at step {d2h[1]}, before the first H2D at {h2d[0]}")

    # --- ranks that would plan DIFFERENT chunk counts must still agree ------
    # token shares are not equal, so a rank's own tensor can fall on the other side of the byte
    # threshold from its peers'. Disagreeing means waiting on a different number of per-chunk
    # barriers, which deadlocks -- so the count is agreed behind the full barrier.
    n = 3
    thresh = 400 * 4 * 4                                # a (400, 4) float32 tensor, exactly
    sizes = [200, 400, 4000]                            # below, at, far above the threshold
    piped = C("host", n, chunks=4, min_chunk_bytes=thresh)
    assert len({piped._plan(torch.zeros(r, 4)) for r in sizes}) > 1, "sizes must actually disagree"
    data = [torch.full((r, 4), float(q)) for q, r in enumerate(sizes)]
    done = threading.Event()
    res = [None]
    def race():
        res[0] = run_ranks(piped, lambda r: piped.all_gather(r, data[r], "cpu"), n)
        done.set()
    threading.Thread(target=race, daemon=True).start()
    assert done.wait(20), "ranks planned different chunk counts and deadlocked on the per-chunk barriers"
    assert torch.equal(res[0][0], torch.cat(data, dim=0)), "agreed-count gather produced the wrong result"
    print("  ranks proposing different chunk counts agree on one, and do not deadlock")

    # --- small transfers and c<=1 take the original single-shot path --------
    for chunks, nbytes, label in ((0, 1, "chunks=0"), (1, 1, "chunks=1"), (8, 1 << 30, "tensor below threshold")):
        coll = C("host", 2, chunks=chunks, min_chunk_bytes=nbytes)
        assert coll._plan(torch.zeros(64, 4)) == 0, f"{label} should not pipeline"
    print("  chunks<=1 and sub-threshold tensors take the single-shot path")

    # --- p2p has no two directions to overlap, so it never pipelines --------
    coll = C("p2p", 2, chunks=8, min_chunk_bytes=1)
    assert coll.chunks == 0 and coll._plan(torch.zeros(4096, 64)) == 0
    # the step log must read the EFFECTIVE count off the collective, never the node's request:
    # printing "p2p x8" for an unchunked exchange misleads anyone benchmarking p2p against host.
    for mode, want in (("host", 8), ("p2p", 0)):
        c = C(mode, 2, chunks=8, min_chunk_bytes=1)
        eff = getattr(c, "chunks", 0)
        assert eff == want, f"{mode}: effective chunks {eff}, expected {want}"
        label = mode if not eff else f"{mode} x{eff}"
        assert label == ("host x8" if mode == "host" else "p2p"), label
    print("  p2p never pipelines, and the step label reports the effective count")

    # --- abort unblocks every per-chunk barrier -----------------------------
    coll = C("host", 3, chunks=4, min_chunk_bytes=1)
    coll.abort()
    assert coll.barrier.broken and all(b.broken for b in coll._bars), "abort must break every barrier"
    print("  abort breaks the main barrier and every per-chunk barrier")

    print("PASS: exchange pipelining")


if __name__ == "__main__":
    main()
