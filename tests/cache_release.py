# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""WeightCache.in_use / close / release, with stubs. No torch, no CUDA, no ComfyUI."""
import ast, os, sys, threading, time, types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(PACK, "multistream", "cache.py"), encoding="utf-8").read()
tree = ast.parse(src)
keep = {"WeightCache", "release", "release_async", "clear_all", "acquire", "_group_of", "gib"}
body = [n for n in tree.body if isinstance(n, (ast.Assign, ast.Import, ast.ImportFrom))
        or getattr(n, "name", None) in keep]
body = [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]
import contextlib, dataclasses, os, collections
unreg = []
g = {"contextlib": contextlib, "dataclasses": dataclasses, "os": os, "threading": threading,
     "time": time, "defaultdict": collections.defaultdict,
     "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
         cudart=lambda: types.SimpleNamespace(cudaHostUnregister=lambda p: unreg.append(p)))),
     "log": types.SimpleNamespace(**{k: (lambda *a, **kw: None) for k in
                                     ("info", "warning", "error", "debug", "exception")}),
     "gib": lambda n: f"{n} B", "ram": lambda: "ram", "ram_available": lambda: (10**12,),
     "QuantizedTensor": type("QT", (), {})}
exec(compile(ast.Module(body=body, type_ignores=[]), "cache.py", "exec"), g)
WeightCache, release, _CACHES = g["WeightCache"], g["release"], g["_CACHES"]

def make(group, path, nbytes=1024):
    c = WeightCache((group, path, 1, 1))
    c.blocks[0] = {("m", "weight"): object()}
    c._keep = [types.SimpleNamespace(data_ptr=lambda: id(path))]
    c.bytes = nbytes
    _CACHES[(group, path, 1, 1)] = c
    return c

# 1. a plain release frees and removes the cache
c = make("te", "/te.safetensors", 25 * 2**30)
assert release("te") == 25 * 2**30
assert not _CACHES and c.bytes == 0 and unreg, "released and unregistered"

# 2. release BLOCKS while a reader holds in_use(), then completes
c = make("dit", "/dit.safetensors", 18 * 2**30)
order, entered = [], threading.Event()
def reader():
    with c.in_use() as held:
        assert held is c
        entered.set(); order.append("reader-start")
        time.sleep(0.40)
        order.append("reader-end")
t = threading.Thread(target=reader); t.start(); entered.wait()
t0 = time.perf_counter(); freed = release("dit"); waited = time.perf_counter() - t0
order.append("released"); t.join()
assert order == ["reader-start", "reader-end", "released"], order
assert waited >= 0.3, f"release returned after only {waited:.3f}s -- it did not wait"
assert freed == 18 * 2**30 and not _CACHES

# 3. a reader arriving DURING close gets None and does not resurrect the cache
c = make("te", "/te2.safetensors")
holder, closing = threading.Event(), threading.Event()
def hold():
    with c.in_use():
        holder.set(); time.sleep(0.40)
def do_release():
    release("te")
def late():
    closing.wait()                      # only start AFTER close() has set _closed
    with c.in_use() as held:
        order.append(("late", held))
h = threading.Thread(target=hold); h.start(); holder.wait()
r = threading.Thread(target=do_release); r.start()
while not c._closed:                    # release() is now inside close(), waiting on the reader
    time.sleep(0.005)
closing.set()
l = threading.Thread(target=late); l.start()
h.join(); r.join(); l.join()
assert ("late", None) in order, order
assert c.get_block(0, None) is None, "a closed cache must not hand out blocks"

# 4. close() REFUSES rather than unpinning under a reader that will not finish
c = make("vae:/v.safetensors", "/v.safetensors", 5 * 2**30)
stuck = threading.Event(); go = threading.Event()
def never():
    with c.in_use():
        stuck.set(); go.wait(5)
n = threading.Thread(target=never); n.start(); stuck.set(); stuck.wait()
assert c.close(wait_seconds=0.2) is False, "close() must refuse while a reader holds it"
assert c.bytes == 5 * 2**30, "a refused close must leave the cache intact"
assert not c._closed, "a refused close must let readers back in"
# and "vae" matches the per-file "vae:<path>" group
go.set(); n.join()
assert release("vae") == 5 * 2**30 and not _CACHES

print("all 4 scenarios passed")
