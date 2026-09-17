# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only checks for GPU selection (multistream/gpus.py): parsing, validation and rank plans for 1-8 GPUs.

    python tests/gpu_set_resolve.py
"""
import importlib
import os
import sys

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
sys.argv = sys.argv[:1]

PACK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
g = importlib.import_module(f"{PACK}.multistream.gpus")

failures = []


def ranks(plan):
    return [d.index for d in plan.devices]


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        failures.append(label)


def raises(label, fn, text):
    try:
        fn()
    except g.GPUSelectionError as e:
        ok = text in str(e)
        print(f"{'ok  ' if ok else 'FAIL'} {label}: raised '{e}'")
        if not ok:
            failures.append(label)
        return
    print(f"FAIL {label}: did not raise")
    failures.append(label)


free8 = {i: 15.0 for i in range(8)}
free = lambda i: free8[i]

# auto
check("auto, 2 visible, primary 1", ranks(g.resolve(g.GPUSet(), 1, device_count=2, free_gib=free)), [1, 0])
check("auto, 8 visible, primary 0", ranks(g.resolve(g.GPUSet(), 0, device_count=8, free_gib=free)), list(range(8)))
check("auto, 8 visible, primary 5", ranks(g.resolve(g.GPUSet(), 5, device_count=8, free_gib=free)), [5, 0, 1, 2, 3, 4, 6, 7])
check("auto, 1 visible", ranks(g.resolve(g.GPUSet(), 0, device_count=1, free_gib=free)), [0])

# explicit lists keep their order after the primary
s = g.GPUSet.from_inputs("3,1,2", device_count=4)
check("explicit 3,1,2 primary 1", ranks(g.resolve(s, 1, device_count=4, free_gib=free)), [1, 3, 2])
p = g.resolve(g.GPUSet.from_inputs("2,3", device_count=4), 0, device_count=4, free_gib=free)
check("explicit without primary adds it", (ranks(p), any("added" in n for n in p.notes)), ([0, 2, 3], True))

# exclude, min free VRAM, max_gpus
check("exclude 0 on 4 GPUs, primary 1",
      ranks(g.resolve(g.GPUSet.from_inputs(exclude="0", device_count=4), 1, device_count=4, free_gib=free)), [1, 2, 3])
raises("excluding the primary", lambda: g.resolve(g.GPUSet.from_inputs(exclude="1", device_count=2), 1, device_count=2),
       "is excluded")
lowfree = {0: 15.0, 1: 15.0, 2: 3.0, 3: 15.0}
p = g.resolve(g.GPUSet.from_inputs(min_free_vram_gb=6, device_count=4), 0, device_count=4, free_gib=lambda i: lowfree[i])
check("min free VRAM skips cuda:2", (ranks(p), any("skipped cuda:2" in n for n in p.notes)), ([0, 1, 3], True))
check("min free never drops the primary",
      ranks(g.resolve(g.GPUSet.from_inputs(min_free_vram_gb=6, device_count=4), 2, device_count=4,
                      free_gib=lambda i: lowfree[i])), [2, 0, 1, 3])
check("max_gpus 3 on 8", ranks(g.resolve(g.GPUSet.from_inputs(max_gpus=3, device_count=8), 0, device_count=8,
                                         free_gib=free)), [0, 1, 2])
check("max_gpus 1 = one card", ranks(g.resolve(g.GPUSet.from_inputs(max_gpus=1, device_count=2), 1, device_count=2,
                                                free_gib=free)), [1])

# legacy second_gpu
check("legacy -1 on 2 GPUs, primary 1", ranks(g.resolve(g.GPUSet.legacy(-1), 1, device_count=2, free_gib=free)), [1, 0])
p = g.resolve(g.GPUSet.legacy(0), 1, device_count=2, free_gib=free)
check("legacy second 0, primary 1, no note", (ranks(p), p.notes), ([1, 0], []))

# shares follow the GPU, not the position after reordering
p = g.resolve(g.GPUSet.from_inputs("0,1,2", shares="1,0.7,1", device_count=3), 2, device_count=3, free_gib=free)
check("shares map to devices", list(zip(ranks(p), p.shares)), [(2, 1.0), (0, 1.0), (1, 0.7)])

# limiting to the implemented split size
p = g.resolve(g.GPUSet(), 0, device_count=4, free_gib=free).limited(2)
check("limited(2) keeps primary + next", (ranks(p), any("uses 2" in n for n in p.notes)), ([0, 1], True))

# validation
raises("index out of range", lambda: g.GPUSet.from_inputs("0,4", device_count=2), "does not exist")
raises("duplicate index", lambda: g.GPUSet.from_inputs("1,1", device_count=2), "listed twice")
raises("not a number", lambda: g.GPUSet.from_inputs("0,x", device_count=2), "is not a CUDA index")
raises("share count mismatch", lambda: g.GPUSet.from_inputs("auto", shares="1", device_count=2), "value(s) for 2 GPU(s)")
raises("non-positive share", lambda: g.GPUSet.from_inputs("auto", shares="1,0", device_count=2), "greater than 0")
raises("max_gpus out of range", lambda: g.GPUSet.from_inputs(max_gpus=9, device_count=2), "between 1 and")
check("cuda: prefix accepted", g.GPUSet.from_inputs("cuda:1, cuda:0", device_count=2).gpus, (1, 0))

print("RESULT", "FAIL " + ", ".join(failures) if failures else "all ok")
sys.exit(1 if failures else 0)
