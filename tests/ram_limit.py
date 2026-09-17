# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only check that RAM availability respects container memory limits (multistream/log.py memory_limit).

    python tests/ram_limit.py
"""
import importlib
import os
import sys
import tempfile

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
sys.argv = sys.argv[:1]
PACK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = importlib.import_module(f"{PACK}.multistream.log")
failures = []


def cgroup(files):
    root = tempfile.mkdtemp(prefix="h3ms-cg-")
    for rel, content in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
    return root


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        failures.append(label)


check("v2 unlimited (memory.max = max)", L.memory_limit(cgroup({"memory.max": "max\n", "memory.current": "123\n"})), (None, None))
pod = cgroup({"memory.max": "92999999488\n", "memory.current": "92999610368\n",
              "memory.stat": "anon 73756450816\nfile 18675793920\ninactive_file 4147953664\nactive_file 12851781632\n"})
M = L.RECLAIM_MARGIN
check("v2 Runpod pod at its limit: active file cache counts", L.memory_limit(pod),
      (92999999488, 92999999488 - 92999610368 + (4147953664 + 12851781632 - int(92999999488 * M))))
# the defect seen on a pod: right after loading, almost all file cache is on the ACTIVE list
loaded = cgroup({"memory.max": "93415538688\n", "memory.current": "89128960000\n",
                 "memory.stat": "anon 38000000000\nfile 55000000000\ninactive_file 418762752\nactive_file 54224519168\n"
                                "file_dirty 1048576\nfile_writeback 0\nshmem 2000000000\n"})
lim, avail = L.memory_limit(loaded)
check("v2 just after model load: active file cache is not lost", avail > 40 * 2**30, True)
check("v2 dirty pages and margin excluded", avail,
      93415538688 - 89128960000 + (418762752 + 54224519168 - 1048576 - int(93415538688 * M)))
check("v2 margin never makes reclaimable negative",
      L.memory_limit(cgroup({"memory.max": "8589934592\n", "memory.current": "8000000000\n",
                             "memory.stat": "inactive_file 1000\nactive_file 1000\n"})), (8589934592, 8589934592 - 8000000000))
check("v1 limited", L.memory_limit(cgroup({"memory/memory.limit_in_bytes": "8589934592\n", "memory/memory.usage_in_bytes": "4294967296\n",
                                           "memory/memory.stat": "total_inactive_file 1073741824\ntotal_active_file 2147483648\ntotal_dirty 4096\n"})),
      (8589934592, 4294967296 + (1073741824 + 2147483648 - 4096 - int(8589934592 * M))))
check("v1 unlimited (huge number)", L.memory_limit(cgroup({"memory/memory.limit_in_bytes": "9223372036854771712\n", "memory/memory.usage_in_bytes": "1\n"})), (None, None))
check("no cgroup files", L.memory_limit(cgroup({})), (None, None))
avail, total = L.ram_available(pod)
check("ram_available bounded by the pod limit", (total <= 92999999488, avail <= L.memory_limit(pod)[1]), (True, True))
print("this host:", L.ram())
print("RESULT", "FAIL " + ", ".join(failures) if failures else "all ok")
sys.exit(1 if failures else 0)
