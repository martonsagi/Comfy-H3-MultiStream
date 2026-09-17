# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only, no ComfyUI imports: the VAE worker's shared-memory handoff works on the running Python version.

    python tests/shm_handoff.py      (run it with every Python you support, e.g. 3.12 and 3.13)

A child process creates and fills a segment the way multistream/vae_worker.py does and exits; the parent attaches,
checks the bytes and removes it the way multistream/vae_split.py does. Passes when the data matches, the segment is gone
from /dev/shm and no process printed a resource_tracker warning.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "multistream", "vae_split.py")


def helpers():
    """Load _SHM_HAS_TRACK, _shm_open and _shm_unlink from vae_split.py without importing torch or ComfyUI."""
    text = open(SRC, encoding="utf-8").read()
    check = re.search(r"^_SHM_HAS_TRACK = .*$", text, re.M).group(0)
    body = re.search(r"^def _shm_open\(.*?(?=^class WorkerError)", text, re.M | re.S).group(0)
    ns = {}
    exec("import inspect\nfrom multiprocessing import shared_memory\n" + check + "\n" + body, ns)
    return ns


if len(sys.argv) > 1 and sys.argv[1] == "child":
    ns = helpers()
    shm = ns["_shm_open"](create=True, size=1 << 20)
    shm.buf[:5] = b"h3ms!"
    print(shm.name, flush=True)
    shm.close()
    sys.exit(0)

ns = helpers()
child = subprocess.run([sys.executable, __file__, "child"], capture_output=True, text=True, timeout=60)
name = child.stdout.strip()
shm = ns["_shm_open"](name=name)
data = bytes(shm.buf[:5])
shm.close()
ns["_shm_unlink"](shm)
leftover = os.path.exists(os.path.join("/dev/shm", name.lstrip("/")))
out = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True)
problems = []
if data != b"h3ms!":
    problems.append(f"data {data!r}")
if leftover:
    problems.append("segment still in /dev/shm")
if "resource_tracker" in child.stderr or "KeyError" in child.stderr:
    problems.append("child stderr: " + child.stderr.strip()[:200])
print(f"python {sys.version.split()[0]}: track argument {'yes' if ns['_SHM_HAS_TRACK'] else 'no'}, segment {name}, data {data!r}, "
      f"left in /dev/shm {leftover}")
print("RESULT", "all ok" if not problems else "FAIL: " + "; ".join(problems), flush=True)
sys.exit(1 if problems else 0)
