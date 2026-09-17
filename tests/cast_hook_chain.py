# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only check that MultiStream's cast hook chains with another pack's hook in any order.

    python tests/cast_hook_chain.py

Simulates a second pack that replaces comfy.ops.cast_bias_weight without chaining, installs
both in both orders and repeatedly, and checks that shadow modules of each pack reach their own cast while ordinary
modules reach ComfyUI's original.
"""
import importlib
import os
import sys

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ComfyUI root
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
sys.argv = sys.argv[:1]

import comfy.ops  # noqa: E402

PACK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ms_cast = importlib.import_module(f"{PACK}.multistream.cast")

ORIGINAL = comfy.ops.cast_bias_weight
calls = []


def fake_original(s, *a, **k):
    calls.append("comfy")
    return "comfy"


def fake_shadow_cast(s, *a, **k):
    calls.append("multistream")
    return "multistream"


class Other:
    """A non-chaining hook from another pack: captures the function present at import, replaces unconditionally."""

    def __init__(self):
        self.orig = comfy.ops.cast_bias_weight

    def dispatch(self, s, *a, **k):
        if getattr(s, "_other_rank", None) is not None:
            calls.append("other")
            return "other"
        return self.orig(s, *a, **k)

    def install(self):
        if comfy.ops.cast_bias_weight is not self.dispatch:
            comfy.ops.cast_bias_weight = self.dispatch


class M:
    pass


plain, ns_mod, other_mod = M(), M(), M()
ns_mod._multistream_rank = 0
other_mod._other_rank = 0
ms_cast.shadow_cast = fake_shadow_cast   # the dispatcher looks shadow_cast up at call time


def reset(base=None):
    """Fresh ComfyUI state. base: what MultiStream saw at import time (ComfyUI's cast, or a hook installed before it)."""
    comfy.ops.cast_bias_weight = fake_original
    ms_cast._PREV_CAST = None
    ms_cast._BASE_CAST = base or fake_original


def route(mod):
    calls.clear()
    comfy.ops.cast_bias_weight(mod)
    return calls[-1]


failures = []


def check(label, expect_ns=True):
    got = {"plain": route(plain), "other": route(other_mod)}
    if expect_ns:
        got["multistream"] = route(ns_mod)
    want = {"plain": "comfy", "other": "other", "multistream": "multistream"}
    bad = {k: v for k, v in got.items() if v != want[k]}
    print(f"{label}: {got}{'  FAIL ' + str(bad) if bad else '  ok'}")
    if bad:
        failures.append(label)


# order 1: other pack first (imports and installs), then MultiStream imports (sees the other's hook) and installs
reset()
other = Other()
other.install()
ms_cast._BASE_CAST = comfy.ops.cast_bias_weight
ms_cast.install_cast_hook()
check("other then multistream")

# order 2: MultiStream first, then the other pack imports (captures MultiStream's dispatcher as its original) and installs
reset()
ms_cast.install_cast_hook()
other = Other()
other.install()
check("multistream then other (other imported after)")

# order 2b, the cycle: as order 2, then MultiStream re-hooks on top at its next use (prev = other, other.orig = multistream)
ms_cast.install_cast_hook()
check("multistream, other imported after, multistream re-hooks (cycle guard)")

# order 3: both imported before either installs (each captured ComfyUI's original); the other installs last and wins;
# MultiStream re-installs at its next use and must keep the other's hook as fallback
reset()
other = Other()
ms_cast.install_cast_hook()
other.install()
ms_cast.install_cast_hook()
check("both imported first, other installed last, multistream re-hooks")

# repeated installs must not build a loop
for _ in range(3):
    ms_cast.install_cast_hook()
    other.install()
    ms_cast.install_cast_hook()
check("repeated installs")

comfy.ops.cast_bias_weight = ORIGINAL
print("RESULT", "FAIL " + ", ".join(failures) if failures else "all ok")
sys.exit(1 if failures else 0)
