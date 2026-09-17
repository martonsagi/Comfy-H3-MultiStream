# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""non_dynamic_delegate: take ComfyUI's per-model DynamicVRAM opt-out, or fall back safely.

    python tests/dynamic_vram_optout.py

CPU only, ComfyUI stubbed. Running dynamic is always a valid fallback, so this must NEVER raise --
every failure mode returns the original model with a reason. The interlock that keeps
vram_block_cache off while the model is still dynamic is checked too: holding device tensors across
steps under DynamicVRAM aborts the process (docs/vram-block-residency.md).
"""
import ast
import os
import sys
import types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

WARNINGS = []


def build():
    src = open(os.path.join(PACK, "multistream", "split.py"), encoding="utf-8").read()
    body = [n for n in ast.parse(src).body if getattr(n, "name", None) == "non_dynamic_delegate"]
    g = {"log": types.SimpleNamespace(warning=lambda *a, **kw: WARNINGS.append(a[0] % a[1:] if len(a) > 1 else a[0]),
                                      info=lambda *a, **kw: None, exception=lambda *a, **kw: None)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "split.py", "exec"), g)
    return g["non_dynamic_delegate"]


class Legacy:
    def is_dynamic(self): return False


class Dynamic:
    def __init__(self, delegate="ok"): self._delegate = delegate
    def is_dynamic(self): return True
    def get_non_dynamic_delegate(self):
        if self._delegate == "raise":
            raise RuntimeError("Cannot create non-dynamic delegate: cached_patcher_init is not initialized.")
        if self._delegate == "none":
            return None
        if self._delegate == "still-dynamic":
            return Dynamic()
        return Legacy()


def main():
    delegate = build()

    m, why = delegate(Dynamic())
    assert isinstance(m, Legacy) and why == "delegated", (m, why)
    print("  a dynamic model is delegated to a legacy patcher")

    orig = Legacy()
    m, why = delegate(orig)
    assert m is orig and why == "already non-dynamic"
    print("  an already-legacy model is returned untouched")

    class NoApi:                                  # a build without DynamicVRAM at all
        pass
    orig = NoApi()
    m, why = delegate(orig)
    assert m is orig and "no is_dynamic()" in why, why
    print("  a build without DynamicVRAM is left alone")

    class NoDelegate:
        def is_dynamic(self): return True
    WARNINGS.clear()
    orig = NoDelegate()
    m, why = delegate(orig)
    assert m is orig and "unavailable" in why and WARNINGS, (why, WARNINGS)
    print("  a ComfyUI without get_non_dynamic_delegate() warns and stays dynamic")

    # the documented failure: cached_patcher_init missing -> RuntimeError from ComfyUI
    WARNINGS.clear()
    orig = Dynamic("raise")
    m, why = delegate(orig)
    assert m is orig and why.startswith("failed: RuntimeError"), why
    assert WARNINGS and "staying dynamic" in WARNINGS[0], WARNINGS
    print("  a loader failure is caught: original returned, never raised")

    for bad, label in (("none", "None"), ("still-dynamic", "a still-dynamic delegate")):
        WARNINGS.clear()
        orig = Dynamic(bad)
        m, why = delegate(orig)
        assert m is orig, f"{label} must not be adopted"
        print(f"  {label} is rejected, original kept ({why})")

    # --- the interlock lives in the node; assert it is wired ------------------
    src = open(os.path.join(PACK, "__init__.py"), encoding="utf-8").read()
    body = src[src.index("class H3MultiStream"):]
    assert "if vram_block_cache and still_dynamic:" in body, \
        "vram_block_cache must be refused while the model is still dynamic"
    assert "vram_block_cache = False" in body, "the interlock must actually turn the VRAM tier off"
    assert "ms_split.non_dynamic_delegate(model)" in body, "patch() must take the delegate"
    print("  node interlock: vram_block_cache forced off while the model is still dynamic")

    print("PASS: dynamic vram opt-out")


if __name__ == "__main__":
    main()
