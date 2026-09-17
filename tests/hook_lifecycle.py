# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Hook install / persistence / removal, the thing that made a bypassed node keep running.

    python tests/hook_lifecycle.py

CPU only, no ComfyUI server. Uses multistream/hooks.py directly with stand-in models, because the
behaviour under test is attribute plumbing, not anything model-specific.
"""
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PACK)

from multistream import hooks as ms_hooks   # noqa: E402

ORIG, WRAP = "_orig", "_wrap"
CALLS = []


class Model:
    def decode(self, x):
        CALLS.append(("original", x))
        return x


def install(model, label="Test"):
    original = model.decode
    setattr(model, ORIG, original)

    def wrapped(x):
        CALLS.append(("hooked", x))
        return original(x)

    model.decode = wrapped
    setattr(model, WRAP, {"decode": wrapped})
    ms_hooks.remember(model, label, ORIG, WRAP, ("decode",))
    return wrapped


def main():
    ms_hooks._ROOTS.clear()

    # --- 1. a hook runs, and survives the node not running again -------------
    m = Model()
    install(m)
    CALLS.clear()
    m.decode(1)
    assert CALLS == [("hooked", 1), ("original", 1)], CALLS
    assert ms_hooks.installed() == [{"hook": "Test", "model": "Model", "methods": ["decode"]}]
    # the node is now bypassed: install() is never called again, but the hook is still there.
    # This is the reported bug -- it is documented behaviour, so assert it rather than pretend.
    CALLS.clear()
    m.decode(2)
    assert CALLS == [("hooked", 2), ("original", 2)], "a bypassed node's hook should still be attached"
    print("  installed hook runs, and persists when the node stops running (the reported behaviour)")

    # --- 2. uninstall really restores ---------------------------------------
    assert ms_hooks.uninstall(m, ORIG, WRAP, ("decode",), "Test") is True
    CALLS.clear()
    m.decode(3)
    assert CALLS == [("original", 3)], CALLS
    assert not hasattr(m, ORIG) and not hasattr(m, WRAP), "marker attributes left behind"
    assert ms_hooks.installed() == [], ms_hooks.installed()
    print("  uninstall restores the original, drops the markers, clears the registry")

    # --- 3. uninstalling twice is a no-op, not an error ---------------------
    assert ms_hooks.uninstall(m, ORIG, WRAP, ("decode",), "Test") is False
    print("  uninstalling an unhooked model returns False, no exception")

    # --- 4. re-install after uninstall works --------------------------------
    install(m)
    CALLS.clear()
    m.decode(4)
    assert CALLS == [("hooked", 4), ("original", 4)], CALLS
    ms_hooks.uninstall(m, ORIG, WRAP, ("decode",), "Test")
    print("  re-install after uninstall works")

    # --- 5. REFUSE to restore when someone wrapped on top of ours -----------
    m2 = Model()
    ours = install(m2, "Test")
    outer_calls = []

    def other_pack(x, _inner=m2.decode):
        outer_calls.append(x)
        return _inner(x)

    m2.decode = other_pack                       # a different pack wraps after us
    assert ms_hooks.uninstall(m2, ORIG, WRAP, ("decode",), "Test") is False, \
        "restoring would have silently dropped the other pack's wrapper"
    CALLS.clear()
    m2.decode(5)
    assert outer_calls == [5] and CALLS == [("hooked", 5), ("original", 5)], (outer_calls, CALLS)
    print("  refuses to detach under a foreign wrapper, leaving both chains intact")

    # --- 6. uninstall_all reports what it could and could not do ------------
    m3 = Model()
    install(m3, "Clean")
    out = ms_hooks.uninstall_all("test")
    assert out["detached"] == 1 and out["kept"] == 1, out
    assert "Clean" in out["message"] and "Test" in out["message"], out["message"]
    assert [h["hook"] for h in ms_hooks.installed()] == ["Test"], ms_hooks.installed()
    print(f"  uninstall_all: {out['message']}")

    test_registry_does_not_pin_models()

    ms_hooks._ROOTS.clear()
    print("PASS: hook lifecycle")



def test_registry_does_not_pin_models():
    """Regression, 2026-09-16: _ROOTS held a STRONG reference to the hooked model, so ComfyUI and
    comfy-aimdo could never free its VRAM. The text encoder stayed resident on its card for the life
    of the process and a VAE split worker (a separate process aimdo cannot see) then OOM'd."""
    import gc
    import weakref as _wr

    ms_hooks._ROOTS.clear()
    m = Model()
    install(m, "Pinned?")
    assert len(ms_hooks.installed()) == 1
    probe = _wr.ref(m)

    del m
    gc.collect()
    assert probe() is None, "the hook registry is keeping the model alive -- VRAM can never be freed"
    print("  a hooked model is still collectable (registry holds only a weakref)")

    # and the collected entry is pruned, so a reused id() cannot masquerade as a live hook
    assert ms_hooks.installed() == [], ms_hooks.installed()
    assert ms_hooks._ROOTS == {}, ms_hooks._ROOTS
    print("  the dead entry is pruned from the registry")

    # uninstall_all copes with a model that vanished under it
    m2 = Model()
    install(m2, "Gone")
    ms_hooks._ROOTS[id(m2)] = (*ms_hooks._ROOTS[id(m2)][:1], _wr.ref(Model()),
                               *ms_hooks._ROOTS[id(m2)][2:])   # a ref that is already dead
    out = ms_hooks.uninstall_all("test")
    assert out["detached"] == 0 and out["kept"] == 0, out
    print("  uninstall_all skips entries whose model was already collected")
    ms_hooks._ROOTS.clear()

if __name__ == "__main__":
    main()
