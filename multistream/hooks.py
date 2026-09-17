# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""Installing and REMOVING this pack's method hooks.

Why this exists. The text-encoder cache, the VAE cache and the VAE split decode do not wrap a
ModelPatcher the way the MultiStream node does -- they replace methods on `clip.cond_stage_model` and
`vae.first_stage_model`. ComfyUI shares and caches those objects across prompts, so a hook installed
by one prompt outlives the node that installed it: bypassing or muting the node means install() is
never called again, the hook stays attached, and it keeps running with the configuration from the
last prompt that DID run it. (The MultiStream node is unaffected -- its wrappers live on a clone that
is rebuilt per prompt, so bypassing it genuinely disables it.)

ComfyUI gives a custom node no signal that it was bypassed, so this cannot be detected from inside the
hook. What it can do is make removal possible and explicit:

  * every node here now has an `enabled` input, and `enabled=False` really uninstalls -- the original
    method goes back and the marker attributes are dropped, so nothing of ours runs or logs;
  * `uninstall_all()` (menu: "Detach all hooks", POST /h3multistream/hooks/detach) clears every hook
    this process installed, which is the escape hatch for the bypass case;
  * `installed()` reports what is currently hooked, so /h3multistream/status can show it.

Restoring is refused when something else has wrapped on top of ours, because putting the original
back would silently drop the other pack's wrapper.
"""
import threading
import weakref

from .log import log

# id(model) -> (label, WEAKREF to model, orig attr, wrap attr, method names).
#
# The reference MUST be weak. ComfyUI and comfy-aimdo free a model's VRAM when the model object is
# dropped, so a strong reference here pins the text encoder (and the VAEs) in VRAM for the life of
# the process: the DiT then shares a card with an encoder that will never be evicted, and a VAE split
# worker -- a separate process aimdo cannot see -- OOMs. That regression shipped on 2026-09-16 and is
# why te_cache.ATTACHED_ROOTS was already a WeakSet; this registry must follow the same rule.
_ROOTS = {}
_LOCK = threading.Lock()


def remember(model, label, orig_attr, wrap_attr, methods):
    """Record a freshly installed hook so uninstall_all() and installed() can find it."""
    with _LOCK:
        _ROOTS[id(model)] = (label, weakref.ref(model), orig_attr, wrap_attr, tuple(methods))


def _live():
    """[(key, label, model, orig_attr, wrap_attr, methods)] for entries whose model is still alive.

    Prunes collected ones: id() is reused after a model is freed, so a stale key could otherwise be
    mistaken for a live hook on a different object."""
    out = []
    with _LOCK:
        for key, (label, ref, orig_attr, wrap_attr, methods) in list(_ROOTS.items()):
            model = ref()
            if model is None:
                del _ROOTS[key]
                continue
            out.append((key, label, model, orig_attr, wrap_attr, methods))
    return out


def installed():
    """[(label, [method names])] for every hook this process currently has attached."""
    return [{"hook": label, "model": type(model).__name__, "methods": list(methods)}
            for _key, label, model, orig_attr, _wrap, methods in _live()
            if getattr(model, orig_attr, None) is not None]


def uninstall(model, orig_attr, wrap_attr, methods, label):
    """Put the original method(s) back. False (and a warning) if someone wrapped on top of ours."""
    original = getattr(model, orig_attr, None)
    if original is None:
        return False
    ours = getattr(model, wrap_attr, None)
    for name in methods:
        current = getattr(model, name, None)
        if ours is not None and name in ours and current is not ours[name]:
            log.warning("[%s] not detaching %s: another wrapper was installed on top of ours; "
                        "removing ours would drop theirs. Restart ComfyUI to clear it.", label, name)
            return False
    if isinstance(original, dict):
        for name, fn in original.items():
            setattr(model, name, fn)
    else:
        setattr(model, methods[0], original)
    try:
        delattr(model, orig_attr)
    except AttributeError:
        pass
    for attr in (wrap_attr,):
        try:
            delattr(model, attr)
        except AttributeError:
            pass
    with _LOCK:
        _ROOTS.pop(id(model), None)
    log.info("[%s] hook detached: %s restored", label, ", ".join(methods))
    return True


def uninstall_all(reason="detached on request"):
    """Remove every hook this process installed. The escape hatch for a bypassed node."""
    done, kept = [], []
    for _key, label, model, orig_attr, wrap_attr, methods in _live():
        (done if uninstall(model, orig_attr, wrap_attr, methods, label) else kept).append(label)
    msg = (f"detached {len(done)} hook(s): {', '.join(sorted(set(done)))}" if done
           else "no hooks were attached")
    if kept:
        msg += f"; {len(kept)} left in place ({', '.join(sorted(set(kept)))}) -- see the log"
    log.info("[Hooks] %s (%s)", msg, reason)
    return {"detached": len(done), "kept": len(kept), "message": msg}
