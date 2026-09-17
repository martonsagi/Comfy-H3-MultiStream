# SPDX-FileCopyrightText: 2026 Márton Sági
# SPDX-License-Identifier: GPL-3.0-only
"""H3 MultiStream: MiniMax H3 across several GPUs for one render, plus caches and resource controls.

Nodes: H3 MultiStream, H3 MS GPU Set, H3 MS Text Encoder Cache, H3 MS VAE Cache, H3 MS VAE Split Decode,
H3 MS Release Resources.
UI: Extensions -> H3 MultiStream menu (web/h3multistream.js). HTTP: /h3multistream/status, /h3multistream/vae_workers/release,
/h3multistream/cache/clear_weights, /h3multistream/cache/clear_outputs.
"""
import os

import comfy.patcher_extension

from .multistream import cache as ms_cache
from .multistream import hooks as ms_hooks
from .multistream import vram_cache as ms_vram
from .multistream import cast as ms_cast
from .multistream import gpus as ms_gpus
from .multistream import split as ms_split
from .multistream import te_cache as ms_te_cache
from .multistream import vae_cache as ms_vae_cache
from .multistream import vae_split as ms_vae_split
from .multistream.log import log, ram

ms_cast.install_cast_hook()

WEB_DIRECTORY = "./web"


class H3MSGPUSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gpus": ("STRING", {"default": "auto",
                         "tooltip": "auto = every visible CUDA GPU, or a list such as 0,1,3. The model's own GPU always "
                                    "takes part as rank 0; the others follow in this order."}),
                "exclude": ("STRING", {"default": "",
                            "tooltip": "GPUs to leave out, e.g. 2 for a card reserved for another service."}),
                "shares": ("STRING", {"default": "",
                           "tooltip": "Optional relative speed per selected GPU (same order as gpus, or by index for "
                                      "auto), e.g. 1,1,0.7 for a slower or power-capped card. H3 MultiStream sizes each GPU's "
                                      "attention heads and tokens by it; the VAE split divides evenly."}),
                "min_free_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.5,
                                     "tooltip": "Skip GPUs with less free VRAM than this when the plan is made (first "
                                                "use). Never skips the model's own GPU. 0 = no check."}),
                "max_gpus": ("INT", {"default": ms_gpus.MAX_GPUS, "min": 1, "max": ms_gpus.MAX_GPUS,
                             "tooltip": "Upper limit on GPUs used. 1 = run unsplit (the caches still work)."}),
            }
        }

    RETURN_TYPES = ("H3MS_GPUS",)
    RETURN_NAMES = ("gpus",)
    FUNCTION = "build"
    CATEGORY = "advanced/model"
    DESCRIPTION = ("Selects the GPUs for H3 MultiStream and H3 MS VAE Split Decode: include/exclude lists, relative "
                   "shares, a free-VRAM check and a GPU cap. Without this node both use every visible GPU.")

    def build(self, gpus="auto", exclude="", shares="", min_free_vram_gb=0.0, max_gpus=ms_gpus.MAX_GPUS):
        gpu_set = ms_gpus.GPUSet.from_inputs(gpus, exclude, shares, min_free_vram_gb, max_gpus)
        log.info("[GPUs] GPU set: %s (%d CUDA device(s) visible)", gpu_set.describe(), ms_gpus.torch.cuda.device_count())
        return (gpu_set,)


class H3MultiStream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "second_gpu": ("INT", {"default": -1, "min": -1, "max": 15,
                                       "tooltip": "CUDA index of the second GPU; -1 = automatic. Ignored when a GPU "
                                                  "set is connected."}),
                "exchange": (list(ms_split.EXCHANGE_MODES), {"default": "host",
                             "tooltip": "host: stage per-block exchanges through pinned RAM (safe everywhere). "
                                        "p2p: direct GPU-to-GPU copies (needs working P2P)."}),
                "exchange_chunks": ("INT", {"default": 0, "min": 0, "max": 32,
                                    "tooltip": "Pipeline the per-block exchange in this many chunks so each card's "
                                               "upload overlaps its download -- PCIe is full duplex, and the serial "
                                               "path pays D2H + H2D where the pipeline approaches max(D2H, H2D). "
                                               "0 or 1 keeps the original single-shot exchange. 8 is the measured "
                                               "recommendation: 1.12x on the step, 1.41x on the "
                                               "exchange itself. Host mode only; "
                                               "p2p has no two directions to overlap. Exchange is ~36% of a sparse "
                                               "step, so this is where the remaining headroom is."}),
                "sparse_attention": ("BOOLEAN", {"default": False,
                                     "tooltip": "Run ComfyUI's Model Sparse Attention node inside the split instead "
                                                "of refusing it. Each rank runs the sparse kernel over its own "
                                                "attention heads, which is exact (sol-attn/sla select per head), not "
                                                "an approximation. Needs that node in the chain. Off: a sparse "
                                                "patch raises an error, as before."}),
                "dynamic_vram": (list(ms_split.DYNAMIC_VRAM_MODES), {"default": "keep",
                                 "tooltip": "'off for this model' takes ComfyUI's per-model DynamicVRAM opt-out "
                                            "(ModelPatcherDynamic.get_non_dynamic_delegate) for the DiT only -- every "
                                            "other model in the process keeps DynamicVRAM. The split already works "
                                            "around aimdo in several places, and device tensors it allocates cannot "
                                            "be reused across steps under it, so vram_block_cache REQUIRES this. "
                                            "Costs a second load of the checkpoint, once per process."}),
                "vram_block_cache": ("BOOLEAN", {"default": False,
                                     "tooltip": "Make VRAM the FIRST tier of the weight cache: as many leading blocks "
                                                "as fit live on the cards, and weight_cache holds the rest in pinned "
                                                "host RAM. A block on the cards is NOT pinned in RAM as well, so the "
                                                "host cache shrinks by that much. Sized at run time from free VRAM, "
                                                "so it follows resolution and clip length. Released when sampling "
                                                "ends -- nothing else can free it. Turn weight_cache on too, or the "
                                                "overflow blocks stream from pageable memory."}),
                "vram_reserve_gb": ("FLOAT", {"default": ms_vram.SAFETY_GIB, "min": 0.0, "max": 32.0, "step": 0.5,
                                    "tooltip": "VRAM left free on each rank GPU beyond the step working set, when "
                                               "vram_block_cache is on. ComfyUI can see these allocations but can "
                                               "never reclaim them, so leaving it nothing makes ComfyUI start "
                                               "unloading its own models. Raise it if a later node (the VAE split "
                                               "worker needs ~5.2 GiB) runs short. Default comes from "
                                               "H3MS_VRAM_SAFETY_GIB."}),
                "sparse_vsa": ("BOOLEAN", {"default": False,
                               "tooltip": "Allow the 'vsa' method inside the split (needs sparse_attention on, and "
                                          "FastH3 weights carrying to_gate_compress). Off: a 'vsa' patch raises "
                                          "with a reason -- this is the rollback switch; sol-attn and sla are "
                                          "unaffected. Costs ~322 MB VRAM per GPU for the coarse-gate buffer."}),
                "weight_cache": ("BOOLEAN", {"default": True,
                                 "tooltip": "Keep the block weights in pinned RAM for the life of the ComfyUI process "
                                            "(~18.5 GB for MiniMax H3 int8). Survives prompts and model reloads; "
                                            "ComfyUI never evicts it."}),
                "cache_ram_reserve_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5,
                                         "tooltip": "Only fill the weight cache while at least this much system RAM "
                                                    "would stay available. 0 = use RAM as needed."}),
            },
            "optional": {
                "gpus": ("H3MS_GPUS", {"tooltip": "GPU selection from H3 MS GPU Set. Without it: every visible GPU "
                                                  "(or second_gpu)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"
    DESCRIPTION = ("Splits MiniMax H3's transformer blocks across up to 8 GPUs (sequence + head parallel, one thread "
                   "per GPU). Weights stream from host RAM to every card; output matches one GPU. With one usable GPU "
                   "the model runs unsplit.")

    def patch(self, model, enabled, second_gpu, exchange="host", exchange_chunks=0, sparse_attention=False,
              sparse_vsa=False,
              dynamic_vram="keep", vram_block_cache=False, vram_reserve_gb=None, weight_cache=True,
              cache_ram_reserve_gb=0.0, gpus=None):
        dyn = "kept"
        if dynamic_vram != "keep":
            model, dyn = ms_split.non_dynamic_delegate(model)
        still_dynamic = callable(getattr(model, "is_dynamic", None)) and model.is_dynamic()
        if vram_block_cache and still_dynamic:
            # INTERLOCK, not a preference. Holding device tensors across sampler steps under
            # DynamicVRAM raises an illegal access on the second step and ABORTS the process
            # (2026-09-16; docs/vram-block-residency.md). Refusing is the only safe default.
            log.warning("[MultiStream] vram_block_cache needs dynamic_vram='off for this model'; "
                        "the model is still dynamic (%s), so the VRAM tier stays OFF", dyn)
            vram_block_cache = False
        m = model.clone()
        if not enabled:
            log.info("[MultiStream] node disabled: model passes through unsplit")
            return (m,)
        gpu_set = gpus if gpus is not None else ms_gpus.GPUSet.legacy(second_gpu)
        plan = ms_gpus.resolve(gpu_set, model.load_device)
        if plan.n < 2:
            # no wrappers at all: the model runs exactly as in plain ComfyUI (compiler on, no split, no DiT weight cache)
            ms_gpus.log_plan("dit", gpu_set, plan)
            log.info("[MultiStream] 1 usable GPU (%s): model left unsplit, no split wrappers installed; the text-encoder "
                     "and VAE cache nodes still apply", model.load_device)
            return (m,)
        key = ms_split.cache_key_for(model) if weight_cache else None
        if weight_cache and key is None:
            log.warning("[MultiStream] weight cache unavailable: the model's loader has no reload factory")
        if not weight_cache:
            # switching the toggle off used to stop the cache being used and refilled while leaving every
            # cudaHostRegister'd page pinned, so the RAM never came back; release it for real.
            ms_cache.release_async("dit", "DiT weight cache switched off")
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, ms_split.WRAPPER_KEY,
                               ms_split.make_apply_model_wrapper())
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, ms_split.WRAPPER_KEY,
                               ms_split.make_wrapper(None if second_gpu < 0 else second_gpu, exchange, key,
                                                     cache_ram_reserve_gb, gpu_set=gpus, sparse=sparse_attention,
                                                     vsa=sparse_vsa, vram_blocks=vram_block_cache,
                                                     vram_reserve_gib=vram_reserve_gb,
                                                     exchange_chunks=exchange_chunks))
        if vram_block_cache:
            # NOTHING ELSE CAN FREE THESE: allocations made by this pack are invisible to ComfyUI's
            # model management. ON_CLEANUP fires from a `finally` at the end of outer_sample, so it
            # runs before the VAE decode node and on the interrupt path too.
            m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, ms_split.WRAPPER_KEY,
                                    lambda _patcher: ms_vram.release_all("sampling finished"))
        selection = gpus.describe() if gpus is not None else ms_gpus.GPUSet.legacy(second_gpu).describe()
        log.info("[MultiStream] node configured: model %s on %s, GPUs %s, exchange %s, sparse attention %s "
                 "(vsa %s), dynamic vram %s, vram blocks %s, weight cache %s, RAM reserve %.1f GiB | %s",
                 os.path.basename(key[0]) if key else type(model.model).__name__,
                 model.load_device, selection, exchange, "on" if sparse_attention else "off",
                 "on" if sparse_vsa else "off", dyn, "on" if vram_block_cache else "off",
                 "on" if key else "off", cache_ram_reserve_gb, ram())
        return (m,)


class H3MSTextEncoderCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip":
                            "Off REMOVES this node's hook from the shared model, so nothing of ours runs or logs. BYPASSING or MUTING the node cannot do this -- ComfyUI never calls the node, so the hook from the last run stays attached with its old settings. Use this toggle, or the H3 MultiStream menu's 'Detach all hooks'."}),
                "weight_cache": ("BOOLEAN", {"default": True,
                                 "tooltip": "Keep the text encoder's weights in pinned RAM for the life of the ComfyUI "
                                            "process (~14.6 GB for MiniMax H3's Qwen3-VL-32B). A prompt change after "
                                            "the DiT has evicted the encoder then costs its compute (~10 s) instead of "
                                            "a full re-stage (~68 s)."}),
                "cond_cache_entries": ("INT", {"default": 64, "min": 0, "max": 4096,
                                       "tooltip": "Remember this many encoder outputs (~12 MB each for a 768x1344 "
                                                  "I2VA scene). Re-rendering an encoded scene skips the encoder. "
                                                  "0 = off."}),
                "cache_ram_reserve_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5,
                                         "tooltip": "Only fill the weight cache while at least this much system RAM "
                                                    "would stay available. 0 = use RAM as needed."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = ("Pinned-RAM weight cache and output cache for the text encoder. Works on the shared encoder model, "
                   "so it survives CLIP clones such as Select CLIP Device.")

    def patch(self, clip, enabled=True, weight_cache=True, cond_cache_entries=64, cache_ram_reserve_gb=0.0):
        ms_te_cache.install(clip, enabled, weight_cache, cond_cache_entries, cache_ram_reserve_gb)
        return (clip.clone(),)


class H3MSVAECache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip":
                            "Off REMOVES this node's hook from the shared model, so nothing of ours runs or logs. BYPASSING or MUTING the node cannot do this -- ComfyUI never calls the node, so the hook from the last run stays attached with its old settings. Use this toggle, or the H3 MultiStream menu's 'Detach all hooks'."}),
                "weight_cache": ("BOOLEAN", {"default": True,
                                 "tooltip": "Keep this VAE's weights in pinned RAM for the life of the ComfyUI process "
                                            "(H3: video int8 2.95 GB, audio 0.56 GB), so it does not re-stage after the "
                                            "DiT has evicted it. Place after Select VAE Device."}),
                "cache_ram_reserve_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5,
                                         "tooltip": "Only fill the weight cache while at least this much system RAM "
                                                    "would stay available. 0 = use RAM as needed."}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "advanced/latent"
    DESCRIPTION = "Pinned-RAM weight cache for a VAE. Works on the VAE's model, so place it after Select VAE Device."

    def patch(self, vae, enabled=True, weight_cache=True, cache_ram_reserve_gb=0.0):
        ms_vae_cache.install(vae, enabled, weight_cache, cache_ram_reserve_gb)
        return (vae,)


class H3MSVAESplitDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "second_gpu": ("INT", {"default": -1, "min": -1, "max": 15,
                                       "tooltip": "CUDA index of the second GPU; -1 = automatic. Ignored when a GPU "
                                                  "set is connected."}),
            },
            "optional": {
                "gpus": ("H3MS_GPUS", {"tooltip": "GPU selection from H3 MS GPU Set. Without it: every visible GPU "
                                                  "(or second_gpu)."}),
                "max_gpus": ("INT", {"default": ms_vae_split.DEFAULT_MAX_GPUS, "min": 1, "max": ms_gpus.MAX_GPUS,
                             "tooltip": "At most this many GPUs decode (ComfyUI's process counts as one; each other "
                                        "GPU runs a worker process). The gain flattens after 3-4 GPUs."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "advanced/latent"
    DESCRIPTION = ("Decodes MiniMax H3 video latents with temporal chunks split across GPUs: every GPU besides the "
                   "VAE's own runs a persistent worker process. Place after Select VAE Device. Output matches one GPU. "
                   "Release the workers with H3 MS Release Resources or the H3 MultiStream menu.")

    def patch(self, vae, enabled=True, second_gpu=-1, gpus=None, max_gpus=ms_vae_split.DEFAULT_MAX_GPUS):
        ms_vae_split.install(vae, enabled, None if second_gpu < 0 else second_gpu, gpu_set=gpus, max_gpus=max_gpus)
        return (vae,)


def _prompt_running():
    """True while ComfyUI is executing a prompt (clearing weights or stopping workers then could break it)."""
    try:
        from server import PromptServer
        running, _pending = PromptServer.instance.prompt_queue.get_current_queue()
        return len(running) > 0
    except Exception:
        return False


def _clear_weights(allow_running=False):
    if not allow_running and _prompt_running():
        msg = "a prompt is running; weight caches left in place"
        log.warning("[Release] %s", msg)
        return {"freed_GiB": 0.0, "busy": True, "message": msg}
    ms_te_cache.detach_all()
    freed = ms_cache.clear_all()
    return {"freed_GiB": round(freed / 2**30, 2), "busy": False,
            "message": f"cleared weight caches: {freed / 2**30:.2f} GiB released; they refill on next use"}


def _release_streams(allow_running=False):
    if not allow_running and _prompt_running():
        msg = "a prompt is running; prefetch side streams left in place"
        log.warning("[Release] %s", msg)
        return {"released": 0, "busy": True, "message": msg}
    out = ms_split.release_streams("released from the UI / HTTP")
    out["busy"] = False
    return out


def _clear_outputs():
    n = ms_te_cache.COND_CACHE.clear()
    return {"cleared": n, "message": f"cleared {n} cached text-encoder output(s)"}


def _status():
    return {"gpu_plans": dict(ms_gpus.LAST_PLANS), "last_split_step": dict(ms_split.LAST_STEP),
            "vae_workers": ms_vae_split.worker_status(), "side_streams": ms_split.stream_status(),
            "hooks": ms_hooks.installed(), "vram_blocks": ms_vram.stats(),
            "caches": ms_cache.all_stats(),
            "text_encoder_outputs": ms_te_cache.COND_CACHE.stats(), "ram": ram(), "prompt_running": _prompt_running()}


class H3MSReleaseResources:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "release_vae_workers": ("BOOLEAN", {"default": True,
                                        "tooltip": "Stop the H3 VAE Split Decode worker process(es); they restart on "
                                                   "the next split decode."}),
                "detach_hooks": ("BOOLEAN", {"default": False,
                                 "tooltip": "Remove every text-encoder / VAE hook this pack installed. Use after "
                                            "bypassing or deleting a cache node: ComfyUI never runs a bypassed "
                                            "node, so its hook would otherwise stay attached with old settings."}),
                "release_side_streams": ("BOOLEAN", {"default": False,
                                         "tooltip": "Drop the prefetch side stream held on each GPU the split has "
                                                    "run on; the next split step creates them again."}),
                "clear_weight_caches": ("BOOLEAN", {"default": False,
                                        "tooltip": "Release the pinned DiT / text-encoder / VAE weight caches."}),
                "clear_text_encoder_outputs": ("BOOLEAN", {"default": False,
                                               "tooltip": "Forget cached text-encoder outputs."}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "advanced/model"
    DESCRIPTION = ("Releases H3 MultiStream resources when queued: VAE worker processes, per-GPU prefetch side "
                   "streams, pinned weight caches, cached text-encoder outputs. Runs every time it is queued.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, release_vae_workers=True, detach_hooks=False, release_side_streams=False,
            clear_weight_caches=False, clear_text_encoder_outputs=False):
        lines = []
        if detach_hooks:
            lines.append(ms_hooks.uninstall_all("detached by H3 MS Release Resources node")["message"])
        if release_vae_workers:
            lines.append(ms_vae_split.release_workers("released by H3 MS Release Resources node")["message"])
        if release_side_streams:
            # the split stack has finished by the time a downstream node runs, so synchronizing here is safe
            lines.append(_release_streams(allow_running=True)["message"])
        if clear_weight_caches:
            # this node runs inside a prompt, but nothing else of this prompt uses the caches while it executes
            lines.append(_clear_weights(allow_running=True)["message"])
        if clear_text_encoder_outputs:
            lines.append(_clear_outputs()["message"])
        if not lines:
            lines.append("nothing selected")
        lines.append(ram())
        log.info("[Release] %s", " | ".join(lines))
        return {"ui": {"text": lines}}


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:  # imported outside the server (standalone tests)
        return
    routes = instance.routes

    @routes.get("/h3multistream/status")
    async def h3ms_status(request):
        return web.json_response(_status())

    @routes.post("/h3multistream/vae_workers/release")
    async def h3ms_release_workers(request):
        return web.json_response(ms_vae_split.release_workers("released from the UI / HTTP"))

    @routes.post("/h3multistream/hooks/detach")
    async def h3ms_detach_hooks(request):
        if _prompt_running():
            return web.json_response({"detached": 0, "busy": True,
                                      "message": "a prompt is running; hooks left attached"})
        return web.json_response(ms_hooks.uninstall_all("detached from the UI / HTTP"))

    @routes.post("/h3multistream/streams/release")
    async def h3ms_release_streams(request):
        return web.json_response(_release_streams())

    @routes.post("/h3multistream/cache/release/{group}")
    async def h3ms_release_group(request):
        group = request.match_info["group"]
        if group not in ("te", "dit", "vae"):
            return web.json_response({"error": f"unknown cache group {group!r}; use te, dit or vae"}, status=400)
        freed = ms_cache.release(group, "released from the UI / HTTP")
        return web.json_response({"group": group, "freed_GiB": round(freed / 2**30, 2),
                                  "message": f"released {group} weight cache(s): {freed / 2**30:.2f} GiB"})

    @routes.post("/h3multistream/cache/clear_weights")
    async def h3ms_clear_weights(request):
        return web.json_response(_clear_weights())

    @routes.post("/h3multistream/cache/clear_outputs")
    async def h3ms_clear_outputs(request):
        return web.json_response(_clear_outputs())

    # kept for compatibility with earlier scripts
    @routes.get("/h3multistream/cache")
    async def h3ms_cache_stats(request):
        return web.json_response({"caches": ms_cache.all_stats(), "text_encoder_outputs": ms_te_cache.COND_CACHE.stats()})

    @routes.post("/h3multistream/cache/clear")
    async def h3ms_cache_clear(request):
        out = _clear_weights()
        out.update(_clear_outputs())
        return web.json_response(out)


_register_routes()

NODE_CLASS_MAPPINGS = {
    "H3MultiStream": H3MultiStream,
    "H3MSGPUSet": H3MSGPUSet,
    "H3MSTextEncoderCache": H3MSTextEncoderCache,
    "H3MSVAECache": H3MSVAECache,
    "H3MSVAESplitDecode": H3MSVAESplitDecode,
    "H3MSReleaseResources": H3MSReleaseResources,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MultiStream": "H3 MultiStream",
    "H3MSGPUSet": "H3 MS GPU Set",
    "H3MSTextEncoderCache": "H3 MS Text Encoder Cache",
    "H3MSVAECache": "H3 MS VAE Cache",
    "H3MSVAESplitDecode": "H3 MS VAE Split Decode",
    "H3MSReleaseResources": "H3 MS Release Resources",
}
