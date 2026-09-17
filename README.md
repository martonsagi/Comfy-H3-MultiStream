<!--
SPDX-FileCopyrightText: 2026 Márton Sági
SPDX-License-Identifier: GPL-3.0-only
-->

# ComfyUI-H3-MultiStream

Multi-GPU acceleration and caching for **MiniMax H3** video generation in ComfyUI.

[![First frame of the sample render](docs/demo/h3_multistream_demo1.png)](docs/demo/)

> **Status:** The transformer split runs on **1 to 8 GPUs** and the VAE split decode on up to 8 (4 by default). Both reproduce single-GPU output bit for bit at every tested rank count. Speed is measured on 2- and 3-GPU systems (see [`docs/performance.md`](docs/performance.md)); 4 to 8 GPUs are verified for correctness with rank maps but have not been timed on real machines.

The node pack does four things:

- **Splits the H3 transformer across GPUs.** Each denoising step runs on every selected card (up to 8).
- **Splits the video VAE decode across GPUs** (up to 8, 4 by default).
- **Keeps model weights in pinned system RAM,** so switching between scenes no longer reloads the text encoder, DiT or VAE.
- **Caches text-encoder outputs,** so re-rendering a scene does not run the encoder again.

The transformer split, the VAE split and the caches do not change the computation. On the tested systems the output matched single-GPU ComfyUI bit for bit, checked with per-frame MD5 hashes of the decoded video and the MD5 of the decoded audio; see [`docs/performance.md`](docs/performance.md#what-does-not-change-the-output).

> **MiniMax H3 license notice.** MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, which is not part of this project. That license excludes the European Union, the United Kingdom, the United States and the Republic of Korea. This node pack does not include the model and grants no rights to it. Before using MiniMax H3 with this node pack, users must obtain their own authorization from MiniMax where the license requires it. See [NOTICE](NOTICE).

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Nodes](#nodes)
- [Recommended workflow wiring](#recommended-workflow-wiring)
- [Performance](#performance)
- [Memory requirements](#memory-requirements)
- [Multi-GPU machines and cloud pods](#multi-gpu-machines-and-cloud-pods)
- [Releasing resources](#releasing-resources)
- [Security](#security)
- [Logging](#logging)
- [How it works](#how-it-works)
- [Measurements](#measurements)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Credits](#credits)
- [License](#license)

## Requirements

| Component | Requirement |
|---|---|
| Operating system | Linux. The VAE split decode worker uses AF_UNIX sockets and POSIX shared memory |
| GPUs | NVIDIA CUDA GPUs: two or more for the splits (the transformer split uses up to 8, the VAE split up to 8, 4 by default); one GPU runs unsplit with the caches. Peer-to-peer support is not required |
| ComfyUI | 0.35.0 or newer (native MiniMax H3 and the Model Sparse Attention node). Tested on 0.35.0 with comfy-aimdo 0.5.3 (DynamicVRAM) |
| PyTorch | Tested on 2.13.0+cu130 |
| Python | Tested on 3.12 and 3.13 |
| Model | MiniMax H3 (FL2VA / T2VA / I2VA). Tested with int8 transformer weights (standard and turbo variants), the int8 text encoder (the NVFP4/AWQ text encoder was also tested), an int8 video VAE and the fp32 audio VAE |
| System RAM | See [Memory requirements](#memory-requirements). The caches are optional |

There are no extra Python dependencies. Everything used (`torch`, `psutil`, `tqdm`, `aiohttp`) already ships with ComfyUI.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/martonsagi/Comfy-H3-MultiStream ComfyUI-H3-MultiStream
```

Restart ComfyUI. The nodes appear under `advanced/model`, `advanced/conditioning` and `advanced/latent`. An **H3 MultiStream** entry is added to the main menu.

## Nodes

### H3 MultiStream

`MODEL -> MODEL`. Runs the MiniMax H3 transformer blocks split across the selected GPUs (up to 8). With only one usable GPU, the node installs nothing and the model runs exactly as in plain ComfyUI.

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | | The patched H3 model, after device selection and model patches |
| `enabled` | BOOLEAN | `true` | Pass the model through unchanged when disabled |
| `second_gpu` | INT | `-1` | CUDA index of the second GPU; `-1` is automatic. Ignored when `gpus` is connected |
| `exchange` | `host` / `p2p` | `host` | How activations move between the GPUs. `host` stages through pinned RAM and works everywhere. `p2p` uses direct GPU-to-GPU copies |
| `exchange_chunks` | INT 0-32 | `0` | Split each per-block exchange into this many chunks and overlap the two directions, so a card uploads one chunk while it downloads the next. PCIe is full duplex; the serial path pays D2H + H2D. `0` or `1` keeps the original single-shot exchange. `host` mode only. `8` measured best on the reference system: 1.12x per step, 1.41x on the exchange itself |
| `sparse_attention` | BOOLEAN | `false` | Run ComfyUI's **Model Sparse Attention** node inside the split. Each rank runs the sparse kernel over its own attention heads. Because sol-attn and sla select blocks per head, and a rank holds every token for its heads, the split gives the same result as the sparse node on one GPU. Without this, a sparse patch raises, because it replaces the block's whole attention stage and cannot be passed through as an attention override |
| `dynamic_vram` | `keep` / `off for this model` | `keep` | `off for this model` takes ComfyUI's per-model DynamicVRAM opt-out for the DiT alone. Provided for experimentation; **measured slower on the systems tested, so `keep` is the default** |
| `vram_block_cache` | BOOLEAN | `false` | VRAM as the first tier of the weight cache. **Requires `dynamic_vram` off, and is refused otherwise** -- holding device tensors across sampler steps under DynamicVRAM aborts the process. Provided for experimentation; it did not improve performance on the systems tested |
| `vram_reserve_gb` | FLOAT | `2.0` | VRAM left free per rank GPU beyond the step working set when the VRAM tier is on. Default from `H3MS_VRAM_SAFETY_GIB` |
| `sparse_vsa` | BOOLEAN | `false` | Allow the sparse node's `vsa` method inside the split. Needs a checkpoint carrying `to_gate_compress` weights; correct and ~1.27x, but sol-attn is faster here |
| `weight_cache` | BOOLEAN | `true` | Keep the transformer block weights in pinned RAM for the life of the ComfyUI process |
| `cache_ram_reserve_gb` | FLOAT | `0.0` | Only fill the cache while at least this much system RAM stays available. `0` uses RAM as needed |
| `gpus` (optional) | H3MS_GPUS | | GPU selection from **H3 MS GPU Set**. Without it, every visible GPU is used (or `second_gpu`) |

### H3 MS GPU Set

`-> H3MS_GPUS`. Chooses which GPUs H3 MultiStream and H3 MS VAE Split Decode use. The model's own GPU always takes part as rank 0. The plan is made when the model first runs and is logged with a `[GPUs]` line.

| Input | Type | Default | Description |
|---|---|---|---|
| `gpus` | STRING | `auto` | `auto` for every visible GPU, or a list such as `0,1,3` (rank order after the model's GPU) |
| `exclude` | STRING | empty | GPUs to leave out, e.g. `2` for a card reserved for another service |
| `shares` | STRING | empty | Relative speed per selected GPU, e.g. `1,1,0.7` for a power-capped card. H3 MultiStream sizes each GPU's attention heads and tokens by it; the VAE split divides evenly |
| `min_free_vram_gb` | FLOAT | `0` | Skip GPUs with less free VRAM when the plan is made. Never skips the model's GPU |
| `max_gpus` | INT | `8` | Upper limit; `1` runs unsplit (the text-encoder and VAE caches still work) |

Invalid selections fail when the node runs: an index that does not exist, a duplicate, excluding the model's GPU, or a share count that does not match the GPUs.

### H3 MS Text Encoder Cache

`CLIP -> CLIP`. A pinned-RAM weight cache and an output cache for the text encoder.

| Input | Type | Default | Description |
|---|---|---|---|
| `clip` | CLIP | | The H3 text encoder, after Select CLIP Device |
| `enabled` | BOOLEAN | `true` | Off **removes** this node's hook from the shared encoder. Bypassing or muting the node cannot do this: ComfyUI never runs a bypassed node, so the hook from the last run stays attached with its old settings. Use this toggle, or the menu's *Detach all hooks* |
| `weight_cache` | BOOLEAN | `true` | Keep the encoder weights in pinned RAM, so a prompt change costs only the encoder compute |
| `cond_cache_entries` | INT | `64` | Number of encoder outputs to remember; `0` disables the output cache |
| `cache_ram_reserve_gb` | FLOAT | `0.0` | As above |

Both caches hook the encoder model shared by all CLIP clones, so they keep working after Select CLIP Device and similar nodes.

### H3 MS VAE Split Decode

`VAE -> VAE`. Decodes H3 video latents with the temporal chunks split across GPUs. ComfyUI's process decodes on the VAE's own GPU; every other GPU runs a persistent worker process. Chunks are dealt out in turn, so with N GPUs each decodes about 1/N of them. With one usable GPU the decode runs unsplit.

| Input | Type | Default | Description |
|---|---|---|---|
| `vae` | VAE | | The H3 video VAE, after Select VAE Device |
| `enabled` | BOOLEAN | `true` | Decode on one GPU when disabled |
| `second_gpu` | INT | `-1` | CUDA index of the second GPU; `-1` is automatic. Ignored when `gpus` is connected |
| `gpus` (optional) | H3MS_GPUS | | GPU selection from **H3 MS GPU Set** |
| `max_gpus` (optional) | INT | `4` | At most this many GPUs decode, counting ComfyUI's own. The gain flattens after 3 to 4 GPUs: a 192-frame clip has 11 chunks, and the ordered blend runs on one GPU |

The worker processes start in the background when the node executes, in parallel. They exit together with ComfyUI. Decoded chunks return through POSIX shared memory; when `/dev/shm` is too small for them, as in containers that mount only 64 MB, they return over each worker's socket instead, and a warning is logged.

### H3 MS VAE Cache

`VAE -> VAE`. A pinned-RAM weight cache for a VAE. Place it after Select VAE Device.

This only helps when something evicts the VAE from VRAM between scenes. In the reference workflow at 192 frames, nothing did, so it is not part of the recommended wiring.

| Input | Type | Default | Description |
|---|---|---|---|
| `vae` | VAE | | Video or audio VAE |
| `enabled` | BOOLEAN | `true` | Off **removes** this node's hook from the shared VAE; see the note on the text-encoder cache |
| `weight_cache` | BOOLEAN | `true` | Keep the VAE weights in pinned RAM |
| `cache_ram_reserve_gb` | FLOAT | `0.0` | As above |

### H3 MS Release Resources

Output node that runs every time it is queued. Use it in API workflows or scripts.

| Input | Type | Default | Description |
|---|---|---|---|
| `release_vae_workers` | BOOLEAN | `true` | Stop the VAE split decode worker processes. They restart on the next split decode |
| `detach_hooks` | BOOLEAN | `false` | Remove every text-encoder and VAE hook this pack installed. Use after bypassing or deleting a cache node |
| `release_side_streams` | BOOLEAN | `false` | Drop the per-GPU prefetch side stream; the next split step creates it again |
| `clear_weight_caches` | BOOLEAN | `false` | Release the pinned DiT, text-encoder and VAE weight caches |
| `clear_text_encoder_outputs` | BOOLEAN | `false` | Forget cached text-encoder outputs |

## Recommended workflow wiring

```text
UNETLoader -> (LoraLoaderModelOnly) -> Select Model Device -> model patches ... -> H3 MultiStream -> BasicGuider
                                                                                                        -> BasicScheduler

CLIPLoader -> Select CLIP Device -> H3 MS Text Encoder Cache -> MiniMaxH3ImageToVideo (clip)

VAELoader (video) -> Select VAE Device -> H3 MS VAE Split Decode -> MiniMaxH3ImageToVideo (vae)
                                                                      -> VAEDecode

optional:  H3 MS GPU Set -> H3 MultiStream (gpus)
                         -> H3 MS VAE Split Decode (gpus)
```

In the reference layout, the DiT and the VAEs live on `gpu:1` and the text encoder on `gpu:0`. Without a GPU set, the nodes use the other GPU automatically. To choose GPUs explicitly, connect one **H3 MS GPU Set** to both H3 MultiStream and H3 MS VAE Split Decode.

If you submit workflows through the HTTP API, see [Troubleshooting](#troubleshooting) for the required `preview_method`.

## Performance

Measured under a MiniMax authorization granted to Dynasist Solutions Kft.; see [NOTICE](NOTICE).

The two-GPU figures below were measured with an earlier two-GPU build of this code path; the current version reproduced its 192-frame render bit for bit at the same step times (40 to 42 s). Three-GPU figures are in [Against a single GPU](#against-a-single-gpu).

Reference system:

| Item | Value |
|---|---|
| GPUs | 2x RTX 5060 Ti 16 GB, PCIe gen3 x8 each |
| CPU | Xeon E5-1620 v3 |
| RAM | 104 GB |
| Disk | SATA SSD |
| Workflow | H3 turbo variant, I2VA, 768x1344, 4 sampling steps, CFG 1 |

The split outputs in this section matched the single-GPU reference bit for bit. Sparse attention changes the output; see below.

### Against a single GPU

Three RTX PRO 4000 Blackwell 24 GiB GPUs over PCIe, 243 frames at 1344x768 (75794 tokens), 8 steps,
same seed. The 1-GPU row runs with this pack disabled.

| configuration | s/step | vs 1 GPU |
|---|---|---|
| 1 GPU | 65.0 | 1.00x |
| 3 GPUs | 25.1 | **2.59x** |
| 3 GPUs + `sparse_attention` | 15.1 | **4.30x** |

End to end, text encode and VAE decode included: 597 s -> 185 s, 3.24x. The single 24 GiB card had
about 800 MiB of VRAM left on this job.

### Transformer split (H3 MultiStream)

| Clip | Single GPU sampling | MultiStream sampling | Speedup |
|---|---|---|---|
| 192 frames | 224 s (~56 s/step) | 168 s (~42 s/step) | 1.33x |
| 192 frames, turbo variant | 229 s | 182 s | 1.26x |
| 362 frames | 661 s (~165 s/step) | 424 s (~104 s/step) | 1.56x |

The speedup grows with clip length, because attention cost grows quadratically while the GPU-to-GPU exchange grows linearly. `exchange = p2p` was slower than `host` on the reference system: 189 s vs 168 s at 192 frames. There, bidirectional peer-to-peer copies reach only ~2.4 GiB/s per direction, while pinned-RAM staging uses each card's own link.

Weight prefetch, which is on by default, removes another ~3 s per step at 192 frames: 41.9 s to 38.8 s per transformer pass in the standalone parity test, and 43.2 s to 41.5 s with the turbo variant. It costs about 1 GiB of extra VRAM per GPU at the split peak. Keeping more block weights in VRAM on top of that was also measured and gave no further gain, because the copy no longer adds to the step once it overlaps compute.

### Sparse attention inside the split (`sparse_attention`)

Measured on the reference system, H3 turbo variant, 39975 tokens, 2 ranks, `exchange host`,
weight cache warm. Sampling begins dense because the sparse node's `start_percent` holds it off, so
the dense rows are an in-run baseline on identical work rather than a separate measurement.

| Configuration | s/step | vs dense |
|---|---|---|
| dense | 21.2 | 1.00x |
| **sol-attn** | **14.9** | **1.42x** |
| vsa (checkpoint with `to_gate_compress`, coarse branch) | 16.9 | 1.25x |

sol-attn is recommended. vsa is correct and faster than dense, but slower than sol-attn
here, and it needs a checkpoint carrying `to_gate_compress` -- most MiniMax H3 checkpoints tested do
not have it, and the sparse node silently falls back to its fine stage without the coarse branch on
those.

#### What sol-attn does to the output

Measured frame by frame, 124 frames at 768x1344, fixed seed, H3 turbo variant, with everything except
the sparse window held identical. Compared on the decoded frames before super-resolution.

As a control, a dense run with `exchange_chunks 8` and the same dense run with `exchange_chunks 0`
were bit-identical on all 124 frames, so the pipeline is deterministic and the differences below come
from sol-attn.

| frame | 1 | 8 | 24 | 48 | 96 | 124 |
|---|---|---|---|---|---|---|
| PSNR dB | 32.2 | 25.3 | 18.5 | 14.8 | 16.8 | 17.0 |
| SSIM | 0.972 | 0.912 | 0.716 | 0.539 | 0.613 | 0.652 |

Mean over the clip: PSNR 17.7 dB, SSIM 0.666; no frame is identical.

The first frame, anchored by the reference image, nearly matches. Agreement drops as motion
accumulates and levels off from around frame 48. Viewed side by side, the two renders have the same
character, framing, palette and background; the main visible difference is the timing of motion, for
example where a hand is in a gesture at a given frame. No blur, ghosting or artefacts were seen in one
render and not the other.

In practice:

* Composition and styling are usually kept; motion timing changes.
* sol-attn is deterministic: the same seed reproduces the same output.
* A dense render cannot be reproduced by enabling sol-attn afterwards, so choose the mode before
  searching for a seed.

The exchange time does not change with sparsity (5.1 to 5.6 s per rank in the rows above), so it
becomes a larger share of the shorter step; `exchange_chunks` below reduces it. On a single card,
sparse attention measured 22.0 s/step against 34.0 s dense, a similar gain to the two-GPU split, and
the two combine.

The first sparse step of a run is slower (16.7 s here) while `kmean`/`vscale` are computed, and filling
the DiT weight cache adds about 55 s to the first step of a new process.

### Text encoder (H3 MS Text Encoder Cache)

| Case | Text-encoder time |
|---|---|
| Prompt change after the DiT evicted the encoder, uncached | 67 s (in service: 74 to 90 s) |
| Same, weight cache warm | 12.5 s (in service: ~13 s) |
| Re-render of an already encoded scene (output cache hit) | 0.2 s (in service: ~1 s) |

Measured with the 15 GB NVFP4/AWQ text encoder. Only ~10 s of the uncached time is encoder compute; the rest is ComfyUI re-staging the encoder into VRAM. With the 27 GB int8 text encoder, a warm encode (weight cache on) took 12.0 s, and filling its weight cache took about 118 s on first use.

### VAE decode (H3 MS VAE Split Decode)

| Case | Time |
|---|---|
| Video decode, 90 frames, single GPU | 19.3 s |
| Video decode, 90 frames, split | 12.3 s |
| Decode + audio + mux per 192-frame scene, in service | 51 s -> 34 s |

### End to end

A 192-frame scene after a prompt change drops from about 400 s (single GPU, no caches) to about 216 s with MultiStream, the text-encoder cache and the split decode.

### The split is bit-exact

Verified end to end on three GPUs: the same prompt and seed rendered twice, once with this pack
disabled on a single GPU, once across three ranks with head groups 18/19/19, the
per-block exchanges, the shadow cast and `exchange_chunks 8`.

    243 frames compared -- 243/243 IDENTICAL, SSIM 1.000000, max pixel delta 0/255

A render approved on one GPU therefore comes out the same on the split. `sparse_attention` is an
approximation and gives a different sample.

### Pipelining the exchange (`exchange_chunks`)

Each card has its own full-duplex link to host RAM, but the exchange used only one direction at a
time: stage everything down, barrier, fetch everything up. Chunking it lets chunk *j*'s upload
overlap chunk *j+1*'s download.

Same run conditions as the table above, three arms back to back in one process.
Steady-state steps only -- the first step pays the weight-cache fill, and the first sparse step pays
the `kmean`/`vscale` warm-up.

| `exchange_chunks` | exchange, per rank | dense s/step | sol-attn s/step | vs serial |
|---|---|---|---|---|
| `0` (default) | 5.3 / 5.1 | 21.2 | 14.9 | 1.00x |
| `4` | 4.0 / 3.8 | 19.9 | 13.5 | 1.10x |
| **`8`** | **3.8 / 3.6** | **19.7** | **13.3** | **1.12x** |

Combined: dense 21.2 s/step, `sparse_attention` 14.9, plus `exchange_chunks 8` 13.3, 1.59x in total.

The transfer itself is 1.41x faster, and the step saves about the same time. The step gain is below
the ~1.22x a full-duplex model predicts, most likely because staging is a pinned-host copy that also
uses host memory bandwidth, and both ranks stage at the same time. 4 and 8 chunks differ by 0.2 s;
higher counts were not measured.

The chunked exchange moves the same bytes in the same order with a different schedule; the tests check
that it is bit-identical to the serial path at 2, 3, 5 and 8 ranks. It is off by default because it adds
concurrency to the hot path, and the other measurements here used the serial exchange.

## Memory requirements

| Item | System RAM | Optional |
|---|---|---|
| Exchange buffers (pinned RAM) | ~0.5 GB per GPU rank at 192 frames | no |
| DiT weight cache (H3 int8 checkpoint) | 18.0 GiB | yes |
| Text-encoder weight cache | 25.3 GiB (int8), 14.6 GiB (NVFP4/AWQ) | yes |
| VAE weight cache (int8 video + fp32 audio) | 3.5 GiB (5.4 GiB with an fp16 video VAE) | yes |
| Text-encoder output cache | ~12 MB per 768x1344 scene | yes |
| Worker processes for the split decode | A CUDA context on each additional GPU; the VAE is loaded only during a decode | no |

ComfyUI and MiniMax H3 need substantial RAM of their own. The decoded frames of a 362-frame clip alone take about 11 GB.

| System RAM | Guidance |
|---|---|
| 32 GB | MultiStream and the split decode with all weight caches disabled |
| 64 GB | DiT cache plus the NVFP4/AWQ text-encoder cache, or the DiT cache alone with the int8 text encoder, for clips around 192 frames |
| 96 GB or more | All caches with the int8 text encoder. With both caches pinned, about 25 GiB of 100 GiB stayed available at 192 frames, so watch RAM on long clips |

Each cache refuses to fill when that would leave less than `cache_ram_reserve_gb` of RAM available. On machines near these limits, set a reserve so that long clips keep headroom for decoding.

## Multi-GPU machines and cloud pods

Checklist for machines with more than two GPUs, such as rented multi-GPU pods:

| Check | How | Why |
|---|---|---|
| GPU interconnect | `nvidia-smi topo -m` | On NVLink (`NV#`), try `exchange = p2p`. On PCIe, keep `host`: bidirectional peer-to-peer copies over PCIe were slower than pinned-RAM staging. Avoid GPU pairs marked `SYS` (different CPU sockets) |
| Shared memory | `df -h /dev/shm` | Docker mounts 64 MB by default. The VAE split then falls back to socket transfer; `--shm-size=8g` restores shared memory |
| System RAM | pod specification | The DiT and text-encoder weight caches pin about 43 GB with the int8 text encoder (about 33 GB with NVFP4/AWQ). On small pods, disable them or set `cache_ram_reserve_gb` |
| GPU architecture | first test render | The int8 and NVFP4 kernels were verified on NVIDIA Blackwell only. Run `tests/parity_aimdo.py` once on other architectures before relying on the output |
| Unequal or throttled GPUs | `nvidia-smi -q -d PERFORMANCE` | Every rank waits for the slowest one. Give slower cards a smaller share in **H3 MS GPU Set** |
| License | your MiniMax authorization | The MiniMax H3 Community License excludes several territories. Check that your authorization covers the data centre's location |

## Releasing resources

| Method | How |
|---|---|
| Menu | Main menu, **H3 MultiStream**: Release VAE workers, Clear weight caches, Clear text-encoder output cache, Show status |
| Node | **H3 MS Release Resources**, queued like any output node |
| HTTP | See the endpoints below |

| Method | Endpoint | Effect |
|---|---|---|
| `GET` | `/h3multistream/status` | GPU plans, the last split step (ranks, heads per rank, timings), worker processes, weight caches, output cache statistics, available RAM |
| `POST` | `/h3multistream/vae_workers/release` | Stop idle VAE worker processes |
| `POST` | `/h3multistream/cache/clear_weights` | Release all pinned weight caches |
| `POST` | `/h3multistream/cache/clear_outputs` | Clear the text-encoder output cache |

Safety rules:

- A worker release is refused while a split decode is running.
- Clearing weight caches from the menu or HTTP is refused while a prompt is running.
- Released workers and caches are recreated automatically the next time they are needed.

## Security

- **HTTP endpoints.** The `/h3multistream/*` endpoints have no authentication of their own, like the rest of ComfyUI's API. Anyone who can reach the ComfyUI port can read the status and release workers or caches. ComfyUI rejects cross-site browser requests, but not direct requests from other machines. Keep ComfyUI on `127.0.0.1` or a trusted network, and do not expose a `--listen` instance to the internet without an authenticating reverse proxy.
- **Worker processes.** Each VAE worker connects over an AF_UNIX socket in a private per-process directory (mode 0700) and authenticates with a random key generated for each start. Decoded frames return through POSIX shared memory segments (mode 0600) that are removed after reading, or over that authenticated socket when `/dev/shm` is too small.
- **No network access.** The node pack makes no network requests and downloads nothing. Model files are read through ComfyUI's own loaders.

## Logging

All messages go to the ComfyUI console through the `h3_multistream` logger. Each line starts with `[H3 MultiStream]`, followed by a subsystem tag: `[MultiStream]`, `[GPUs]`, `[Cache]`, `[TextEncoder]`, `[VAE]`, `[VAE split]` or `[Release]`.

At the default INFO level you get:

- **Node configuration and activation:** the GPU plan (ranks, selection, anything dropped and why, free VRAM), captured per-block patches, cache state.
- **One line per sampling step:** tokens, wall time, exchange time per rank, data moved between ranks, weight cache and prefetch state, the step split peak per GPU, and driver-level VRAM in use per GPU. The step split peak is the split's peak PyTorch allocation; it is reset every step, so it stays constant for a fixed resolution and clip length. The driver-level figure includes staged weights, other models and the VAE worker.
- **Cache events:** creation, fill, eviction and disabling, with sizes and available RAM.
- **Text-encoder output cache:** hits and misses, with encode time and token count.
- **VAE split decode:** chunk plan and timings, worker start and load times. On a worker failure, the last lines of the worker log.

| Environment variable | Effect |
|---|---|
| `H3MS_DEBUG=1` | Per-block and per-chunk debug lines |
| `H3MS_PREFETCH=0` | Turn off the weight prefetch (for comparisons; the output is identical either way) |
| `H3MS_VAE_WORKER_START_TIMEOUT` | Seconds to wait for a VAE worker to start and load (default `300`) |
| `H3MS_VAE_TRANSPORT` | `auto` (default), `shm` or `bytes`: how decoded VAE chunks return from the workers |

The VAE worker writes its own timestamped log to `vae-worker-gpu<N>.log` in a private temporary directory (`<tmp>/h3ms-<pid>-*/`, mode 0700). The console shows the path when the worker starts, and **Show status** lists it. The directory is removed when ComfyUI exits.

## How it works

### Transformer split

H3 packs text, conditioning, audio and video tokens into one sequence. Each GPU (rank) owns a contiguous range of tokens and a contiguous group of attention heads, both sized by its share. With equal shares, the 56 heads split 28/28 on two GPUs, 18/19/19 on three and 7 each on eight. For each block, the ranks work in five steps:

1. **Local normalization.** Each rank applies the block's normalization and modulation to its own tokens.
2. **Hidden-state all-gather.** Every rank receives the other ranks' hidden states, so each can project all tokens.
3. **Split projection.** Each rank computes only its own head group. It slices the query/key/value projection rows of the int8 weight, which keeps the result exact.
4. **Attention and all-to-all.** Attention runs over the full sequence for those heads. Each rank then receives every other head group's output for its own tokens.
5. **Local finish.** The output projection and MLP run locally.

Implementation details:

- **Threads:** one thread per rank. With `exchange = p2p`, peer access is enabled once for every GPU pair before the threads start.
- **Kernels:** ComfyUI's own kernels (comfy-kitchen int8 linear, fused RMSNorm/RoPE, the configured attention backend) are used throughout.
- **Hook:** the node hooks in as a `DIFFUSION_MODEL` wrapper. H3's own forward still handles embedding, layout, RoPE and the final layer.
- **Attention overrides:** per-block overrides from other nodes are captured and replayed on every rank.
- **Weight prefetch:** while a block computes, each GPU copies the next block's weights from the pinned cache on a separate CUDA stream, so the copy overlaps compute instead of adding to the step. It holds about one extra block of weights per GPU and needs the weight cache. `H3MS_PREFETCH=0` turns it off.

### Weights and DynamicVRAM

ComfyUI's DynamicVRAM (comfy-aimdo) keeps per-module cast state and maps its GPU memory for a single device. The node therefore:

- builds per-GPU shadow modules that share the parameter objects;
- casts weights through a path that mirrors ComfyUI's dequantize, LoRA and requantize sequence, including the stochastic-rounding seed and the model dtype;
- moves any tensor it did not allocate between GPUs through host RAM;
- disables the ComfyUI model compiler only for the split model call;
- silences comfy-aimdo's native logging while the rank threads run.

The weight caches copy each model's host weights once into CPU memory registered with CUDA (exact size, no power-of-two rounding). They are keyed by file path, size, modification time and model options. They are invisible to ComfyUI's model management, so they survive prompt changes and model reloads.

### VAE split decode

The H3 video VAE decodes overlapping temporal chunks independently and blends them in order. With N GPUs, chunk `i` belongs to GPU rank `i % N`.

1. ComfyUI's process sends each worker its chunks as one job up front.
2. It decodes its own chunks (rank 0) itself.
3. It replays the upstream blend-and-write loop in chunk order, taking each worker's chunks as they arrive (shared memory, or the worker's socket when `/dev/shm` is too small).

A separate process is required: with two threads in one process, the Python global interpreter lock starved both GPUs and the split was slower than a single GPU.

## Measurements

[`docs/performance.md`](docs/performance.md) has the measured numbers: multi-GPU scaling against a
single GPU, sparse attention, the pipelined exchange, the VAE decode split, `host` against `p2p`, and
what each of them does or does not change about the output. [`docs/demo/`](docs/demo/) has a sample
render produced on three GPUs.

## Troubleshooting

**The process aborts with `aimdo memory compile error` during the first sampling step of an API-submitted prompt.**
With `--preview-method taesd`, H3 previews use the TAEHV decoder, which can crash under DynamicVRAM. The ComfyUI web UI sends its own preview setting; API clients must send it explicitly:

```json
{"prompt": { ... }, "extra_data": {"preview_method": "latent2rgb"}}
```

**`unsupported dit patches` or `block N carries a patch that replaces the block computation`.**
MultiStream supports per-block patches that only adjust `transformer_options`, such as a per-block attention backend. Patches that replace the block computation cannot be split. Disable them, or bypass MultiStream for that workflow.

If the patch is ComfyUI's **Model Sparse Attention** node, turn on `sparse_attention` instead: that node replaces the block's whole attention stage, so it cannot be passed through as an attention override, but each rank can run its kernel over its own heads. The error message says so when the switch is off.

**`the Model Sparse Attention node is set to 'vsa'` ... `turn on sparse_vsa`.**
The `vsa` method needs both the `sparse_vsa` switch and a checkpoint carrying `to_gate_compress`. Most MiniMax H3 checkpoints do not have it. On those the sparse node logs `VSA: the model has no to_gate_compress layers` and runs its fine stage without the coarse branch, which is not the full VSA method. `sol-attn` was faster on the tested systems.

**`vram_block_cache needs dynamic_vram='off for this model'` in the console.**
Deliberate, and the VRAM tier stays off. Device tensors this pack allocates cannot be reused across sampler steps under DynamicVRAM -- the second step raises an illegal memory access and aborts the process. Both switches exist for experimentation and neither improved performance on the systems tested, so the defaults leave them off.

**A cache node still logs after you bypassed or muted it.**
This is expected: the text-encoder, VAE cache and VAE split nodes hook methods on the shared `cond_stage_model` / `first_stage_model`, which ComfyUI caches across prompts. ComfyUI never runs a bypassed node, so nothing gets the chance to remove the hook, and it keeps running with the settings from the last prompt that did run it. Use the node's own `enabled` input, which removes the hook, or **Detach all hooks** in the H3 MultiStream menu. `GET /h3multistream/status` lists what is currently attached under `hooks`.

**`VAE worker on GPU N did not connect` or `exited with ...`.**
Check the worker log; its path is printed when the worker starts and listed by **Show status**. The last lines are also printed to the console. The worker starts again on the next decode. Release it manually from the menu or with the H3 MS Release Resources node.

**`chunks return over the worker sockets instead of shared memory` in the console.**
`/dev/shm` has too little free space for the decoded chunks, which is typical in Docker (64 MB by default). The decode still works and stays bit-identical, just a little slower. To use shared memory, give the container more: `docker run --shm-size=8g`, or in Kubernetes an `emptyDir` with `medium: Memory` mounted at `/dev/shm`.

**`weight cache DISABLED` in the console.**
Filling the cache would have left less RAM available than `cache_ram_reserve_gb`. The node keeps working without the cache (weights stream from the model file). Lower the reserve, free RAM, or disable the cache for that model.

**`exchange_chunks` changes nothing, and the step line still reads `exchange host:` without `xN`.**
Pipelining is host mode only -- `p2p` hands the device tensors over untouched, so there are no two
directions to overlap and the setting is ignored. It is also skipped for a transfer below the chunk
threshold, where the extra barriers would cost more than the overlap. When it is active the step line
reads `exchange host x8:`.

**The first render after starting ComfyUI is slower.**
The weight caches fill on first use, and the VAE worker process starts when the node first executes. Later renders use the warm caches.

## Limitations

- **Model:** MiniMax H3 only. The transformer split relies on the H3 block structure and batch size 1.
- **GPUs:** the transformer split uses up to 8 GPUs and needs at least one attention head per GPU (H3 has 56). The VAE split decode uses up to `max_gpus` (default 4) of the selected GPUs. One GPU runs unsplit.
- **Loaders:** the caches need a loader that records a single-file reload factory. The core `UNETLoader`, `CLIPLoader` and `VAELoader` do.
- **Platform:** the VAE split decode requires Linux.
- **ComfyUI internals:** the VAE split decode mirrors the chunk planning and blending of ComfyUI's H3 video VAE, and MultiStream mirrors ComfyUI's weight-cast semantics. After a ComfyUI update, re-run the parity tests before relying on bit-identical output.
- **Bundled library internals:** the pack uses private helpers from comfy-kitchen and comfy-aimdo (see [Dependencies and code provenance](#dependencies-and-code-provenance)). A ComfyUI update that bumps those packages can break it or, for the int8 kernel, fall back to a slower path.
- **CFG:** H3 runs at CFG 1, so ComfyUI's core MultiGPU CFG Split offers no benefit. MultiStream parallelizes a single model pass instead.

## Tests

The scripts in `tests/` run outside the ComfyUI service and need the GPUs idle. The `_aimdo` variants initialize comfy-aimdo the same way ComfyUI's `main.py` does.

The scripts take your own model files: `--ckpt` for the transformer tests (an absolute path or a file name under `models/diffusion_models`), and `--lora`, `--image` or `--prompt-json` where a script needs them. Run a script with `--help` for its options.

| Script | Purpose |
|---|---|
| `parity.py` | Transformer split vs single GPU, legacy model patcher (optional turbo LoRA) |
| `parity_aimdo.py` | Transformer split vs single GPU under DynamicVRAM: pairwise output comparison, `host` / `p2p`, weight cache, prefetch (`_nopf`), 1-GPU plan (`_1gpu`), LoRA |
| `gpu_set_resolve.py` | GPU selection: parsing, validation and rank plans for 1–8 GPUs (CPU only) |
| `cast_hook_chain.py` | The weight-cast hook chains with another pack's hook in every install order (CPU only) |
| `block_probe_aimdo.py` | Per-submodule comparison of original and shadow modules for one block |
| `weight_probe_aimdo.py` | Shadow weight copies vs ComfyUI's own cast |
| `alloc_thread_repro.py` | Two-thread allocator stress under comfy-aimdo |
| `te_bench_aimdo.py` | Text-encoder timing and cache parity |
| `vae_bench_aimdo.py` | VAE encode/decode timing and cache parity |
| `vae_split_bench_aimdo.py` | VAE split decode vs single-GPU decode, timing and parity. `--cases 0,0+1,0+0+1,bytes@0+1` lists worker GPUs per case; a GPU may repeat or be the VAE's own, so more ranks than physical cards can be tested |
| `tile_batch_bench_aimdo.py` | Experiment: batched spatial tiles on one GPU (no speedup) |

These need no GPU and no checkpoint -- run them after any change to the split:

| Script (CPU only) | Purpose |
|---|---|
| `exchange_pipelining.py` | The chunked exchange is bit-identical to the serial one at 2, 3, 5 and 8 ranks and uneven splits, D2H(j+1) is issued before H2D(j), ranks proposing different chunk counts agree instead of deadlocking, `c <= 1` and sub-threshold tensors take the single-shot path, and `abort()` breaks every barrier |
| `sparse_head_split_parity.py` | Per-rank sparse attention, reassembled, equals the whole-tensor reference for sol-attn, sla, sinks, token_aug and vsa with the coarse gate |
| `sparse_capture.py` | The sparse node's patch is read out of its closure correctly, and an unrecognised patch is refused rather than silently ignored |
| `sparse_vsa_plan.py` | The vsa plan and padded rope are hoisted once per rank per step, bypassing upstream's single-slot cache |
| `rows_linear_hoist.py` | Per-chunk weight resolution is hoisted out of the attention loop, and a linear that is not a shadow is refused |
| `hook_lifecycle.py` | Hooks install, uninstall and survive a model being garbage collected; uninstall refuses under a foreign wrapper |
| `cache_release.py` | The weight cache will not close while a reader holds it, and releases cleanly afterwards |
| `staged_size_accounting.py` | Quantized tensors are sized by their packed bytes, not their dequantized shape |
| `weight_cache_tiering.py` | Block tiering across the VRAM and host tiers |
| `vram_block_cache.py` | The VRAM residency budget, its idempotent plan and the peak accounting that must not count the cache itself |
| `free_rank_devices.py` | Freeing rank GPUs requests the whole card rather than stopping at the first satisfied byte |
| `dynamic_vram_optout.py` | The per-model DynamicVRAM delegate, and the interlock that refuses the VRAM tier while the model is still dynamic |
| `gpu_set_resolve.py`, `cast_hook_chain.py` | Listed above |

Example:

```bash
python custom_nodes/ComfyUI-H3-MultiStream/tests/parity_aimdo.py --ckpt <h3-dit>.safetensors --modes single,host,host_cache
```

`H3MS_UNSAFE=1` with `parity_aimdo.py --modes unsafe` reproduces the behaviour before the DynamicVRAM fixes (no allocator graph pause, direct peer-to-peer copies) and is expected to crash. It exists for that test only; never set it for ComfyUI.

`parity_aimdo.py` modes accept a rank map, so more ranks than physical GPUs can be tested: `host_cache@1+0+1+0` runs four logical ranks on two GPUs (rank 0 must be the model's GPU), and a suffix such as `~1+1+0.5` sets shares. Correctness for 2 to 8 ranks is verified this way; speed needs real GPUs.

## Repository layout

```text
__init__.py              node classes, HTTP routes
multistream/split.py      transformer split, DIFFUSION_MODEL / APPLY_MODEL wrappers
multistream/gpus.py       GPU selection and rank plans
multistream/cast.py       shadow modules and weight cast
multistream/cache.py      pinned-RAM weight cache
multistream/te_cache.py   text-encoder weight and output caches
multistream/vae_cache.py  VAE weight cache
multistream/vae_split.py  VAE split decode (ComfyUI side, worker management)
multistream/vae_worker.py VAE split decode worker process
multistream/log.py        logger, prefix, progress-bar-safe line breaks
web/h3multistream.js      main menu commands
tests/                   parity tests and benchmarks
```

## Credits

This pack builds on, was tested with, or was measured against the following projects. Their licenses
apply to them; nothing from them is redistributed here beyond what ComfyUI's license covers (see
[License](#license)).

**Built on**

| Project | Used for |
|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | The host application: model management, DynamicVRAM, the MiniMax H3 model implementation, the sampling pipeline, and the Model Sparse Attention node that `sparse_attention` runs inside the split |
| [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo) | ComfyUI's dynamic model offloader (DynamicVRAM), which the weight cache and weight streaming work with |
| [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | ComfyUI's kernel library: the int8 and NVFP4 linear kernels, fused RMSNorm/RoPE, and the sparse attention kernels |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensors, CUDA streams, pinned memory and the allocator |
| [aiohttp](https://github.com/aio-libs/aiohttp), [psutil](https://github.com/giampaolo/psutil), [tqdm](https://github.com/tqdm/tqdm) | HTTP routes, RAM accounting, progress reporting (all shipped with ComfyUI) |
| MiniMax H3, by MiniMax | The video model this pack accelerates. Not included; see the license notice above and [NOTICE](NOTICE) |

**Methods**

| Project | Credit |
|---|---|
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) (DeepSpeed-Ulysses) | Ulysses sequence parallelism, the all-to-all head/sequence split the transformer split is based on |
| [xDiT](https://github.com/xdit-project/xDiT) | Reference for sequence parallelism in diffusion transformers, and the communication comparison of Ulysses, Ring, PipeFusion and DistriFusion used when choosing what to build |
| Sol-Attn and SLA-style top-k selection, as implemented in comfy-kitchen | The block-sparse attention methods behind `sparse_attention` |
| [FastVideo](https://github.com/hao-ai-lab/FastVideo) | VSA (video sparse attention), the method behind `sparse_vsa`, and a checkpoint carrying its weights used to test it |

**Compared against**

| Project | How |
|---|---|
| [Raylight](https://github.com/komikndr/raylight) with xDiT's xFuser | Ulysses and FSDP multi-GPU runs on the 2-GPU reference system; see [Performance](docs/performance.md) |
| [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) | Weight placement across devices (DisTorch), assessed against measured link speeds |
| [NCCL](https://github.com/NVIDIA/nccl) | Collective bandwidth measured over PCIe as the baseline for the host-staged exchange |

**Tested on**

| Provider | Used for |
|---|---|
| [Runpod](https://www.runpod.io) | The 3-GPU reference system (3x RTX PRO 4000 Blackwell) |

### Dependencies and code provenance

The pack has no dependencies beyond ComfyUI. Every import is either the Python standard library or something
ComfyUI itself installs:

| Import | Comes from |
|---|---|
| `comfy`, `comfy_extras`, `nodes`, `server` | ComfyUI |
| `torch` | ComfyUI's requirements |
| `comfy_aimdo` | ComfyUI's requirements (`comfy-aimdo`) |
| `comfy_kitchen` | ComfyUI's requirements (`comfy-kitchen`) |
| `aiohttp`, `psutil`, `tqdm` | ComfyUI's requirements |
| `web/h3multistream.js` | ComfyUI's frontend `app.js` and `api.js` only |

The pack does use internals of those packages, which are not stable APIs: the private
`_dtype_code` helper from `comfy_kitchen.tensor.int8` and `comfy_aimdo.control`, and one test imports
comfy-kitchen's `sol_attn` kernel directly. If `_dtype_code` or the `int8_linear` kernel is missing, the
sliced int8 projections fall back to dequantize + linear and a warning is logged; that path is slower
and may not match single-GPU output bit for bit. A ComfyUI update that moves to newer comfy-kitchen or
comfy-aimdo versions can still break the other internals; see [Limitations](#limitations).

Code derived from ComfyUI (GPL-3.0) is marked with a `Contains logic derived from ComfyUI` header in
each file that has it:

| File | Derived from |
|---|---|
| `multistream/split.py` | The MiniMax H3 transformer block and attention computation (`comfy/ldm/minimax/model.py`), reimplemented per rank |
| `multistream/cast.py` | The DynamicVRAM weight post-cast semantics (`comfy/ops.py`) |
| `multistream/vae_split.py` | The MiniMax H3 video VAE temporal chunk plan and blend/write loop (`comfy/ldm/minimax/vae.py`) |
| `tests/tile_batch_bench_aimdo.py` | The MiniMax H3 video VAE tiled-decode blend logic (`comfy/ldm/minimax/vae.py`) |

The pack is therefore licensed GPL-3.0, like ComfyUI.

No code is copied from the other projects credited above. comfy-kitchen (Apache-2.0) and
comfy-aimdo are imported, not copied. The Ulysses split is implemented on ComfyUI's H3 model code, not
taken from DeepSpeed, xDiT, xFuser or Raylight; those are credited for the method and as comparison
points. This was checked by searching the sources and the development history for copied-code markers
and for identifiers from those projects; such a search cannot rule out unattributed copying, but found
none.

## License

Copyright (C) 2026 Márton Sági.

Licensed under the [GNU General Public License v3.0](LICENSE) (`GPL-3.0-only`), the same license as ComfyUI.

This node pack imports ComfyUI internals and contains logic derived from ComfyUI's code, so it is distributed under ComfyUI's license:

- the MiniMax H3 video VAE chunk plan and blend/write loop (`multistream/vae_split.py`);
- the DynamicVRAM weight post-cast semantics (`multistream/cast.py`);
- the H3 transformer block and attention computation, reimplemented per rank (`multistream/split.py`);
- the tiled-decode blend logic in the tile batching experiment (`tests/tile_batch_bench_aimdo.py`).

MiniMax H3 itself (model weights, documentation and related materials) is not included in or distributed with this project. It is licensed separately under the MiniMax H3 Community License Agreement, whose license grant excludes certain territories. Users are responsible for obtaining and using MiniMax H3 in accordance with that license. The required MiniMax H3 license notice is in [NOTICE](NOTICE).
