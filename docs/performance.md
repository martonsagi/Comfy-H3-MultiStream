<!--
SPDX-FileCopyrightText: 2026 Márton Sági
SPDX-License-Identifier: GPL-3.0-only
-->

# Measured performance

All figures below were measured on real hardware with MiniMax H3, using the node defaults except
where a row says otherwise. Comparisons hold the seed, resolution, frame count and step count fixed;
where more than one setting differs between rows, the section says so.

Unless a section says otherwise, step times are steady state: the first step of a run is excluded
because it includes one-off costs (weight cache fill, sparse plan construction).

## Test systems

| | **A — 2 GPUs** | **B — 3 GPUs** |
|---|---|---|
| GPUs | 2x RTX 5060 Ti 16 GiB | 3x RTX PRO 4000 Blackwell 24 GiB |
| link | PCIe gen3 x8, no NVLink | PCIe, no NVLink |
| workload | 75794 tokens (243 frames, 1344x768) | 75794 tokens (243 frames, 1344x768) |
| schedule | 8 steps | 8 steps |

## Multi-GPU scaling — system B

Identical workflow, identical seed. The 1-GPU row has this pack disabled.

| configuration | s/step | vs 1 GPU |
|---|---|---|
| 1 GPU | 65.0 | 1.00x |
| 3 GPUs | 25.1 | **2.59x** |
| 3 GPUs + `sparse_attention` | 15.1 | **4.30x** |

End to end, including text encode and VAE decode: **597.3 s -> 184.6 s, 3.24x.**

The whole-job gain is lower than the per-step gain: two of the eight steps run dense because the
sparse node's `start_percent` holds it off at the start of the schedule, and the text encode, VAE
decode and video encode do not get faster with more GPUs.

The single 24 GiB card used 23648 of 24467 MiB on this job.

## Multi-GPU scaling — system A

The same job as system B — 243 frames at 1344x768, 75794 tokens, 8 steps, the same seed — on the two
16 GiB cards, with a different int8 H3 checkpoint in its 8-step turbo variant, so absolute times
do not compare across systems. The 1-GPU row has this pack disabled. The 1-GPU and sparse
rows ran back to back in one ComfyUI process; the dense row ran the next day after a reboot.

| configuration | sampler | s/step | vs 1 GPU |
|---|---|---|---|
| 1 GPU | 693.6 s | 86.7 (mean, all 8 steps) | 1.00x |
| 2 GPUs + `H3MSVAESplitDecode`, dense | 409.1 s | **49.3** | **1.70x** |
| 2 GPUs + `H3MSVAESplitDecode` + `sparse_attention` | 307.7 s | **29.3** | **2.25x** |

All split rows use `exchange host`, `exchange_chunks 8`. Sampler time is the sum of all eight steps,
first step included, so the three rows are directly comparable; the dense row's first step is 64.8 s,
and the sparse row runs its first two steps dense (74.1 and 53.4 s) because of `start_percent 0.2`.

End to end, with the text encoding cached in both runs: **814 s on one GPU -> 413.2 s on two GPUs with
sparse attention, 1.97x.** The VAE decode split took 73.5 s there.

A single 16 GiB card runs this job by streaming weights: ComfyUI stages the ~20 GiB model through
dynamic VRAM. On two cards the dense step is 1.76x faster than the single-GPU mean.

The dense 2-GPU row was measured with the GPU clocks capped at 3000 MHz; the other two rows at stock
settings. For comparison, the dense steps at stock settings in the sparse run took 53.4 s.

## Compared with other multi-GPU approaches — system A

Other ways to use a second GPU for MiniMax H3, tried on system A. Each row is compared with a
single-GPU run measured alongside it, on its own workload. The workloads are not identical across
rows, so compare the right-hand column rather than absolute times.

| approach | workload | result vs 1 GPU |
|---|---|---|
| Raylight: xDiT/xFuser Ulysses over NCCL | 192 frames, 4 steps | **1.45x slower** (446 s vs 308 s, excluding text encoding) |
| Raylight: FSDP with CPU offload + Ulysses, PCIe peer-to-peer enabled | 192 frames, 4 steps | **1.35x slower** (417 s vs 308 s) |
| ComfyUI core model placement (`SelectModelDevice` and related) | — | Places whole models on a card; no transformer speedup |
| ComfyUI core `MultiGPU CFG Split` | — | Splits the conditional and unconditional passes; no gain at CFG 1, which is how the distilled / turbo H3 workflows run |
| Parking weights on the second card (e.g. ComfyUI-MultiGPU DisTorch) | — | Spreads VRAM, not compute. Not run end to end; expected slower than one GPU here, since card-to-card copies measured 4.84 GB/s against 6.47 GB/s from pinned host RAM, which ComfyUI already streams weights from |
| xDiT PipeFusion / DistriFusion | — | Not applicable: they rely on warm-up steps that a 4-8 step schedule does not have |
| **This pack, dense** | 192 frames, 4 steps | **1.33x faster** (sampling 224 s -> 168 s; earlier build, without `exchange_chunks`) |
| **This pack, dense** | 243 frames, 8 steps, 75794 tokens | **1.70x faster** (86.7 -> 49.3 s/step) |
| This pack, dense (system B, for reference) | 3 GPUs, 243 frames, 8 steps, 75794 tokens | 2.59x faster (65.0 -> 25.1 s/step) |

Raylight and this pack both use Ulysses sequence parallelism; they differ in transport and memory
handling. On system A an H3 step exchanges about 67 GB at 192 frames, and NCCL's
all-to-all measured 3.7 GB/s over PCIe gen3 x8, so the exchange alone took longer than a single-GPU
step. This pack stages the exchange through pinned host memory over each card's own link, can overlap
upload and download (`exchange_chunks`), and keeps block weights in a pinned-RAM cache inside
ComfyUI's dynamic VRAM, instead of sharding them with FSDP (which ran out of memory at 192 frames
without CPU offload).

What these figures do and do not show:

- Sparse attention is excluded. It speeds up a single GPU too, so it is not part of the multi-GPU
  gain; the dense rows are the scaling figures.
- The single-GPU baseline on system A streams weights. A 16 GiB card cannot hold the ~20 GiB
  transformer, so its step includes weight transfer, which favours the split. System B's 24 GiB card
  held the model, so its 2.59x is the more representative scaling figure.
- The gain depends on clip length. Attention grows quadratically with tokens while the exchange grows
  linearly: 1.33x at 192 frames and 1.70x at 243 frames on this system.
- Only PCIe systems without NVLink were measured. Results on hardware with a fast GPU interconnect may
  differ.
- The Raylight rows are one run each, with `xfuser` 0.4.5.

## Sparse attention — system A

39975 tokens. The sparse node's `start_percent` holds sparsity off at the beginning of the schedule,
so the dense rows are an in-run baseline on the same work.

| configuration | s/step | vs dense |
|---|---|---|
| dense | 21.2 | 1.00x |
| **sol-attn** | **14.9** | **1.42x** |
| vsa (needs `to_gate_compress` weights) | 16.9 | 1.25x |

On system B, at 75794 tokens, sol-attn gives **1.66x** (25.1 -> 15.1 s). The gain grows with sequence
length, because attention is a larger share of a longer step.

sol-attn is recommended. vsa is correct and faster than dense, but slower than sol-attn
here and it requires a checkpoint carrying `to_gate_compress` tensors, which most MiniMax H3
checkpoints do not have.

## Pipelining the exchange — system A

39975 tokens. `exchange_chunks` overlaps each card's upload with its download. PCIe is full duplex;
the unchunked exchange runs the two directions one after the other.

| `exchange_chunks` | exchange per rank | dense s/step | sol-attn s/step |
|---|---|---|---|
| `0` (default) | 5.3 / 5.1 s | 21.2 | 14.9 |
| `4` | 4.0 / 3.8 s | 19.9 | 13.5 |
| **`8`** | **3.8 / 3.6 s** | **19.7** | **13.3** |

The transfer itself is 1.41x faster, and the step saves about the same time. 4 and 8 chunks differ
by 0.2 s; higher counts were not measured. Combined with sparse attention on system A: 21.2 -> 13.3
s/step, 1.59x.

`exchange_chunks` applies to `exchange = host` only. In `p2p` the device tensors are handed over
untouched, so there are no two directions to overlap.

## VAE decode split

Splitting the temporal chunks of the VAE decode across GPUs, system B, 243 frames:

| | prompt time |
|---|---|
| `H3MSVAESplitDecode` off | 299.9 s |
| `H3MSVAESplitDecode` on | **274.7 s** |

A saving of 25.2 s, about 8% of the job. The sampler was unchanged between the two runs (200.8 s
against 201.1 s), so the difference is decode alone.

## Exchange mode: `host` vs `p2p`

Measured on system B, where peer-to-peer access is fully available between all three GPUs:

| exchange | per rank | s/step |
|---|---|---|
| `host` with `exchange_chunks 8` | 3.8 / 3.0 / 3.0 s | **25.0-25.5** |
| `p2p` | 4.7 / 3.9 / 3.9 s | 26.1 |

Use the default `host` unless the GPUs are connected with NVLink. Staging through pinned host memory
uses each card's own link and can be pipelined; peer-to-peer copies over PCIe were slower on both
systems and cannot be pipelined.
The same ordering was measured on system A (189 s against 168 s for a 192-frame job).

## What does not change the output

System B: the same prompt at the same seed, rendered under four configurations. The encoded video and
audio streams are byte-identical:

| configuration | video stream | audio stream |
|---|---|---|
| 1 GPU, pack disabled | `b2a6bc705e596b4d` | `cbd424afeb507448` |
| 3 GPUs, `exchange host`, `exchange_chunks 8` | `b2a6bc705e596b4d` | `cbd424afeb507448` |
| 3 GPUs + `H3MSVAESplitDecode` | `b2a6bc705e596b4d` | `cbd424afeb507448` |
| 3 GPUs, `exchange p2p` | `b2a6bc705e596b4d` | `cbd424afeb507448` |

On system B, splitting the transformer across GPUs, splitting the VAE decode across GPUs and
switching the exchange transport did not change the output. A frame-level comparison of the 1-GPU and
3-GPU renders agrees — 243/243 frames identical, SSIM 1.000000, max pixel delta 0/255 — as does a
separate 124-frame check of `exchange_chunks 0` against `8`.

System A, 75794 tokens: the 2-GPU + `H3MSVAESplitDecode` dense render and the 1-GPU render with the
pack disabled have a byte-identical video stream (`d2a79f937bcede7d`).

A render approved on one GPU therefore comes out the same on the split.

The clips are in [`demo/`](demo/), so the checksums can be reproduced:
`ffmpeg -i FILE -map 0:v -f md5 -`.

## What does change the output

`sparse_attention` is the single setting above that alters the result.

It is an approximation — each query attends to a selected subset of key blocks — so it produces a
different sample. Frame by frame against a dense render at the same seed, 124 frames:

| frame | 1 | 8 | 24 | 48 | 96 | 124 |
|---|---|---|---|---|---|---|
| PSNR dB | 32.2 | 25.3 | 18.5 | 14.8 | 16.8 | 17.0 |
| SSIM | 0.972 | 0.912 | 0.716 | 0.539 | 0.613 | 0.652 |

The first frame, anchored by the reference image, nearly matches. Agreement drops as motion
accumulates and levels off. Viewed side by side, the two renders have the same character, framing,
palette and background; the main visible difference is the timing of motion.

In practice, composition and styling are usually kept and the result is reproducible at a fixed seed,
but a dense render cannot be reproduced by enabling sparse attention afterwards. Choose the mode
before searching for a seed.

Higher `tau` is sparser and diverges further; `start_percent` controls how much of the schedule stays
dense.

## Recommended settings

    sparse_attention  true       exchange         host
    exchange_chunks   8          weight_cache     true
    sparse_vsa        false      vram_block_cache false
    dynamic_vram      keep

`vram_block_cache` and `dynamic_vram = off for this model` are provided for experimentation and are
off by default; neither improved performance on the systems measured here.
