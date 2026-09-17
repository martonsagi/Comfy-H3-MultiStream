<!--
SPDX-FileCopyrightText: 2026 Márton Sági
SPDX-License-Identifier: GPL-3.0-only
-->

# Sample renders

![First frame of the sample render](h3_multistream_demo1.png)

Five renders of the same prompt at the same seed, differing only in this pack's settings.
MiniMax H3 `fl2va` int8, 8-step turbo variant, 243 frames at 1344x768 (75794 tokens), 8 steps,
seed `1019897027042686`. Each clip is 10.1 s at 24 fps with H3's generated audio.

Hardware: 3x NVIDIA RTX PRO 4000 Blackwell 24 GiB, PCIe, no NVLink.

| file | settings | s/step | whole job |
|---|---|---|---|
| [`1-gpu-baseline.mp4`](1-gpu-baseline.mp4) | pack disabled, one GPU | 65.0 | 597.3 s |
| [`3-gpu.mp4`](3-gpu.mp4) | 3-GPU split, `exchange host`, `exchange_chunks 8` | 25.1 | 299.9 s |
| [`3-gpu-vae-split.mp4`](3-gpu-vae-split.mp4) | as above + `H3MSVAESplitDecode` | 25.1 | **274.7 s** |
| [`3-gpu-p2p.mp4`](3-gpu-p2p.mp4) | as `3-gpu.mp4` but `exchange p2p` | 26.1 | — |
| [`3-gpu-sparse-attention.mp4`](3-gpu-sparse-attention.mp4) | 3-GPU + VAE split + `sparse_attention` | **18.0** | **184.6 s** |

End to end, 597.3 s for the first row and 184.6 s for the last (3.24x).

## Four of these five files are the same render

The encoded video and audio streams are byte-identical across every configuration except the last:

    1-gpu-baseline          video b2a6bc705e596b4d   audio cbd424afeb507448
    3-gpu                   video b2a6bc705e596b4d   audio cbd424afeb507448
    3-gpu-vae-split         video b2a6bc705e596b4d   audio cbd424afeb507448
    3-gpu-p2p               video b2a6bc705e596b4d   audio cbd424afeb507448
    3-gpu-sparse-attention  video 768416cb9dc5caa5   audio d978708a4d40ff70

(`ffmpeg -i FILE -map 0:v -f md5 -`, and `-map 0:a` for audio. The `.mp4` container files differ by
about 20 bytes of muxer metadata, so compare the streams rather than the files.)

Splitting the transformer across three GPUs, splitting the VAE decode across three GPUs, and
switching the exchange transport from host staging to peer-to-peer left both video and audio
identical to the single-GPU render.

`sparse_attention` is the one setting here that changes the result, because it is an approximation.
It produces a different sample with the same composition, framing and styling and different motion
timing. See
[`../performance.md`](../performance.md) for the frame-level comparison.
