---
title: Kraken+ Medieval HTR Training
emoji: 📜
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "6.15.2"
app_file: app.py
pinned: false
hardware: a100-large
license: mit
---

# Kraken+ Medieval HTR — Training Space

Trains the **kraken+** model (a VGSL rebuild of HTR+ with 3× BiLSTM) on the
[dh-unibe/image-text_medieval-scripts_xiv-xv-xvi](https://huggingface.co/datasets/dh-unibe/image-text_medieval-scripts_xiv-xv-xvi)
dataset (~497 K medieval line images, 14th–16th c. Flemish).

## Architecture

```
[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2
 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]
```

~2 M parameters · CNN feature extractor → 3× BiLSTM(256) → CTC output

## Setup

Set the following **Space secrets** before starting training:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your HF write-access token |
| `OUTPUT_REPO` | HF repo to push checkpoints, e.g. `yourname/kraken-plus-medieval` |

## Hardware

Configured for **Nvidia A100 80 GB** (`a100-large`).  
Estimated runtime: ~22 h (1.5 h prepare + ~21 h training).  
Estimated cost: ~$55 at $2.50/hr.
