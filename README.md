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

# Kraken+ Medieval HTR

Trains the **kraken+** model (VGSL rebuild of HTR+) on
[dh-unibe/image-text_medieval-scripts_xiv-xv-xvi](https://huggingface.co/datasets/dh-unibe/image-text_medieval-scripts_xiv-xv-xvi)
(~497 K medieval line images, 14th–16th c. Flemish).

## Architecture (kraken+)

```
[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2
 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]
```

~2 M parameters · 2× CNN → 3× BiLSTM(256) → CTC

---

## Option A — HuggingFace Space (A100 80 GB)

Set Space secrets: `HF_TOKEN`, `OUTPUT_REPO`, optionally `GT_REPO`.

Pipeline (auto-chained via UI checkbox):
```
prepare → save-gt → compile → train → push
```

**Cost-optimised workflow:**
```bash
# 1. Preprocess for free on CPU
hf spaces settings thodel/kraken-plus-medieval-htr --hardware cpu-basic
# → click Prepare GT in the UI (~10 h, free)

# 2. Switch to A40/A100 only for training
hf spaces settings thodel/kraken-plus-medieval-htr --hardware a100-large
# → click Load GT ← Hub, then Start Training (~12–21 h)
```

---

## Option B — Standalone server (2× A40 recommended)

### Requirements
- Ubuntu 20.04+
- CUDA 12.x
- Python 3.11 or 3.12
- 100+ GB disk

### Setup
```bash
git clone https://github.com/thodel/kraken-htr-space.git
cd kraken-htr-space
bash setup.sh
```

### Run full pipeline
```bash
# Set credentials
export HF_TOKEN=hf_...
export OUTPUT_REPO=thodel/kraken-plus-medieval
export GT_REPO=thodel/kraken-htr-data

# Full pipeline (prepare → save-gt → compile → train → push)
bash run_pipeline.sh

# Skip prepare if GT already saved on Hub
bash run_pipeline.sh --skip-prepare
```

### Custom parameters
```bash
NUM_GPUS=2 EPOCHS=50 BATCH=32 LR=0.0002 bash run_pipeline.sh
```

### Manual stage-by-stage
```bash
source .venv/bin/activate
export KRAKEN_DATA_DIR=~/kraken_data

python train.py prepare
python train.py save-gt
python train.py compile
python train.py train --num-gpus 2 --epochs 50 --batch-size 32 --lr 2e-4
python train.py push --output-repo thodel/kraken-plus-medieval
```

---

## Pipeline stages

| Stage | Tool | Time (2× A40) | Notes |
|---|---|---|---|
| `prepare` | streaming + ThreadPool | ~10 h | CPU/network only |
| `save-gt` | HF Hub upload | ~1–2 h | ~20 GB JPEG + txt |
| `compile` | pyarrow | ~20 min | → train.arrow + val.arrow |
| `train` | ketos + Lightning DDP | ~12 h | 2 GPUs, batch 32/GPU |
| `push` | HF Hub upload | ~5 min | .mlmodel checkpoints |

### Resource estimate (2× A40 48 GB)

| | |
|---|---|
| Training samples | ~447 K |
| Validation samples | ~50 K |
| Batch per GPU | 32 |
| Effective batch | 64 (DDP) |
| Throughput | ~600 samples/s |
| Time per epoch | ~12 min |
| Total (50 epochs) | **~10 h** |
| GPU RAM per card | ~18 GB |

---

## Repository layout

```
├── train.py          # all pipeline stages (CLI)
├── app.py            # Gradio UI for HF Spaces
├── requirements.txt  # Python dependencies
├── setup.sh          # server environment setup
├── run_pipeline.sh   # one-command server pipeline
└── README.md
```
