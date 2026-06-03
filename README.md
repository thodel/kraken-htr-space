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

## Option A — Standalone server (2× A40)

### Requirements

- Ubuntu 20.04+, CUDA 12.x
- Python 3.11 or 3.12
- Access to `/mnt/wbkolleg_dh_1/Data_Kraken-training` (data)
- Write access to `/home/tobias/training-checkpoints` (checkpoints)

### 1. Clone & setup

```bash
git clone https://github.com/thodel/kraken-htr-space.git
cd kraken-htr-space

# Creates virtualenv at ~/kraken-env, installs PyTorch (CUDA 12.4) + all deps
bash setup.sh
```

After setup, the environment is at `~/kraken-env`. Activate with:
```bash
source ~/kraken-env/bin/activate
```

### 2. Run the full pipeline

```bash
export HF_TOKEN=hf_...                          # HF write token (for save/push)
export OUTPUT_REPO=thodel/kraken-plus-medieval   # model checkpoint repo on Hub
export GT_REPO=thodel/kraken-htr-data            # ground-truth storage repo on Hub

bash run_pipeline.sh
```

This runs all stages sequentially and logs everything to
`/mnt/wbkolleg_dh_1/Data_Kraken-training/logs/pipeline_YYYYMMDD_HHMMSS.log`.

### 3. Skip prepare if GT is already saved on Hub

```bash
bash run_pipeline.sh --skip-prepare
```

Downloads GT from `GT_REPO` to the data directory, then compiles and trains.

### 4. Manual stage-by-stage

```bash
source ~/kraken-env/bin/activate
export KRAKEN_DATA_DIR=/mnt/wbkolleg_dh_1/Data_Kraken-training
export KRAKEN_MODELS_DIR=/home/tobias/training-checkpoints

python train.py prepare                         # ~10 h (CPU + network)
python train.py save-gt                         # push GT to Hub
python train.py compile                         # ~20 min (CPU)
python train.py train \
    --num-gpus   2 \
    --epochs     50 \
    --batch-size 32 \
    --lr         2e-4                           # ~10 h (2× A40)
python train.py push \
    --output-repo thodel/kraken-plus-medieval   # upload checkpoints
```

### 5. Custom parameters

```bash
NUM_GPUS=2 EPOCHS=100 BATCH=64 LR=0.0001 bash run_pipeline.sh
```

### Server paths

| Purpose | Path |
|---|---|
| Training data & GT | `/mnt/wbkolleg_dh_1/Data_Kraken-training/` |
| Checkpoints | `/home/tobias/training-checkpoints/` |
| Virtualenv | `~/kraken-env/` |
| Pipeline log | `/mnt/wbkolleg_dh_1/Data_Kraken-training/logs/` |

All paths can be overridden via environment variables:

| Variable | Default |
|---|---|
| `KRAKEN_DATA_DIR` | `/mnt/wbkolleg_dh_1/Data_Kraken-training` |
| `KRAKEN_MODELS_DIR` | `/home/tobias/training-checkpoints` |
| `VENV_DIR` | `~/kraken-env` |
| `GT_REPO` | `thodel/kraken-htr-data` |
| `OUTPUT_REPO` | `thodel/kraken-plus-medieval` |

### Resource estimate (2× A40 48 GB)

| | |
|---|---|
| Training samples | ~447 K |
| Validation samples | ~50 K |
| Batch per GPU | 32 (effective batch 64 via DDP) |
| Estimated throughput | ~600 samples/s |
| Time per epoch | ~12 min |
| **Total training (50 epochs)** | **~10 h** |
| GPU RAM per card | ~18 GB |

---

## Option B — HuggingFace Space (A100 80 GB)

The Space at [thodel/kraken-plus-medieval-htr](https://huggingface.co/spaces/thodel/kraken-plus-medieval-htr)
provides a Gradio UI with the same pipeline stages.

### Setup

Set the following **Space secrets** (Settings → Repository secrets):

| Secret | Value |
|---|---|
| `HF_TOKEN` | HF write-access token |
| `OUTPUT_REPO` | e.g. `thodel/kraken-plus-medieval` |
| `GT_REPO` | e.g. `thodel/kraken-htr-data` (default) |

### Cost-optimised workflow

```bash
# 1. Preprocess for free on cpu-basic
hf spaces settings thodel/kraken-plus-medieval-htr --hardware cpu-basic
# → in the UI: click "▶ Prepare GT" with auto-pipeline checked (~10 h, free)
# → save-gt runs automatically after prepare

# 2. Switch to A100 only for training
hf spaces settings thodel/kraken-plus-medieval-htr --hardware a100-large
# → in the UI: click "📥 Load GT ← Hub", then "🚀 Start Training" (~21 h, ~$52)
```

| Phase | Hardware | Cost |
|---|---|---|
| Prepare + compile | cpu-basic (free) | $0 |
| Train 50 epochs | a100-large ($2.50/hr) | ~$52 |
| **Total** | | **~$52** |

---

## Pipeline stages

| Stage | What it does | Time |
|---|---|---|
| `prepare` | Stream HF dataset → JPEG + `.gt.txt` pairs | ~10 h |
| `save-gt` | Push ground truth to `GT_REPO` on HF Hub | ~1–2 h |
| `load-gt` | Restore ground truth from Hub (after restart) | ~30 min |
| `compile` | Convert JPEG + `.gt.txt` → Arrow binary for ketos | ~20 min |
| `train` | `ketos -d cuda:0,cuda:1 train -f binary …` | ~10 h (2× A40) / ~21 h (A100) |
| `push` | Upload `.mlmodel` checkpoints to Hub | ~5 min |

---

## Repository layout

```
├── train.py          # all pipeline stages (CLI: prepare / save-gt / load-gt /
│                     #   compile / train / push)
├── app.py            # Gradio UI for HF Spaces (Option B)
├── requirements.txt  # Python dependencies
├── setup.sh          # server environment setup (Option A)
├── run_pipeline.sh   # one-command pipeline runner (Option A)
└── README.md
```
