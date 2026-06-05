#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_pipeline.sh — full training pipeline for the 2× A40 server
#
# Stages:
#   prepare  — stream HF dataset → JPEG + .gt.txt  (~10 h, CPU/network)
#   save-gt  — push ground truth to HF Hub          (persist before compile)
#   compile  — JPEG + .gt.txt → Arrow binary        (~20 min, CPU)
#   train    — ketos train on 2× A40 GPUs           (~10 h)
#   push     — upload checkpoints to HF Hub
#
# Usage:
#   Full pipeline:             bash run_pipeline.sh
#   Skip prepare (GT on Hub):  bash run_pipeline.sh --skip-prepare
#   Custom params:             EPOCHS=100 BATCH=64 bash run_pipeline.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Server-specific paths (edit here or override via environment)
# ---------------------------------------------------------------------------
KRAKEN_DATA_DIR="${KRAKEN_DATA_DIR:-/mnt/wbkolleg_dh_1/Data_Kraken-training}"
KRAKEN_MODELS_DIR="${KRAKEN_MODELS_DIR:-/home/tobias/training-checkpoints}"
VENV_DIR="${VENV_DIR:-$HOME/kraken-env}"

# Training parameters
NUM_GPUS="${NUM_GPUS:-2}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
LR="${LR:-0.0002}"

# HF Hub repos (set via env or export before running)
OUTPUT_REPO="${OUTPUT_REPO:-thodel/kraken-plus-medieval}"
GT_REPO="${GT_REPO:-thodel/kraken-htr-data}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_PREPARE=0
SKIP_SAVE_GT=0
for arg in "$@"; do
    case $arg in
        --skip-prepare) SKIP_PREPARE=1 ;;
        --skip-save-gt) SKIP_SAVE_GT=1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $VENV_DIR"
    echo "       Run:  bash setup.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"
export KRAKEN_DATA_DIR KRAKEN_MODELS_DIR GT_REPO

# HF token/config cache → small dir on home (auth files only, not parquet data)
# HF_DATASETS_CACHE is intentionally NOT set here: caching is disabled in
# train.py (disable_caching()) so streaming parquets never touch the disk.
HF_CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="$KRAKEN_DATA_DIR/logs"
mkdir -p "$LOG_DIR" "$KRAKEN_MODELS_DIR"

LOGFILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_stage() {
    local stage="$1"; shift
    echo ""
    echo "============================================================"
    echo "  [$(timestamp)] STAGE: $stage"
    echo "============================================================"
    python train.py "$stage" "$@"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== kraken+ HTR pipeline — $(timestamp) ==="
echo "  Data dir   : $KRAKEN_DATA_DIR"
echo "  Checkpoints: $KRAKEN_MODELS_DIR"
echo "  Venv       : $VENV_DIR"
echo "  GPUs       : $NUM_GPUS"
echo "  Epochs     : $EPOCHS  |  Batch/GPU: $BATCH  |  LR: $LR"
echo "  Output repo: $OUTPUT_REPO"
echo "  GT repo    : $GT_REPO"
echo "  Log        : $LOGFILE"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Stage 1 — Prepare or load from Hub
if [ "$SKIP_PREPARE" -eq 0 ]; then
    run_stage prepare
else
    echo ""
    echo "[skip] prepare — restoring GT from Hub ($GT_REPO)"
    run_stage load-gt
    SKIP_SAVE_GT=1   # no need to re-save what we just loaded
fi

# Stage 2 — Persist GT to Hub
if [ "$SKIP_SAVE_GT" -eq 0 ]; then
    run_stage save-gt
fi

# Stage 3 — Compile to Arrow binary
run_stage compile

# Stage 4 — Train on 2× A40
run_stage train \
    --epochs     "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr         "$LR" \
    --num-gpus   "$NUM_GPUS"

# Stage 5 — Push checkpoints to Hub
if [ -n "${HF_TOKEN:-}" ]; then
    run_stage push --output-repo "$OUTPUT_REPO"
else
    echo ""
    echo "[skip] push — HF_TOKEN not set"
fi

echo ""
echo "=== Pipeline finished at $(timestamp) ==="
echo "    Log      : $LOGFILE"
echo "    Ckpts    : $KRAKEN_MODELS_DIR"
