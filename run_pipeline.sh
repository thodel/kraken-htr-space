#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_pipeline.sh — full training pipeline for a 2× A40 server
#
# Stages:
#   prepare  — stream HF dataset → JPEG + .gt.txt  (~10 h, CPU/network only)
#   save-gt  — push ground truth to HF Hub          (persist before compile)
#   compile  — JPEG + .gt.txt → Arrow binary        (~20 min, CPU)
#   train    — ketos train on 2× A40 GPUs           (~12 h)
#   push     — upload checkpoints to HF Hub
#
# Usage:
#   Full pipeline:               bash run_pipeline.sh
#   Skip prepare (GT on Hub):    bash run_pipeline.sh --skip-prepare
#   Custom params:               EPOCHS=100 BATCH=64 bash run_pipeline.sh
#
# Environment variables (override defaults):
#   KRAKEN_DATA_DIR   local data directory       [default: ~/kraken_data]
#   NUM_GPUS          number of GPUs to use      [default: 2]
#   EPOCHS            training epochs            [default: 50]
#   BATCH             batch size per GPU         [default: 32]
#   LR                learning rate              [default: 0.0002]
#   OUTPUT_REPO       HF model repo for ckpts   [default: thodel/kraken-plus-medieval]
#   GT_REPO           HF dataset repo for GT    [default: thodel/kraken-htr-data]
#   HF_TOKEN          HF write token            [required for push steps]
# ---------------------------------------------------------------------------
set -euo pipefail

# --- defaults ---
KRAKEN_DATA_DIR="${KRAKEN_DATA_DIR:-$HOME/kraken_data}"
NUM_GPUS="${NUM_GPUS:-2}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
LR="${LR:-0.0002}"
OUTPUT_REPO="${OUTPUT_REPO:-thodel/kraken-plus-medieval}"
GT_REPO="${GT_REPO:-thodel/kraken-htr-data}"
SKIP_PREPARE=0
SKIP_SAVE_GT=0

# parse flags
for arg in "$@"; do
    case $arg in
        --skip-prepare)   SKIP_PREPARE=1 ;;
        --skip-save-gt)   SKIP_SAVE_GT=1 ;;
    esac
done

export KRAKEN_DATA_DIR GT_REPO

VENV="${VENV_DIR:-.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $VENV — run bash setup.sh first"; exit 1
fi
source "$VENV/bin/activate"

PYTHON="$VENV/bin/python"
LOG_DIR="$KRAKEN_DATA_DIR/logs"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
run_stage() {
    local stage="$1"; shift
    echo ""
    echo "============================================================"
    echo "  [$(timestamp)] STAGE: $stage"
    echo "============================================================"
    "$PYTHON" train.py "$stage" "$@" 2>&1 | tee -a "$LOG_DIR/${stage}.log"
}

echo "=== kraken+ HTR pipeline ==="
echo "  Data dir   : $KRAKEN_DATA_DIR"
echo "  GPUs       : $NUM_GPUS"
echo "  Epochs     : $EPOCHS  |  Batch/GPU: $BATCH  |  LR: $LR"
echo "  Output     : $OUTPUT_REPO"
echo "  GT repo    : $GT_REPO"
echo ""

# Stage 1 — Prepare (optional, skip if GT already on Hub)
if [ "$SKIP_PREPARE" -eq 0 ]; then
    run_stage prepare
else
    echo "[skip] prepare — loading GT from Hub instead"
    run_stage load-gt
fi

# Stage 2 — Save GT to Hub (skip if we just loaded it)
if [ "$SKIP_SAVE_GT" -eq 0 ] && [ "$SKIP_PREPARE" -eq 0 ]; then
    run_stage save-gt
fi

# Stage 3 — Compile to Arrow binary
run_stage compile

# Stage 4 — Train (2× A40)
run_stage train \
    --epochs    "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr         "$LR" \
    --num-gpus   "$NUM_GPUS"

# Stage 5 — Push checkpoints to Hub
if [ -n "${HF_TOKEN:-}" ] || [ -n "${OUTPUT_REPO:-}" ]; then
    run_stage push --output-repo "$OUTPUT_REPO"
else
    echo "[skip] push — set HF_TOKEN and OUTPUT_REPO to enable"
fi

echo ""
echo "=== Pipeline complete at $(timestamp) ==="
echo "    Checkpoints: $KRAKEN_DATA_DIR/models/"
