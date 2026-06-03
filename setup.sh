#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-time environment setup for the 2× A40 server
#
# Installs a self-contained virtualenv at VENV_DIR (default: ~/kraken-env)
# so nothing touches the system Python.
#
# Usage:
#   bash setup.sh
#   VENV_DIR=/opt/kraken-env bash setup.sh   # custom location
# ---------------------------------------------------------------------------
set -euo pipefail

# Prefer Python 3.12, fall back to 3.11 or 3.10
for v in python3.12 python3.11 python3.10 python3; do
    if command -v $v &>/dev/null; then
        PYTHON=$v; break
    fi
done
PYTHON="${PYTHON:-python3}"

VENV_DIR="${VENV_DIR:-$HOME/kraken-env}"

echo "=== kraken+ HTR — server setup ==="
echo "Python    : $PYTHON  ($(${PYTHON} --version))"
echo "Venv      : $VENV_DIR"
echo "CUDA      : $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not in PATH — verify CUDA manually')"
echo ""

# 1. Create virtualenv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtualenv at $VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[1/4] Virtualenv already exists — skipping creation."
fi

source "$VENV_DIR/bin/activate"
echo "      Active Python: $(python --version)  |  $(which python)"

# 2. Upgrade pip inside the venv
echo "[2/4] Upgrading pip …"
pip install --upgrade pip wheel setuptools -q

# 3. Install PyTorch with CUDA 12.4 (override the CPU wheel from requirements.txt)
echo "[3/4] Installing PyTorch (CUDA 12.4) …"
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 \
    -q

# 4. Install project dependencies
echo "[4/4] Installing project requirements …"
pip install -r requirements.txt -q

# Create server data/checkpoint directories if they don't exist
DATA_DIR="${KRAKEN_DATA_DIR:-/mnt/wbkolleg_dh_1/Data_Kraken-training}"
CKPT_DIR="${KRAKEN_MODELS_DIR:-/home/tobias/training-checkpoints}"
mkdir -p "$DATA_DIR/ground_truth" "$CKPT_DIR" 2>/dev/null && \
    echo "" && echo "  Data dir  : $DATA_DIR" && \
    echo "  Ckpt dir  : $CKPT_DIR" || \
    echo "  Note: could not create $DATA_DIR or $CKPT_DIR — check permissions"

# Verify GPU access
python - <<'EOF'
import torch
print("\n=== GPU check ===")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
n = torch.cuda.device_count()
print(f"GPUs     : {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}  : {p.name}  —  {p.total_memory // 1024**3} GB VRAM")
if n == 0:
    print("  WARNING: no GPUs detected — check CUDA installation")
EOF

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
