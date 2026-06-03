#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-time environment setup for a CUDA Linux server
#
# Requirements: Python 3.11/3.12, CUDA 12.x, conda or system pip
# Usage:        bash setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON=${PYTHON:-python3.12}
VENV_DIR="${VENV_DIR:-.venv}"

echo "=== kraken+ HTR — server setup ==="
echo "Python  : $PYTHON"
echo "Venv    : $VENV_DIR"
echo "CUDA    : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found — verify CUDA manually')"
echo ""

# 1. Create virtualenv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtualenv..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[1/3] Virtualenv already exists — skipping."
fi

source "$VENV_DIR/bin/activate"

# 2. Install PyTorch with CUDA support first
#    (requirements.txt installs CPU torch; override here for CUDA 12.x)
echo "[2/3] Installing PyTorch (CUDA 12.4)..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# 3. Install project requirements
echo "[3/3] Installing project requirements..."
pip install -r requirements.txt -q

# Verify GPU access
python - <<'EOF'
import torch
print(f"\n=== GPU check ===")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPUs     : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i} : {p.name}  {p.total_memory // 1024**3} GB")
if torch.cuda.device_count() == 0:
    print("  WARNING: no GPUs detected!")
EOF

echo ""
echo "Setup complete. Activate with:  source $VENV_DIR/bin/activate"
