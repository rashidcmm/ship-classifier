#!/usr/bin/env bash
# One-time setup on a Linux or macOS training machine. Run from the project folder:
#   bash setup/setup_unix.sh
# Needs Python 3.10+.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ needed, found ' + sys.version"
[ -d .venv ] || "$PY" -m venv .venv
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU found: installing PyTorch with CUDA support (~3 GB download)..."
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
  echo "No NVIDIA GPU: installing the default PyTorch build (uses Apple GPU on M-series Macs, else CPU)."
  "$VPY" -m pip install torch torchvision
fi
"$VPY" -m pip install -r requirements.txt
"$VPY" src/check_setup.py
echo
echo "Next: source .venv/bin/activate   then   python src/benchmark.py"
