#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Missing $PYTHON_BIN. Install Python 3.12 first." >&2
  exit 2
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

MAX_JOBS="${MAX_JOBS:-8}" \
  python -m pip install --no-build-isolation -r "$PROJECT_ROOT/requirements.txt"

python - <<'PY'
import importlib.metadata as metadata
import torch

for package in ("torch", "openrlhf", "vllm", "ray", "deepspeed", "transformers", "flash-attn"):
    print(f"{package}={metadata.version(package)}")

print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    torch.manual_seed(42)
    x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    print("CUDA_WITNESS", tuple(y.shape), torch.cuda.get_device_name(), float(y.float().abs().sum()))
PY

echo "Environment ready: $VENV_DIR"
