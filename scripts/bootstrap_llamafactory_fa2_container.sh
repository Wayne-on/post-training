#!/usr/bin/env bash
set -euo pipefail

export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

if ! python -c "from torch.utils.tensorboard import SummaryWriter" >/dev/null 2>&1; then
  echo "[bootstrap-fa2] TensorBoard is missing; installing tensorboard==2.19.0"
  python -m pip install tensorboard==2.19.0
fi

python -c "from torch.utils.tensorboard import SummaryWriter; print('[bootstrap-fa2] TensorBoard import: OK')"
python scripts/patch_qwen35_fa2_transformers.py
python scripts/verify_llamafactory_fa2.py

exec sleep infinity
