#!/usr/bin/env bash
set -euo pipefail

if ! python -c "from torch.utils.tensorboard import SummaryWriter" >/dev/null 2>&1; then
  echo "[bootstrap] TensorBoard is missing; installing tensorboard==2.19.0"
  python -m pip install tensorboard==2.19.0
fi

python -c "from torch.utils.tensorboard import SummaryWriter; print('[bootstrap] TensorBoard import: OK')"

exec sleep infinity
