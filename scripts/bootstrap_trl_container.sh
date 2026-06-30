#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
import trl
from torch.utils.tensorboard import SummaryWriter
from trl import GRPOConfig, GRPOTrainer

print("[bootstrap-trl] torch:", torch.__version__)
print("[bootstrap-trl] trl:", trl.__version__)
print("[bootstrap-trl] GRPOTrainer import: OK")
print("[bootstrap-trl] TensorBoard import: OK")
PY

exec sleep infinity
