#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/torchrun_multinode.sh <script.py> <config.yaml>"
  exit 2
fi

: "${NNODES:?Set NNODES, for example NNODES=2}"
: "${NODE_RANK:?Set NODE_RANK to 0 on the master node and 1 on the second node}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the master node IP reachable from all nodes}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "$@"
