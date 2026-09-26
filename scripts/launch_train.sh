#!/usr/bin/env bash
# Launch a training run on the configuration named by the first argument.
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if command -v torchrun >/dev/null 2>&1 && [ "${WORLD_SIZE:-1}" -gt 1 ]; then
  exec torchrun --nproc_per_node "${WORLD_SIZE}" -m mechphase.harness.fit \
    --experiment "${EXPERIMENT}" "$@"
fi

if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  exec srun python3 -m mechphase.harness.fit --experiment "${EXPERIMENT}" "$@"
fi

exec python3 -m mechphase.harness.fit --experiment "${EXPERIMENT}" "$@"
