#!/usr/bin/env bash
# Score a configuration and then run the verification pass.
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

python3 -m mechphase.harness.score --experiment "${EXPERIMENT}" "$@"
exec python3 -m mechphase.harness.audit
