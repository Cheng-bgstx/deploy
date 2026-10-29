#!/usr/bin/env bash
# Startup releases G1's current motion mode; this is a real command bridge.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset PYTHONPATH
runtime_python="${SCALECAR_PYTHON:-$repo_root/.venv/bin/python}"
exec "$runtime_python" -u scripts/g1/real_bridge.py \
  --interface "${G1_INTERFACE:-enp5s0}" "$@"
