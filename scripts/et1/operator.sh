#!/usr/bin/env bash
# Keyboard operator on the ET1 board: ssh -t <board> <root>/repo/scripts/et1/operator.sh
set -euo pipefail
root="${PROTEUS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export PYTHONPATH="$root/site-packages:$root/repo" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME="$root/hf"
exec python3 -u "$root/repo/scripts/et1/operator_keyboard.py"
