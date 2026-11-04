#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <X2 any4hdmi motion path> [policy.yaml]" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
# Source ROS 2 Humble and AimDK before invoking this launcher. It does not stop
# the native mc service; release ownership on the robot explicitly before use.
exec uv run --no-sync python -m sim2real.rl_policy.tracking \
  --robot x2 --robot-io inline --policy-config "${2:-checkpoints/proteus/wsq61/lingxi_x2/policy.yaml}" \
  --rl-rate 50 --inference-backend onnx-cpu --controller keyboard \
  --motion-backend npz --motion-path "$1"
