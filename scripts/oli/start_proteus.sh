#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <OLI any4hdmi motion path> [robot-ip=127.0.0.1] [policy.yaml]" >&2
  exit 2
fi
export ROBOT_TYPE=HU_D04_01
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
policy_path="${3:-checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml}"
exec uv run --no-sync python -m sim2real.rl_policy.tracking \
  --robot oli --robot-io inline --robot-ip "${2:-127.0.0.1}" \
  --policy-config "$policy_path" --rl-rate 50 --inference-backend onnx-cpu \
  --controller keyboard --motion-backend npz --motion-path "$1"
