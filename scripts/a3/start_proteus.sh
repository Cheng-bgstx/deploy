#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
motion="checkpoints/agibot_a3/motions/motions/walk1_subject1.npz"
if [[ $# -gt 0 && "$1" != -* ]]; then motion="$1"; shift; fi
exec uv run --no-sync python -m sim2real.rl_policy.tracking \
  --robot a3 --robot-io inline \
  --policy-config "${PROTEUS_POLICY_CONFIG:-checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml}" \
  --rl-rate 50 --inference-backend onnx-cpu --controller keyboard \
  --motion-backend npz --motion-path "$motion" "$@"
