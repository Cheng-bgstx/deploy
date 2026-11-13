#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/pm01/env.sh
set -u
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
motion="checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz"
if [[ $# -gt 0 && "$1" != -* ]]; then motion="$1"; shift; fi
if [[ " $* " != *" --help "* ]]; then
  uv run --no-sync python -m scripts.pm01.check_robot
fi
exec uv run --no-sync python -m sim2real.rl_policy.tracking \
  --robot pm01 --robot-io inline \
  --policy-config "${PROTEUS_POLICY_CONFIG:-checkpoints/proteus/wsq61/engineai_pm01/policy.yaml}" \
  --rl-rate 50 --inference-backend onnx-cpu --controller keyboard --keyboard-layout et1 \
  --motion-backend npz --motion-path "$motion" "$@"
