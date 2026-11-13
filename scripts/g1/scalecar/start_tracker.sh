#!/usr/bin/env bash
# A to shape, A again for Hold, Start for Go, Select to stop.
# While suspended, use only A -> A and Select; do not press Start.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset PYTHONPATH
runtime_python="${SCALECAR_PYTHON:-$repo_root/.venv/bin/python}"
run_output="${SCALECAR_RUN_DIR:-$repo_root/outputs/real_runs/$(date +%Y%m%d_%H%M%S_%N)}"
mkdir -p "$run_output"
exec > >(tee -a "$run_output/tracker.log") 2>&1
printf 'Tracker logs and recording: %s\n' "$run_output"
exec "$runtime_python" -u sim2real/rl_policy/tracking.py \
  --policy-config "${SCALECAR_POLICY_CONFIG:-checkpoints/scalecar_d1146/tracker/ft500ema/policy.yaml}" \
  --motion-path "${SCALECAR_MOTION_PATH:-checkpoints/scalecar_d1146/motions/motions/take1512.npz}" \
  --motion-backend planner --robot-io zmq --controller joystick \
  --robot-interface "${G1_INTERFACE:-enp5s0}" \
  --depth-ssh-host "${G1_SSH_HOST:-unitree@192.168.123.164}" \
  --depth-remote-executable "${G1_DEPTH_EXECUTABLE:-/home/unitree/sim2real_depth_stream/build/scalecar-depth}" \
  --record --record-output "$run_output/tracking.npz" "$@"
