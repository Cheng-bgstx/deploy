#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
mode="${1:-}"
[[ $# -eq 0 ]] || shift
case "$mode" in
  publisher)
    exec uv run --project venv/pico --no-sync python -m sim2real.teleop.pico_retarget_pub --robot oli "$@"
    ;;
  sim)
    exec uv run --no-sync python -m sim2real.sim_env.base_sim --robot oli "$@"
    ;;
  policy-sim|robot)
    publisher_ip="${1:-127.0.0.1}"
    [[ $# -eq 0 ]] || shift
    io_args=(--robot-io zmq)
    policy="checkpoints/proteus/wsq61/limx_oli_edu/policy-sim.yaml"
    if [[ "$mode" == robot ]]; then
      export ROBOT_TYPE=HU_D04_01
      io_args=(--robot-io inline --robot-ip "${1:-127.0.0.1}")
      [[ $# -eq 0 ]] || shift
      policy="checkpoints/proteus/wsq61/limx_oli_edu/policy.yaml"
    fi
    exec uv run --no-sync python -m sim2real.rl_policy.tracking --robot oli "${io_args[@]}" \
      --policy-config "$policy" --rl-rate 50 --inference-backend onnx-cpu --controller pico \
      --motion-backend zmq --motion-path outputs/oli/pico_hold \
      --motion-zmq-connect "tcp://${publisher_ip}:28701" --pico-zmq-connect "tcp://${publisher_ip}:5592" "$@"
    ;;
  *)
    echo "Usage: $0 publisher [publisher flags] | sim [sim flags] | policy-sim [publisher-ip] [flags] | robot [publisher-ip] [robot-ip] [flags]" >&2
    exit 2
    ;;
esac
