#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
mode="${1:-}"
[[ $# -eq 0 ]] || shift
case "$mode" in
  publisher)
    exec uv run --project venv/pico --no-sync python -m sim2real.teleop.pico_retarget_pub --robot a3 "$@"
    ;;
  sim)
    exec uv run --no-sync python -m sim2real.sim_env.base_sim --robot a3 "$@"
    ;;
  policy-sim|robot)
    publisher_ip="127.0.0.1"
    if [[ $# -gt 0 && "$1" != -* ]]; then publisher_ip="$1"; shift; fi
    io_args=(--robot-io zmq)
    policy="checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml"
    if [[ "$mode" == robot ]]; then
      io_args=(--robot-io inline)
      policy="checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml"
    fi
    hold="checkpoints/agibot_a3/pico_hold"
    if [[ ! -f "$hold/manifest.json" ]]; then
      uv run --no-sync python -m scripts.make_hold_motion --robot a3 --output "$hold"
    fi
    exec uv run --no-sync python -m sim2real.rl_policy.tracking --robot a3 "${io_args[@]}" \
      --policy-config "${PROTEUS_POLICY_CONFIG:-$policy}" --rl-rate 50 --inference-backend onnx-cpu --controller pico \
      --motion-backend zmq --motion-path "$hold" \
      --motion-zmq-connect "tcp://${publisher_ip}:28701" --pico-zmq-connect "tcp://${publisher_ip}:5592" "$@"
    ;;
  *)
    echo "Usage: $0 publisher [flags] | sim [flags] | policy-sim [publisher-ip] [flags] | robot [publisher-ip] [flags]" >&2
    exit 2
    ;;
esac
