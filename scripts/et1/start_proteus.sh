#!/usr/bin/env bash
# Galbot ET1 Proteus deployment: Bridge (200 Hz) and Tracking (50 Hz) on this machine.
# DDS startup is passive; a ready operator A on the ET1 remote requests takeover.
# A shape -> A reference pose and Hold -> lower the hoist -> Start. Select stops.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
python_bin="${SIM2REAL_PYTHON:-$repo/.venv/bin/python}"
interface="${ET1_INTERFACE:?Set ET1_INTERFACE to the NIC on the ET1 network (192.168.123.x)}"
policy="${PROTEUS_POLICY_CONFIG:-checkpoints/proteus/wsq61/galbot_et1/policy-pc.yaml}"
motion="${PROTEUS_MOTION_PATH:-checkpoints/galbot_et1/motions/motions/taichi.npz}"
playback_limit="${PROTEUS_PLAYBACK_LIMIT_S:-8}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
# Batch-one CPU inference must leave time for DDS callbacks and the watchdog.
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# One I/O jitter window for both processes; sustained loss still latches damping.
export SIM2REAL_IO_TIMEOUT_S="${PROTEUS_IO_TIMEOUT_S:-0.25}"
export PYTHONPATH="$repo"
[[ -x "$python_bin" ]] || { echo 'Set SIM2REAL_PYTHON to the Python 3.10 inference environment.' >&2; exit 1; }
mkdir -p outputs/proteus-et1
exec 9>outputs/proteus-et1/active.lock
flock -n 9 || { echo 'A Proteus ET1 supervisor is already running.' >&2; exit 1; }
if pgrep -af '^([^ ]*/)?python[^ ]* .*(sim2real/rl_policy/tracking.py|scripts/g1/real_bridge.py)'; then
  echo 'An existing robot controller must be stopped before starting another writer.' >&2
  exit 1
fi
# et1-dep's Loco and its gamepad controllers also write rt/lowcmd (stop them with dist/cpp/loco/<arch>/tmux.sh stop;
# when they run on the robot and this script on a PC, this check cannot see them).
if pgrep -af '(^|/)(loco|gamepad-[a-z]+)( |$)'; then
  echo 'et1-dep Loco or a gamepad controller is running: stop it first.' >&2
  exit 1
fi
[[ -f "$policy" ]] || "$python_bin" scripts/g1/prepare_proteus_deployment.py --robot et1 \
  --policy-config checkpoints/proteus/wsq61/galbot_et1/policy.yaml
"$python_bin" -c 'import sys, yaml; import unitree_sdk2py.core.channel; from sim2real.config.robots import ET1_CFG; from sim2real.rl_policy.utils.motion import MotionDataset; c = yaml.safe_load(open(sys.argv[1])); assert c["proteus"]["reference_forcing"] and c.get("soft_joint_pos_limits"); assert ET1_CFG.mjcf_path.is_file(); MotionDataset.create_from_path(sys.argv[2], robot_cfg=ET1_CFG); print("SDK2, ET1 MJCF, policy overlay and motion ready")' "$policy" "$motion"
run="${PROTEUS_RUN_DIR:-$repo/outputs/proteus-et1/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run"
printf 'Logs: %s\nPolicy: %s\nMotion: %s; playback limit: %s s\n' "$run" "$policy" "$motion" "$playback_limit"
printf 'I/O jitter tolerance: %s s. Wait for "Tracker and LowState ready" in bridge.log.\n' "$SIM2REAL_IO_TIMEOUT_S"
printf 'A shape -> A reference pose -> wait Hold ready -> lower hoist -> Start. Select loses active posture support.\n'
bridge_args=()
if [[ "${PROTEUS_READ_ONLY:-0}" == 1 ]]; then
  bridge_args+=(--read-only --max-runtime-s "${PROTEUS_READ_ONLY_S:-30}")
fi
bridge_pid='' tracker_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$tracker_pid" ]] || kill -TERM "$tracker_pid" 2>/dev/null || true
  [[ -z "$bridge_pid" ]] || kill -TERM "$bridge_pid" 2>/dev/null || true
  [[ -z "$tracker_pid" ]] || wait "$tracker_pid" 2>/dev/null || true
  [[ -z "$bridge_pid" ]] || wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$python_bin" -u scripts/g1/real_bridge.py --robot et1 --interface "$interface" \
  --timeout-s "$SIM2REAL_IO_TIMEOUT_S" --state-log "$run/bridge-state.json" "${bridge_args[@]}" > "$run/bridge.log" 2>&1 &
bridge_pid=$!
"$python_bin" -u sim2real/rl_policy/tracking.py --robot et1 --robot-io zmq \
  --inference-backend onnx-cpu --controller joystick --robot-interface "$interface" \
  --policy-config "$policy" --npz-playback-limit-s "$playback_limit" --motion-path "$motion" \
  --record --record-output "$run/tracking.npz" > "$run/tracker.log" 2>&1 &
tracker_pid=$!
printf '%s\n' "$bridge_pid" > "$run/bridge.pid"
printf '%s\n' "$tracker_pid" > "$run/tracker.pid"
wait -n "$bridge_pid" "$tracker_pid"
