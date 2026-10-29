#!/usr/bin/env bash
# PC-only deployment. DDS startup is passive; a ready operator A requests takeover.
set -euo pipefail
proteus_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$proteus_repo"
proteus_python="${SIM2REAL_PYTHON:-$proteus_repo/.venv/bin/python}"
proteus_interface="${G1_INTERFACE:-enp5s0}"
# Original walk1 seconds 3-11: eight seconds of forward walking at source speed.
proteus_default_motion="$proteus_repo/checkpoints/proteus/proteus_60/unitree_g1/motions/walk_forward_original"
proteus_motion="${PROTEUS_MOTION_PATH:-$proteus_default_motion}"
proteus_playback_limit="${PROTEUS_PLAYBACK_LIMIT_S:-8}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
# Batch-one CPU inference must leave time for DDS callbacks and the watchdog.
# ONNXModule reads this before creating its session; changing it cannot retune
# an already-running Tracker.
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# A bounded I/O jitter window; cached feedback and the latest command stay active.
export SIM2REAL_IO_TIMEOUT_S="${PROTEUS_IO_TIMEOUT_S:-0.25}"
export PYTHONPATH="$proteus_repo"
if [[ ! -x "$proteus_python" ]]; then
  echo 'Set SIM2REAL_PYTHON to the existing Python 3.10 inference environment.' >&2
  exit 1
fi
mkdir -p outputs/proteus-pc
exec 9>outputs/proteus-pc/active.lock
flock -n 9 || { echo 'A Proteus PC supervisor is already running.' >&2; exit 1; }
if pgrep -af '^([^ ]*/)?python[^ ]* .*(sim2real/rl_policy/tracking.py|scripts/g1/real_bridge.py|g1_debug_mode)' ; then
  echo 'An existing robot controller must be stopped before starting another writer.' >&2
  exit 1
fi
if pgrep -af '^([^ ]*/)?g1_29dof_controller([[:space:]]|$)'; then
  echo 'An existing G1 controller must be stopped before starting another writer.' >&2
  exit 1
fi
proteus_policy=checkpoints/proteus/proteus_60/unitree_g1/policy-pc.yaml
[[ -f "$proteus_policy" ]] || "$proteus_python" scripts/g1/prepare_proteus_deployment.py
if [[ "$proteus_motion" == "$proteus_default_motion" && ! -e "$proteus_motion" ]]; then
  "$proteus_python" scripts/g1/prepare_proteus_motion.py --output-dir "$proteus_motion"
fi
"$proteus_python" .agents/skills/configure-g1-sim2real/scripts/verify_install.py \
  --profile root-cpu --onnx checkpoints/proteus/proteus_60/unitree_g1/policy.onnx \
  --asset hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml
"$proteus_python" -c 'import sys; import cyclonedds; import unitree_sdk2py.core.channel; import yaml; from sim2real.config.robots import G1_CFG; from sim2real.rl_policy.utils.motion import MotionDataset; c=yaml.safe_load(open("checkpoints/proteus/proteus_60/unitree_g1/policy-pc.yaml")); assert c["proteus"]["reference_forcing"]; MotionDataset.create_from_path(sys.argv[1], robot_cfg=G1_CFG); print("PC SDK2, ONNX and offline motion ready")' "$proteus_motion"
proteus_run="${PROTEUS_RUN_DIR:-$proteus_repo/outputs/proteus-pc/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$proteus_run"
printf 'Logs: %s\n' "$proteus_run"
printf 'Motion: %s; reference playback limit: %s s\n' "$proteus_motion" "$proteus_playback_limit"
printf 'ONNX CPU threads: %s\n' "$SIM2REAL_ORT_NUM_THREADS"
printf 'I/O jitter tolerance: %s s; continue through brief gaps, latch on sustained loss.\n' "$SIM2REAL_IO_TIMEOUT_S"
printf 'Wait for Tracker and LowState ready. A arms-down stand -> A 2s reference transition -> wait Hold ready -> lower gantry -> Start. Select loses active posture support.\n'
bridge_args=()
if [[ "${PROTEUS_READ_ONLY:-0}" == 1 ]]; then
  bridge_args+=(--read-only --max-runtime-s 5)
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
"$proteus_python" -u scripts/g1/real_bridge.py --interface "$proteus_interface" \
  --timeout-s "$SIM2REAL_IO_TIMEOUT_S" --state-log "$proteus_run/bridge-state.json" "${bridge_args[@]}" > "$proteus_run/bridge.log" 2>&1 &
bridge_pid=$!
"$proteus_python" -u sim2real/rl_policy/tracking.py --robot g1 --robot-io zmq \
  --inference-backend onnx-cpu --controller joystick --robot-interface "$proteus_interface" \
  --policy-config "$proteus_policy" \
  --npz-playback-limit-s "$proteus_playback_limit" \
  --motion-path "$proteus_motion" \
  --record --record-output "$proteus_run/tracking.npz" > "$proteus_run/tracker.log" 2>&1 &
tracker_pid=$!
printf '%s\n' "$bridge_pid" > "$proteus_run/bridge.pid"
printf '%s\n' "$tracker_pid" > "$proteus_run/tracker.pid"
printf '%s\n' "$$" > "$proteus_run/supervisor.pid"
wait -n "$bridge_pid" "$tracker_pid"
