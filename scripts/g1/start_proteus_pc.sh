#!/usr/bin/env bash
# PC-only deployment. DDS startup is passive; a ready operator A requests takeover.
set -euo pipefail
proteus_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$proteus_repo"
proteus_python="${SIM2REAL_PYTHON:-$proteus_repo/.venv/bin/python}"
proteus_interface="${G1_INTERFACE:-enp5s0}"
# Stand -> straight forward walk -> stand clips at source speed (scripts/g1/prepare_proteus_motion.py), so Hold
# freezes a stand: walk_straight_short (5.1 s, 2.2 m) or walk_forward_to_stand (8.6 s, 3.6 m); in-place dances
# dance_in_place (6 s, feet down) or dance_steps (8 s, two steps); jump_once (3.2 s, one jump in place).
proteus_clip="${PROTEUS_CLIP:-walk_straight_short}"
case "$proteus_clip" in
  walk_straight_short) proteus_clip_s=5.1 ;;
  walk_forward_to_stand) proteus_clip_s=8.63 ;;
  dance_steps) proteus_clip_s=8.0 ;;
  dance_in_place) proteus_clip_s=6.0 ;;
  jump_once) proteus_clip_s=3.16 ;;
  *) echo "Unknown PROTEUS_CLIP: $proteus_clip" >&2; exit 1 ;;
esac
proteus_default_motion="$proteus_repo/checkpoints/proteus/proteus_60/unitree_g1/motions/$proteus_clip"
proteus_motion="${PROTEUS_MOTION_PATH:-$proteus_default_motion}"
proteus_playback_limit="${PROTEUS_PLAYBACK_LIMIT_S:-$proteus_clip_s}"
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
# Prepended, so an onboard environment (scripts/g1/start_proteus_onboard.sh) keeps its offline packages.
export PYTHONPATH="$proteus_repo${PYTHONPATH:+:$PYTHONPATH}"
# Keep the 50 Hz loop on performance cores: on the i9-14900HX PC, efficiency cores (16-31) stretched the Proteus
# observation to 10-35 ms, overrunning about 8% of steps; pinned to performance cores none overran. The Bridge
# and the LowState gap monitor get the other performance cores, so DDS reception is not squeezed onto two.
proteus_bridge_cpus="${PROTEUS_BRIDGE_CPUS:-0-3,6-15}"
proteus_tracker_cpus="${PROTEUS_TRACKER_CPUS:-4,5}"
proteus_monitor_cpus="${PROTEUS_MONITOR_CPUS:-$proteus_bridge_cpus}"
# 1: a LowState or Tracker command gap keeps the last PD targets and control resumes with fresh data, instead of
# latching kp=0 damping (which dropped the robot during a 343 ms reception gap). 0 restores the damping latch.
proteus_hold_through_gaps="${PROTEUS_HOLD_THROUGH_GAPS:-1}"
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
# The world action model actor by default; PROTEUS_POLICY_DIR=checkpoints/proteus/proteus_60/unitree_g1 for the older one.
proteus_policy_dir="${PROTEUS_POLICY_DIR:-checkpoints/proteus/proteus_wamseq_62/unitree_g1}"
proteus_policy="$proteus_policy_dir/policy-pc.yaml"
[[ -f "$proteus_policy" ]] || "$proteus_python" scripts/g1/prepare_proteus_deployment.py --policy-config "$proteus_policy_dir/policy.yaml"
if [[ "$proteus_motion" == "$proteus_default_motion" && ! -e "$proteus_motion" ]]; then
  "$proteus_python" scripts/g1/prepare_proteus_motion.py --clip "$proteus_clip" --output-dir "$proteus_motion"
fi
"$proteus_python" .agents/skills/configure-g1-sim2real/scripts/verify_install.py \
  --profile root-cpu --onnx "$proteus_policy_dir/policy.onnx" \
  --asset hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml
"$proteus_python" -c 'import sys; import cyclonedds; import unitree_sdk2py.core.channel; import yaml; from sim2real.config.robots import G1_CFG; from sim2real.rl_policy.utils.motion import MotionDataset; c=yaml.safe_load(open(sys.argv[2])); assert c["proteus"]["reference_forcing"]; MotionDataset.create_from_path(sys.argv[1], robot_cfg=G1_CFG); print("PC SDK2, ONNX and offline motion ready")' "$proteus_motion" "$proteus_policy"
proteus_run="${PROTEUS_RUN_DIR:-$proteus_repo/outputs/proteus-pc/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$proteus_run"
printf 'Logs: %s\n' "$proteus_run"
printf 'Policy: %s\n' "$proteus_policy"
printf 'Motion: %s; reference playback limit: %s s\n' "$proteus_motion" "$proteus_playback_limit"
printf 'ONNX CPU threads: %s; Bridge CPUs %s, Tracker CPUs %s, monitor CPUs %s\n' "$SIM2REAL_ORT_NUM_THREADS" \
  "$proteus_bridge_cpus" "$proteus_tracker_cpus" "$proteus_monitor_cpus"
if [[ "$proteus_hold_through_gaps" == 1 ]]; then
  printf 'I/O gaps over %s s: hold the last PD targets and resume with fresh data (no damping).\n' "$SIM2REAL_IO_TIMEOUT_S"
else
  printf 'I/O jitter tolerance: %s s; continue through brief gaps, latch damping on sustained loss.\n' "$SIM2REAL_IO_TIMEOUT_S"
fi
printf 'Wait for Tracker and LowState ready. A arms-down stand -> A 2s reference transition -> wait Hold ready -> lower gantry -> Start. Select loses active posture support.\n'
bridge_args=()
[[ "$proteus_hold_through_gaps" != 1 ]] || bridge_args+=(--hold-through-gaps)
if [[ "${PROTEUS_READ_ONLY:-0}" == 1 ]]; then
  bridge_args+=(--read-only --max-runtime-s "${PROTEUS_READ_ONLY_S:-5}")
fi
bridge_pid='' tracker_pid='' monitor_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$tracker_pid" ]] || kill -TERM "$tracker_pid" 2>/dev/null || true
  [[ -z "$bridge_pid" ]] || kill -TERM "$bridge_pid" 2>/dev/null || true
  [[ -z "$tracker_pid" ]] || wait "$tracker_pid" 2>/dev/null || true
  [[ -z "$bridge_pid" ]] || wait "$bridge_pid" 2>/dev/null || true
  [[ -z "$monitor_pid" ]] || kill -TERM "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
taskset -c "$proteus_bridge_cpus" "$proteus_python" -u scripts/g1/real_bridge.py --interface "$proteus_interface" \
  --timeout-s "$SIM2REAL_IO_TIMEOUT_S" --state-log "$proteus_run/bridge-state.json" "${bridge_args[@]}" > "$proteus_run/bridge.log" 2>&1 &
bridge_pid=$!
# Read-only: tells robot/network gaps (seen here too) from Bridge-process gaps (only in bridge.log).
taskset -c "$proteus_monitor_cpus" nice -n "${PROTEUS_MONITOR_NICE:-0}" "$proteus_python" -u scripts/g1/lowstate_gap_monitor.py --interface "$proteus_interface" \
  > "$proteus_run/lowstate-monitor.log" 2>&1 &
monitor_pid=$!
taskset -c "$proteus_tracker_cpus" "$proteus_python" -u sim2real/rl_policy/tracking.py --robot g1 --robot-io zmq \
  --inference-backend onnx-cpu --controller "${PROTEUS_CONTROLLER:-joystick}" --robot-interface "$proteus_interface" \
  --policy-config "$proteus_policy" \
  --npz-playback-limit-s "$proteus_playback_limit" \
  --motion-path "$proteus_motion" \
  --record --record-output "$proteus_run/tracking.npz" > "$proteus_run/tracker.log" 2>&1 &
tracker_pid=$!
printf '%s\n' "$bridge_pid" > "$proteus_run/bridge.pid"
printf '%s\n' "$tracker_pid" > "$proteus_run/tracker.pid"
printf '%s\n' "$$" > "$proteus_run/supervisor.pid"
wait -n "$bridge_pid" "$tracker_pid"
