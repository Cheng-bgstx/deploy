#!/usr/bin/env bash
# Galbot ET1 onboard deployment: HDEP bridge + Proteus Tracking on the robot board.
# Startup only reads; a ready operator A requests takeover (enter_debug_mode), never this script.
# Layout under PROTEUS_ROOT: repo/ site-packages/ et1-dep/storage/ hf/ probe/{lib,bridge,libet1_hdep_shim.so}
#   PROTEUS_READ_ONLY=1   bridge never enters debug mode or writes motors (dry run)
#   PROTEUS_MOTION=<dir>  motion dataset (default: 01_idle)
#   PROTEUS_PLAYBACK_LIMIT_S=<s>  stop the reference clock there and hold (e.g. 10.5 for 11_stand_walk_straight)
# Operator without a handset: ssh -t <board> <root>/repo/scripts/et1/operator.sh  (a, a, s; space stops)
set -euo pipefail
root="${PROTEUS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
repo="$root/repo"
cd "$repo"
export PYTHONPATH="$root/site-packages:$repo" ET1_DEP_ROOT="$root/et1-dep" HF_HOME="$root/hf"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 ORT_LOGGING_LEVEL=3
# Batch-one CPU inference on one A76 core leaves the others to the bridge and the board.
export SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SIM2REAL_IO_TIMEOUT_S="${PROTEUS_IO_TIMEOUT_S:-0.2}"
# Measured on the ET1 board: cores 5 and 6 run the WAM actor in 9.7 ms (p99 10.5); core 7 takes 14 ms
# and core 4 shares interrupts (p99 17 ms), so the lighter bridge goes to core 7.
bridge_cpu="${PROTEUS_BRIDGE_CPU:-7}" tracker_cpu="${PROTEUS_TRACKER_CPU:-5,6}"
# PROTEUS_POLICY_DIR: proteus_60 (default), the proteus_wamseq_62 export, or a Galbot tracking export
# (checkpoints/galbot_track/galbot_et1/<name>, with PROTEUS_MOTION pointing at its motions/<name>).
policy="${PROTEUS_POLICY_DIR:-checkpoints/proteus/proteus_60/galbot_et1}/policy-pc.yaml"
motion="${PROTEUS_MOTION:-checkpoints/proteus/proteus_60/galbot_et1/motions/01_idle}"
shim="$root/probe/libet1_hdep_shim.so"
bridge_library="$root/probe/bridge/libhdep_bridge_galbot_et1.so.2.2.0"
for f in "$policy" "$motion/manifest.json" "$shim" "$bridge_library"; do
  [[ -e "$f" ]] || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p "$root/logs"
exec 9>"$root/logs/active.lock"
flock -n 9 || { echo 'A Proteus ET1 supervisor is already running.' >&2; exit 1; }
# Exactly one motor writer: Galbot Loco, its gamepad controllers and other trackers must be stopped.
if pgrep -af '(^|/)(loco|gamepad-recover[^ ]*|motion-sender)( |$)|sim2real/rl_policy/tracking.py|scripts/et1/hdep_bridge.py' ; then
  echo 'Another ET1 controller is running; stop it before starting Proteus.' >&2
  exit 1
fi
# A Proteus export must force the root to the reference; a Galbot tracking export (galbot_track) never reads the root.
python3 -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); assert "galbot_track" in c or c["proteus"]["reference_forcing"]; print("policy ok")' "$policy"
run="${PROTEUS_RUN_DIR:-$root/logs/run-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$run"
bridge_args=()
if [[ "${PROTEUS_READ_ONLY:-0}" == 1 ]]; then
  bridge_args+=(--read-only)
  [[ -z "${PROTEUS_MAX_RUNTIME_S:-}" ]] || bridge_args+=(--max-runtime-s "$PROTEUS_MAX_RUNTIME_S")
fi
printf 'Logs: %s\nPolicy: %s\nMotion: %s\nRead-only: %s; I/O tolerance %s s; CPUs bridge %s, tracker %s\n' \
  "$run" "$policy" "$motion" "${PROTEUS_READ_ONLY:-0}" "$SIM2REAL_IO_TIMEOUT_S" "$bridge_cpu" "$tracker_cpu"
printf 'Wait for "Tracker and LowState ready". a/A stand -> a/A 2 s reference transition -> Hold ready -> lower gantry -> s/Start.\n'
printf 'space/Select or L1+Down latches the stop and drops active posture support.\n'
tracker_args=()
[[ -z "${PROTEUS_PLAYBACK_LIMIT_S:-}" ]] || tracker_args+=(--npz-playback-limit-s "$PROTEUS_PLAYBACK_LIMIT_S")
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
# The HDEP bridge library is dlopen'ed; its CycloneDDS/HDEP dependencies come from probe/lib.
LD_LIBRARY_PATH="$root/probe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
taskset -c "$bridge_cpu" python3 -u scripts/et1/hdep_bridge.py --shim "$shim" --bridge-library "$bridge_library" \
  --timeout-s "$SIM2REAL_IO_TIMEOUT_S" --state-log "$run/bridge-state.json" "${bridge_args[@]}" > "$run/bridge.log" 2>&1 &
bridge_pid=$!
taskset -c "$tracker_cpu" python3 -u sim2real/rl_policy/tracking.py --robot galbot_et1 --robot-io zmq \
  --inference-backend onnx-cpu --controller bridge_joystick --policy-config "$policy" --motion-path "$motion" \
  --record --record-output "$run/tracking.npz" "${tracker_args[@]}" > "$run/tracker.log" 2>&1 &
tracker_pid=$!
printf '%s\n' "$bridge_pid" > "$run/bridge.pid"
printf '%s\n' "$tracker_pid" > "$run/tracker.pid"
wait -n "$bridge_pid" "$tracker_pid"
