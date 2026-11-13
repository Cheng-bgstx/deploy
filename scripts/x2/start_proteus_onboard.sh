#!/usr/bin/env bash
# AgiBot X2 onboard deployment: AimDK bridge + Proteus Tracking on the motion-control board (RK3588).
# Same flow as scripts/et1/start_proteus_onboard.sh. Startup only reads; a ready operator A requests the
# takeover (aima em stop-app mc), never this script.
#   PROTEUS_POLICY_DIR=<dir>  exported actor (default proteus_wamseq_62/lingxi_x2)
#   PROTEUS_POLICY_FILE=<yaml> config in that dir (default policy-pc-damped.yaml: arm kd x5, leg/waist kd x2,
#                             see scripts/x2/make_damped_config.py; policy-pc.yaml has the training gains)
#   PROTEUS_MOTION=<dir>      motion dataset (default: outputs/x2/wave/21_wave_right, standing right-arm wave;
#                             outputs/x2/pico_hold for standing only, outputs/x2/lafan/15_walk_short for a 1.4 m
#                             forward walk that stops, outputs/x2/lafan/1x_* for the longer LAFAN windows)
#   PROTEUS_READ_ONLY=1       bridge never takes over or writes the HAL (dry run, mc may keep running)
#   PROTEUS_PLAYBACK_LIMIT_S=<s>  stop the reference clock there and hold
#   X2_NO_HEAD=1              the HAL has no head joint group
# Operator: second ssh -t session running scripts/x2/operator.sh (a, a, s; space stops).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$repo"
export SIM2REAL_IO_TIMEOUT_S="${PROTEUS_IO_TIMEOUT_S:-0.2}"
# isolcpus=5-7: core 4 runs hal_ethercat, core 6 hosts AgiBot DDS threads; 5 and 7 are free A76 cores.
bridge_cpu="${PROTEUS_BRIDGE_CPU:-5}" tracker_cpu="${PROTEUS_TRACKER_CPU:-7}"
policy="${PROTEUS_POLICY_DIR:-checkpoints/proteus/proteus_wamseq_62/lingxi_x2}/${PROTEUS_POLICY_FILE:-policy-pc-damped.yaml}"
motion="${PROTEUS_MOTION:-outputs/x2/wave/21_wave_right}"
for f in "$policy" "$motion/manifest.json"; do
  [[ -e "$f" ]] || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p "$root/logs"
exec 9>"$root/logs/active.lock"
flock -n 9 || { echo 'A Proteus X2 supervisor is already running.' >&2; exit 1; }
if pgrep -af 'sim2real.rl_policy.tracking|sim2real/rl_policy/tracking.py|scripts/x2/aimdk_bridge.py'; then
  echo 'Another Proteus controller is running; stop it first.' >&2
  exit 1
fi
python3 -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); assert c["proteus"]["reference_forcing"]; print("policy ok")' "$policy"
run="${PROTEUS_RUN_DIR:-$root/logs/run-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$run"
bridge_args=(--timeout-s "$SIM2REAL_IO_TIMEOUT_S" --state-log "$run/bridge-state.json")
[[ "${X2_NO_HEAD:-0}" != 1 ]] || bridge_args+=(--no-head)
if [[ "${PROTEUS_READ_ONLY:-0}" == 1 ]]; then
  bridge_args+=(--read-only)
  [[ -z "${PROTEUS_MAX_RUNTIME_S:-}" ]] || bridge_args+=(--max-runtime-s "$PROTEUS_MAX_RUNTIME_S")
fi
printf 'Logs: %s\nPolicy: %s\nMotion: %s\nRead-only: %s; I/O tolerance %s s; CPUs bridge %s, tracker %s\n' \
  "$run" "$policy" "$motion" "${PROTEUS_READ_ONLY:-0}" "$SIM2REAL_IO_TIMEOUT_S" "$bridge_cpu" "$tracker_cpu"
printf 'Wait for "Tracker and LowState ready". a stand (shaping) -> lower the gantry until the feet are flat and the robot\n'
printf 'is upright, rope still taut -> a (2 s transition, the policy takes over) -> slacken the rope -> s.\n'
printf 'Never press the second a while the robot hangs tilted or with its feet off the floor.\n'
printf 'space latches the stop (damping). The first a also runs "aima em stop-app mc"; restart it with "aima em start-app mc".\n'
if pgrep -x mc_main >/dev/null; then
  printf 'mc is running (fresh boot): stopping it takes ~15 s, so the first a releases mc and then latches damping.\n'
  printf 'After that, quit this launcher (Ctrl-C), start it again and press a: the takeover is then immediate.\n'
fi
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
taskset -c "$bridge_cpu" python3 -u scripts/x2/aimdk_bridge.py "${bridge_args[@]}" > "$run/bridge.log" 2>&1 &
bridge_pid=$!
taskset -c "$tracker_cpu" python3 -u sim2real/rl_policy/tracking.py --robot x2 --robot-io zmq \
  --inference-backend onnx-cpu --controller bridge_joystick --policy-config "$policy" --motion-path "$motion" \
  --record --record-output "$run/tracking.npz" "${tracker_args[@]}" > "$run/tracker.log" 2>&1 &
tracker_pid=$!
printf '%s\n' "$bridge_pid" > "$run/bridge.pid"
printf '%s\n' "$tracker_pid" > "$run/tracker.pid"
wait -n "$bridge_pid" "$tracker_pid"
