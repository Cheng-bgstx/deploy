#!/usr/bin/env bash
# Proteus on the PM01 motion computer (Nezha), the onboard counterpart of start_proteus.sh. Run it in an
# interactive SSH terminal (the keyboard controller reads this terminal):
#   ssh -t user@192.168.0.163 proteus_pm01/repo/scripts/pm01/start_proteus_onboard.sh wave
#   ssh -t user@192.168.0.163 proteus_pm01/repo/scripts/pm01/start_proteus_onboard.sh walk
# The first argument is wave, walk or a motion .npz; further arguments go to sim2real.rl_policy.tracking.
# It first runs the read-only check (scripts/pm01/check_robot.py) and stops unless the robot is in joint_bridge
# with fresh feedback and no other command publisher. It never switches robot modes.
#   PROTEUS_POLICY_CONFIG  policy YAML (default: the proteus_wamseq_62_full hardware export)
#   PM01_ANKLE_PITCH_GAIN_SCALE  multiplies the commanded ankle-pitch kp/kd (default 0.306 = 1.25 / 4.09). The classic
#                          parser makes them about 4.09x stiffer at the joint, which rocks the pelvis at about 2 Hz and
#                          makes the wave step in simulation; 0.306 keeps the joint-side stiffness within 0.8-2x of
#                          training even if the 4.09 estimate is off. 1 sends the training gains unchanged.
#   PM01_ARM_KP_SCALE      multiplies the commanded shoulder/elbow kp (default 0.7). With a command delay the soft
#                          arm servos start a whole-body oscillation: in simulation at about 20 ms observation-to-target
#                          delay with the training gains, about 30 ms with 0.7. Onboard the delay is about 15-18 ms
#                          (12.5 ms inference on one core; two ONNX Runtime threads on CPUs 1,3 measured 26 ms).
#   PM01_TRACKER_CPU       CPU list for taskset (default 3: the Nezha boots with isolcpus=1-3 and the vendor
#                          real-time threads sit on CPUs 1-2; empty disables pinning)
#   PM01_ANKLE_KD_SCALE    multiplies the commanded ankle pitch and roll kd (default 1.5, after the pitch scale). With
#                          the feet lightly loaded the policy rocks the pelvis at 4-5 Hz (2026-11-13, taut gantry
#                          rope); in simulation 1.5 damps it and leaves full-load standing, the wave and the walk as is.
#   PM01_HAL_HZ, PM01_HAL_CPU  pace the ROS callback thread (default 200 Hz) and move it to CPU 0. Unpaced, it decodes
#                          the firmware's joint and IMU streams nonstop and doubled inference on the policy CPU
#                          (about 24 ms instead of 12.5 ms in joint_bridge on 2026-11-13). Empty disables either.
# Keys (as on the ET1): a shapes to the first frame (10 s), a again policy Hold, s play/pause, space or x stop
# (damping), Ctrl+C exit.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env_onboard.sh"
set -u
root="$PROTEUS_ROOT"
cd "$root/repo"
motion="${1:-wave}"
[[ $# -eq 0 ]] || shift
case "$motion" in
  wave) motion=checkpoints/engineai_pm01/wave/motions/wave_right.npz ;;
  walk) motion=checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz ;;
esac
policy="${PROTEUS_POLICY_CONFIG:-checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml}"
export PM01_ANKLE_PITCH_GAIN_SCALE="${PM01_ANKLE_PITCH_GAIN_SCALE:-0.306}" PM01_ARM_KP_SCALE="${PM01_ARM_KP_SCALE:-0.7}"
export PM01_ANKLE_KD_SCALE="${PM01_ANKLE_KD_SCALE:-1.5}"
# The ROS callback thread: 200 Hz on CPU 0, away from the policy (see PM01RobotIO).
export PM01_HAL_HZ="${PM01_HAL_HZ:-200}" PM01_HAL_CPU="${PM01_HAL_CPU-0}"
for f in "$policy" "$motion"; do
  [[ -f "$f" ]] || { echo "Missing $f (run scripts/pm01/deploy_onboard.sh on the PC)" >&2; exit 1; }
done
# The hardware export forces the root to the reference (no world localization on the PM01).
python3 -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); assert c["proteus"]["reference_forcing"], "policy must set reference_forcing"' "$policy"
mkdir -p "$root/logs"
exec 9>"$root/logs/active.lock"
flock -n 9 || { echo 'A Proteus PM01 process is already running from this tree.' >&2; exit 1; }
if pgrep -af 'sim2real\.rl_policy\.tracking|scripts\.pm01\.(shadow_policy|trial_policy|policy_bridge|bench_damping)' | grep -v "^$$ "; then
  echo 'Another Proteus PM01 controller is running; stop it first.' >&2
  exit 1
fi
run="$root/logs/run-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$run"
printf 'Logs: %s\nPolicy: %s\nMotion: %s\nAnkle-pitch gain scale: %s, arm kp scale: %s, ankle kd scale: %s, ROS thread: %s Hz on CPU %s\n' \
  "$run" "$policy" "$motion" "$PM01_ANKLE_PITCH_GAIN_SCALE" "$PM01_ARM_KP_SCALE" "$PM01_ANKLE_KD_SCALE" "${PM01_HAL_HZ:-unpaced}" "${PM01_HAL_CPU:-any}"
# Read-only: subscribes for 5 s, prints JSON, exits 2 unless joint_bridge, fresh feedback and no other publisher.
if ! python3 -m scripts.pm01.check_robot | tee "$run/check_robot.json"; then
  echo 'Robot not ready (see above). Enter joint_bridge from pd_stand with LB + D-pad left, then retry.' >&2
  exit 2
fi
printf 'Keys: a shape (10 s, feet free on the gantry) -> lower until the feet carry the weight -> a Hold -> s play.\n'
printf '      space or x stop (damping: does not hold a standing robot, keep the gantry ready), Ctrl+C exit.\n'
pin=()
cpu="${PM01_TRACKER_CPU-3}"
[[ -z "$cpu" ]] || pin=(taskset -c "$cpu")
exec "${pin[@]}" python3 -u -m sim2real.rl_policy.tracking \
  --robot pm01 --robot-io inline --policy-config "$policy" \
  --rl-rate 50 --inference-backend onnx-cpu --controller keyboard --keyboard-layout et1 \
  --motion-backend npz --motion-path "$motion" --record --record-output "$run/tracking.npz" "$@"
