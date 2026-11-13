#!/usr/bin/env bash
# PC-side control of the Proteus G1 onboard deployment (docs/reference/proteus-g1-onboard.md).
#   onboard.sh deploy   push code, policies, motions and the environment (scripts/g1/deploy_onboard.sh)
#   onboard.sh start    launch on the Orin, detached from SSH: a dropped SSH session or cable does not stop control
#   onboard.sh status   Orin controller processes and the latest Bridge state
#   onboard.sh tail     follow the latest bridge.log and tracker.log (Ctrl+C leaves the robot running)
#   onboard.sh stop     SIGTERM the launcher: orderly shutdown with the Bridge's damping sequence
#   onboard.sh logs     copy the Orin's run directories to outputs/proteus-onboard/
# start forwards PROTEUS_POLICY_DIR, PROTEUS_CLIP, PROTEUS_READ_ONLY, PROTEUS_READ_ONLY_S, PROTEUS_PLAYBACK_LIMIT_S,
# PROTEUS_IO_TIMEOUT_S, PROTEUS_HOLD_THROUGH_GAPS, PROTEUS_MOTION_PATH and SIM2REAL_ORT_NUM_THREADS.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${G1_HOST:-g1}"
target="${G1_TARGET:-/home/unitree/proteus_g1_onboard}"
runs="$target/repo/outputs/proteus-pc"
latest="\$(ls -td $runs/2026*/ 2>/dev/null | head -1)"
cmd="${1:-}"
case "$cmd" in
  deploy) exec "$repo/scripts/g1/deploy_onboard.sh" ;;
  start)
    # One writer per robot: the PC deployment must not run at the same time (read-only never writes).
    if [[ "${PROTEUS_READ_ONLY:-0}" != 1 ]] && pgrep -af '^([^ ]*/)?python[^ ]* .*(sim2real/rl_policy/tracking.py|scripts/g1/real_bridge.py)'; then
      echo 'A PC Proteus controller is running; stop it before starting onboard.' >&2
      exit 1
    fi
    env_args=()
    for name in PROTEUS_POLICY_DIR PROTEUS_CLIP PROTEUS_READ_ONLY PROTEUS_READ_ONLY_S PROTEUS_PLAYBACK_LIMIT_S \
        PROTEUS_IO_TIMEOUT_S PROTEUS_HOLD_THROUGH_GAPS PROTEUS_MOTION_PATH SIM2REAL_ORT_NUM_THREADS; do
      [[ -z "${!name:-}" ]] || env_args+=("$(printf '%s=%q' "$name" "${!name}")")
    done
    stamp="$(date +%Y%m%d_%H%M%S)"
    # Only the launcher is backgrounded, with every stream redirected, so ssh returns at once.
    ssh "$host" "mkdir -p '$target/logs' && cd '$target/repo' && \
      { setsid nohup env ${env_args[*]:-} scripts/g1/start_proteus_onboard.sh > '$target/logs/launch-$stamp.log' 2>&1 < /dev/null & }"
    sleep 3
    ssh "$host" "cat '$target/logs/launch-$stamp.log'"
    echo "Launched; onboard.sh tail follows it. Wait for 'Tracker and LowState ready' before pressing A."
    ;;
  status)
    ssh "$host" "pgrep -af 'start_proteus_pc.sh|real_bridge.py|rl_policy/tracking.py|lowstate_gap_monitor.py' | grep -v pgrep || echo 'No onboard controller running.'
      d=$latest; [ -n \"\$d\" ] && { echo \"Latest run: \$d\"; grep -v ddsi \"\$d/bridge.log\" | tail -3; }"
    ;;
  tail) ssh -t "$host" "d=$latest; tail -n 20 -F \"\$d/bridge.log\" \"\$d/tracker.log\"" ;;
  stop)
    ssh "$host" "d=$latest; [ -f \"\$d/supervisor.pid\" ] || { echo 'No run found.'; exit 0; }
      pid=\$(cat \"\$d/supervisor.pid\"); kill -TERM \"\$pid\" 2>/dev/null || { echo 'Launcher already stopped.'; exit 0; }
      for i in \$(seq 1 40); do kill -0 \"\$pid\" 2>/dev/null || break; sleep 0.25; done
      kill -0 \"\$pid\" 2>/dev/null && echo \"Launcher \$pid still running\" || echo \"Stopped \$d\""
    ;;
  logs)
    mkdir -p "$repo/outputs/proteus-onboard"
    rsync -a --bwlimit="${G1_RSYNC_BWLIMIT:-20000}" "$host:$runs/" "$repo/outputs/proteus-onboard/"
    echo "Copied to $repo/outputs/proteus-onboard/"
    ;;
  *) sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
