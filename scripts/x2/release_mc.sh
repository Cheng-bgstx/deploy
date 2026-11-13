#!/usr/bin/env bash
# Release the native motion controller for the bridge takeover: stop the mc app and confirm mc_main has exited.
# "aima em stop-app mc" can take more than 20 s to return on the X2 Basic (fw 0.8.2), and it fails harmlessly when
# mc is already stopped, so success is judged by the mc_main process, not by the command's exit status.
set -uo pipefail
if ! command -v aima >/dev/null; then
  export AGIBOT_HOME=""
  source /agibot/software/entry/cfg/basic_env.sh >/dev/null 2>&1
fi
if pgrep -x mc_main >/dev/null; then
  timeout 50 aima em stop-app mc
  echo "aima em stop-app mc: exit $?"
fi
for _ in $(seq 100); do
  if ! pgrep -x mc_main >/dev/null; then
    echo 'mc_main is not running'
    exit 0
  fi
  sleep 0.1
done
echo 'mc_main is still running' >&2
exit 1
