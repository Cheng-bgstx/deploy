#!/usr/bin/env bash
# Read-only live view while lowering the gantry: pelvis tilt, ankle deflection and knee load, once per second.
# Press the second a only when it reports UPRIGHT. Reads the running bridge's LowState over ZMQ (latest message,
# once per second), so it adds no ROS subscription; runs on the non-isolated cores 0-3.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$repo"
exec taskset -c 0-3 python3 -u scripts/x2/state_now.py --watch-bridge
