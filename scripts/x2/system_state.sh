#!/usr/bin/env bash
# No arguments: read-only query. --target explicitly requests a system migration.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /opt/ros/humble/setup.bash
source "${X2_AIMDK_SETUP:-third_party/aimdk_x2_ros_v082/install/setup.bash}"
set -u
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${X2_FASTDDS_PROFILE:-$PWD/scripts/x2/fastdds_host.xml}"
exec uv run --no-sync python scripts/x2/system_state.py "$@"
