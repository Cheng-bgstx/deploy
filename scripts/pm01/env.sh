#!/usr/bin/env bash
# Source in the PC shell. Build interface_protocol from this robot before use.
pm01_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pm01_setup="${PM01_ROS_SETUP:-$pm01_root/third_party/engineai_pm01/ros2/install/setup.bash}"
if [[ ! -f "$pm01_setup" ]]; then
  echo "Missing PM01 ROS messages: $pm01_setup" >&2
  echo "Run: bash scripts/pm01/prepare_ros.sh user@192.168.0.163" >&2
  return 2
fi
source /opt/ros/humble/setup.bash
source "$pm01_setup"
export ROS_DOMAIN_ID="${PM01_ROS_DOMAIN_ID:-69}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${PM01_NETWORK_INTERFACE:-eno1}\"/></Interfaces></General></Domain></CycloneDDS>}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}"
unset pm01_root pm01_setup
