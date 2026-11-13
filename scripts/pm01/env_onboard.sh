#!/usr/bin/env bash
# Source on the PM01 motion computer (Nezha, user@192.168.0.163) inside the deployed tree:
#   source <root>/repo/scripts/pm01/env_onboard.sh
# Uses the robot's own ROS Humble and its installed interface_protocol (the firmware's message
# definitions, so no wire-schema mismatch), the Python packages shipped by deploy_onboard.sh, and
# offline Hugging Face assets. Nothing here starts, stops or reconfigures robot software.
#   PROTEUS_ROOT          deployed tree (default: three levels above this file)
#   PM01_ROS_OVERLAY      interface_protocol overlay (default: the vendor install under /apps)
#   PM01_ROS_DOMAIN_ID    DDS domain (default 69, the vendor's)
#   PM01_CYCLONEDDS_URI   optional CycloneDDS config; unset keeps the vendor default
pm01_root="${PROTEUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
pm01_overlay="${PM01_ROS_OVERLAY:-/apps/engineai_robotics/_install/share/interface_protocol/local_setup.bash}"
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "Missing /opt/ros/humble/setup.bash: run this on the PM01 motion computer" >&2
  return 2
fi
# ROS setup scripts read unset variables; source them before any `set -u` in the caller.
source /opt/ros/humble/setup.bash
if [[ -f "$pm01_overlay" ]]; then
  source "$pm01_overlay"
fi
if ! python3 -c 'import interface_protocol.msg' 2>/dev/null; then
  # Older layouts register the message package only through the whole vendor install.
  for f in /apps/engineai_robotics/_install/share/*/local_setup.bash; do source "$f"; done
fi
if ! python3 -c 'from interface_protocol.msg import JointCommand, JointState, ImuInfo, MotionState' 2>/dev/null; then
  echo "interface_protocol is not importable; set PM01_ROS_OVERLAY to the robot's message install" >&2
  return 2
fi
export ROS_DOMAIN_ID="${PM01_ROS_DOMAIN_ID:-69}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [[ -n "${PM01_CYCLONEDDS_URI:-}" ]]; then export CYCLONEDDS_URI="$PM01_CYCLONEDDS_URI"; fi
# Shipped packages first, then the repo; ROS keeps its own entries behind them for rclpy.
export PYTHONPATH="$pm01_root/site-packages:$pm01_root/repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$pm01_root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 ORT_LOGGING_LEVEL=3
# Batch-one CPU inference: one ONNX Runtime thread, no BLAS thread pools next to the vendor controller.
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-1}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PROTEUS_ROOT="$pm01_root"
unset pm01_root pm01_overlay
