#!/usr/bin/env bash
# Copy only the installed message definitions; never changes the robot.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
robot_host="${1:-user@192.168.0.163}"
protocol_path="${PM01_REMOTE_PROTOCOL:-/apps/engineai_robotics/_install/share/interface_protocol}"
workspace="$PWD/third_party/engineai_pm01/ros2"
package="$workspace/src/interface_protocol"
mkdir -p "$package"
# Use the robot's definitions, not a newer GitHub schema: old MotionState has
# current_motion_task_name, while the current release has current_motion_task.
rsync -a --include='msg/***' --include='srv/***' --include='action/***' \
  --include='package.xml' --exclude='*' "$robot_host:$protocol_path/" "$package/"
cat > "$package/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.8)
project(interface_protocol)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
file(GLOB messages RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "msg/*.msg" "srv/*.srv" "action/*.action")
rosidl_generate_interfaces(${PROJECT_NAME} ${messages}
  DEPENDENCIES std_msgs std_srvs geometry_msgs action_msgs)
ament_package()
CMAKE
source /opt/ros/humble/setup.bash
cd "$workspace"
CMAKE_BUILD_PARALLEL_LEVEL=4 colcon build --packages-select interface_protocol \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
echo "Ready. From the sim2real root: source scripts/pm01/env.sh"
