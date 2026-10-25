#!/usr/bin/env bash
# Build the depth capture program on the Onboard Computer with g++ directly.
# CMakeLists.txt needs CMake >= 3.18; JetPack 5 ships 3.16. Needs librealsense2 and zlib
# development files. Outputs build/scalecar-depth (the Host's default remote_executable
# expects this directory at /home/unitree/sim2real_depth_stream) and runs the clock test.
#
#   REALSENSE_PREFIX  librealsense2 install prefix (default /opt/ros/noetic, the G1's ROS librealsense)
#   REALSENSE_LIBDIR  its library directory (default <prefix>/lib/<gcc target> if present, else <prefix>/lib)
set -euo pipefail
cd "$(dirname "$0")"
prefix="${REALSENSE_PREFIX:-/opt/ros/noetic}"
libdir="${REALSENSE_LIBDIR:-$prefix/lib/$(g++ -dumpmachine)}"
[ -d "$libdir" ] || libdir="$prefix/lib"
mkdir -p build
g++ -O2 -std=c++17 test_depth_clock.cpp -o build/depth-clock-tests
build/depth-clock-tests
g++ -O2 -std=c++17 -I"$prefix/include" depth_stream.cpp \
  -L"$libdir" -Wl,-rpath,"$libdir" -lrealsense2 -lz -o build/scalecar-depth
echo "Built $(pwd)/build/scalecar-depth; clock test passed"
