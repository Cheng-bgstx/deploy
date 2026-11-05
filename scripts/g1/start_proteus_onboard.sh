#!/usr/bin/env bash
# Proteus G1 on the robot's own Orin: start_proteus_pc.sh with the environment deploy_onboard.sh pushed.
# Run on the Orin (scripts/g1/onboard.sh start does it detached from SSH). Same remote flow: A -> A -> Start.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
root="$(dirname "$repo")"
export SIM2REAL_PYTHON="${SIM2REAL_PYTHON:-$root/python/bin/python3}"
export PYTHONPATH="$root/site-packages:$root/cyclonedds-py"
export CYCLONEDDS_HOME="$root/cyclonedds"
export LD_LIBRARY_PATH="$root/cyclonedds/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# eth0 is the robot LAN. Orin NX cores: Bridge alone on 1, Tracker on 2-5, the read-only gap monitor on 6 at low
# priority; 0 and 7 stay with the system and the robot's own services.
export G1_INTERFACE="${G1_INTERFACE:-eth0}"
export PROTEUS_BRIDGE_CPUS="${PROTEUS_BRIDGE_CPUS:-1}"
export PROTEUS_TRACKER_CPUS="${PROTEUS_TRACKER_CPUS:-2-5}"
export PROTEUS_MONITOR_CPUS="${PROTEUS_MONITOR_CPUS:-6}" PROTEUS_MONITOR_NICE="${PROTEUS_MONITOR_NICE:-10}"
export SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-4}"
# The handset comes from the Bridge over ZMQ: a Tracker-side DDS subscription to 1 kHz LowState took most of a core.
export PROTEUS_CONTROLLER="${PROTEUS_CONTROLLER:-bridge_joystick}"
exec "$repo/scripts/g1/start_proteus_pc.sh"
