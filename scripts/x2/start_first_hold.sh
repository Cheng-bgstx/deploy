#!/usr/bin/env bash
# First supported hardware Hold with the verified v0.8.2 message layout.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /opt/ros/humble/setup.bash
source "${X2_AIMDK_SETUP:-third_party/aimdk_x2_ros_v082/install/setup.bash}"
set -u
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS=1
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface address="10.0.1.2" multicast="true"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'

echo 'Keep gantry support and an independent stop available. This launcher only checks the system mode.'
bash scripts/x2/system_state.sh --require-develop-mc
uv run --no-sync python scripts/x2/check_feedback.py --duration 5 \
  --require-exclusive-control --require-joint-limits \
  --output outputs/x2/first_hold_preflight.json

# Starting Tracking begins damping commands; i shapes for 10 s and ] enables Hold.
# The static reference does not introduce a dynamic tracking clip.
exec uv run --no-sync python -m sim2real.rl_policy.tracking \
  --robot x2 --robot-io inline \
  --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.yaml \
  --rl-rate 50 --inference-backend onnx-cpu --controller keyboard \
  --motion-backend npz --motion-path outputs/x2/pico_hold \
  --record --record-output "outputs/x2/first_hold_$(date +%Y%m%d_%H%M%S).npz"
