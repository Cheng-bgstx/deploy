#!/usr/bin/env bash
# Read-only: AimDK joint/IMU feedback and ONNX timing on the policy core. Publishes nothing.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$repo"
policy_dir="${PROTEUS_POLICY_DIR:-checkpoints/proteus/proteus_wamseq_62/lingxi_x2}"
exec taskset -c "${PROTEUS_TRACKER_CPU:-7}" python3 -u scripts/x2/state_probe.py --seconds 5 --policy "$policy_dir/policy.onnx"
