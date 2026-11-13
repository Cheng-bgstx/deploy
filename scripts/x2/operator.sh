#!/usr/bin/env bash
# Keyboard operator on the X2 board: ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/operator.sh
# a: stand (shaping); lower the gantry until the feet are flat and the robot is upright (rope taut); a again: 2 s
# reference transition, the policy takes over (Hold); slacken the rope; s: play; space: latched stop (damping).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec python3 -u "$repo/scripts/et1/operator_keyboard.py"
