---
title: Proteus WAMSeq on EngineAI PM01
slug: /tutorials/proteus-pm01
---

# Proteus WAMSeq on EngineAI PM01

Branch `feat/proteus-wamseq-engineai-pm01` supports the **24-joint PM01/pm_v2**
using the released `engineai_pm01` contract and wsq61 WAMSeq actor. Robot aliases:
`pm01`, `pm_v2`, `engineai_pm01`. This is distinct from the T800's 25-joint
contract and from `engineai_pm01_edu_official`, whose defaults and dynamics differ.

## Verified scope

On 2026-11-04, SSH inspection of the connected PM01 found the vendor package
`76445d09e967a16eb191d10aa05fe2ac5df6c19f` (2025-10-16), configured as `pm_v2`.
Its `serial_pm_v2.urdf` has 24 movable joints and matches the vendor ROS workspace
URDF exactly. SDK order is J00–J23: J12 is `J12_WAIST_YAW` and J23 is
`J23_HEAD_YAW`. The IMU site is aligned with the pelvis; deploy YAML exposes
`imu_to_pelvis_wxyz` for a different verified mounting.

The installed `MotionState` has **`current_motion_task_name`**, while the newer
public interface has `current_motion_task` and `available_transition_motions`.
The codec accepts either field, but both DDS endpoints must use matching message
definitions. Merely adding a Python field fallback does not fix a wire-schema mismatch.
`JointState`, `JointCommand`, `ImuInfo`, and `RL_PARSER=1` match the inspected firmware.

Completed: full ONNX equivalence, four-motion integrated MuJoCo evaluation,
SDK-free mapping/watchdog tests, real ROS loopback serialization with the installed
schema, and PICO IK/ZMQ/ONNX tests with a synthetic XR input fixture.
**Physical policy tracking and real PICO hardware have not been tested.** The
robot's `robotics.service` was inactive; the read-only check received no feedback
and found no command subscriber. No robot service or motor control was started.

## Prepare assets

Use the root project for inference, and `venv/pico` for IK/teleoperation. Existing
environments are reused with `--no-sync`. Place source checkouts under one parent
directory: `Proteus-wamseq`, `GMR-engineai`, and
`ubisoft-laforge-animation-dataset`. The GMR checkout needs
`assets/engineai_pm01` and the PM01 BVH/SMPL-X IK profiles.

```bash
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects
# Model + policy + neutral PICO reference only:
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects --skip-motions
```

The model, ONNX and motions are local ignored runtime artifacts. The frozen spec
preserves training defaults, gains, limits, armatures and signed wrist actions.
The source spec rounds world joint axes to four decimals, so model verification
uses an explicit `1e-4` tolerance (measured maximum `8.76e-5`); no joint-limit
overrides were needed. Vendor mass differs from the rounded spec by 0.000189 kg.
`third_party/engineai_pm01/validation.json` records source and output hashes.

`checkpoints/proteus/wsq61/engineai_pm01/` contains one complete `policy.onnx`,
the verbatim source contract, comparison report, `policy.yaml` for hardware and
`policy-sim.yaml` for measured-root simulation. Inputs are
`proprioception[10,68]`, `task[6,253]`, `action_history[10,31]`, and
`history_valid[10]`; outputs are `action[24]` and `next_slot_action[31]`.
The 10-step history is oldest first. All 64 seed-0 random comparisons against the
released actor had zero action error. Export uses IR 8/opset 17.

## PC-to-robot setup

Official SSH endpoints are `user@192.168.0.163` (motion computer) and
`ubuntu@192.168.0.162` (Orin); see the
[vendor connection configuration](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/scripts/sync_src.sh).
The PC runs the policy over Ethernet; no inference installation on Orin is required.

Build the **installed robot's** ROS messages on the PC. This only copies message
definitions and compiles locally; it does not change or start anything on the robot.

```bash
cd /path/to/sim2real
bash scripts/pm01/prepare_ros.sh user@192.168.0.163
source scripts/pm01/env.sh
uv run --no-sync python -m scripts.pm01.check_robot
```

`env.sh` selects ROS Humble, Cyclone DDS, domain 69, Ethernet interface `eno1`, the
local matching message workspace, and both Hugging Face offline exports.
Override `PM01_NETWORK_INTERFACE`, `PM01_ROS_DOMAIN_ID`, or `PM01_ROS_SETUP` as
needed before sourcing. An existing `CYCLONEDDS_URI` takes precedence.
`PM01_REMOTE_PROTOCOL` overrides the remote installed message-package path for
`prepare_ros.sh`. Do not mix incompatible interface overlays in one shell.

The check only subscribes. It exits 0 when it receives fresh 24-joint feedback,
IMU and `joint_bridge` mode, finds a compatible command subscriber, and finds no
other command publisher; otherwise it prints the missing conditions and exits 2.
An inactive service is expected to fail this check.

## Run on hardware

With the PM01 supported by a gantry and its hardware stop available, start the
existing vendor service on the motion computer if it is inactive:

```bash
ssh user@192.168.0.163
sudo systemctl start robotics.service
systemctl status robotics.service --no-pager
```

Use the vendor controller to enter `pd_stand`, then `joint_bridge`, following the
[vendor deployment instructions](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/README_CN.md).
The launchers do not switch robot modes. Back on the PC:

```bash
cd /path/to/sim2real
bash scripts/pm01/start_proteus.sh \
  checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

The launcher sources the PM01 environment and runs the read-only check before
opening the command transport. Default inference is CPU at 50 Hz; the independent
ROS sender runs at 500 Hz. Override the policy with `PROTEUS_POLICY_CONFIG`.

Keys: **i** shapes to the motion's first frame over 10 seconds; **]** enables
policy Hold; **Space** plays/pauses; **o** enters damping; **Ctrl+C** exits.
Keep gantry support through shaping and the initial Hold. Starting the transport
in `joint_bridge` sends damping before shaping, so it does not support a standing
robot by itself. An explicit mode change gives ownership back to the firmware.

The sender maps joints by name, converts serial ankle commands via `RL_PARSER=1`,
and rejects wrong dimensions/nonfinite commands. Commands or joint/IMU feedback
older than 250 ms, a mode heartbeat older than 1 s, or a latched error produce
zero stiffness/velocity/feedforward torque with damping 2. Position increments
are limited to 0.2 rad per publication. Normal shutdown repeats damping commands;
process death or power loss requires firmware-side protection.

## PICO

```bash
# PC terminal 1: XRoboToolkit PC Service and calibrated body/leg trackers
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# PC terminal 2: robot policy; publisher runs on this same PC
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

For simulation, use `start_pico.sh sim` and `start_pico.sh policy-sim 127.0.0.1`.
**A** shapes to neutral; release it, then **A+B** enables Hold; **X** plays/pauses;
**B alone** enters damping. Motion/button TCP ports are 28701/5592. Only one robot
uses the default ports at a time. Live XR scaling accounts for operator height;
IK limits joint speed to 12 rad/s. Missing future samples are padded with received
references. Losing the body stream holds the last reference while feedback stays
healthy; it does not automatically enter damping. No new hand driver is included.

## Reproduce validation

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
```

Seed 0, policy 50 Hz, physics 0.000625 s, one-second initial and final policy holds,
and eight-second real LAFAN1 excerpts retargeted to PM01. Each completed with finite
states and per-frame mean tracked-body error below 0.5 m. This is simulation evidence.

| Motion | Mean world body error (cm) | Mean local body error (cm) | World root endpoint error (cm) |
| --- | ---: | ---: | ---: |
| aiming1_subject1 | 2.60 | 1.71 | 3.97 |
| dance2_subject1 | 3.00 | 1.98 | 2.29 |
| fightAndSports1_subject4 | 4.77 | 2.01 | 7.01 |
| walk1_subject1 | 2.30 | 1.85 | 2.42 |

See `docs/experiments/proteus-pm01.json` for exact commands, hashes and trajectories.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m pytest \
  tests/test_pm01.py tests/test_proteus_transport.py tests/test_proteus_observations.py \
  tests/test_robot_io.py tests/test_command_sender.py -q
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py -q -k pm01
# ROS loopback test: isolates itself on localhost, domain 171, never the robot.
source scripts/pm01/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py -q
```
