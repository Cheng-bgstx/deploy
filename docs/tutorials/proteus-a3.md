---
title: Proteus WAMSeq on AgiBot Expedition A3
slug: /tutorials/proteus-a3
---

# Proteus WAMSeq on AgiBot Expedition A3

Branch `feat/proteus-wamseq-agibot-a3` targets **`agibot_a3` (30 policy joints)**. Robot aliases are `a3` and `agibot_a3`. It specializes the released shared actor; no training or fine-tuning is required. Model, named file tracking, PICO IK/ZMQ and inline hardware I/O are included. **Software and MuJoCo are tested; physical robot and live headset operation are untested.**

## Prepare

```bash
# From this branch's repository/worktree root. Reuse existing source checkouts.
sources_root="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
git -C "$sources_root/Proteus" worktree add --detach "$sources_root/Proteus-wamseq" origin/wamseq
gh repo clone AgibotTech/A3-A3U-robot-model "$sources_root/A3-A3U-robot-model" -- --depth 1
gh repo clone YanjieZe/GMR "$sources_root/GMR" -- --depth 1
gh repo clone ubisoft/ubisoft-laforge-animation-dataset "$sources_root/ubisoft-laforge-animation-dataset" -- --depth 1
uv run --no-sync python -m scripts.a3.prepare --sources-root "$sources_root"
```

`--sources-root /path/to/projects` overrides the common checkout parent (also works from git worktrees). `--proteus-root` selects an existing `wamseq` checkout. `--skip-motions` prepares only model/policy/PICO hold. Use the established root environment for inference and `venv/pico` for teleop. The preparation script launches the offline IK subprocess in `venv/pico`; it does not install robot SDKs. Set up those environments using the existing Getting Started pages before running this recipe. `--no-sync` reuses the local environments.

Pinned sources checked: [Proteus wamseq](https://github.com/Hxxxz0/Proteus/tree/1e0e0381bdec73652f8aa90b871f2980586bb52a) `1e0e038`, [vendor model](https://github.com/AgibotTech/A3-A3U-robot-model) `589f508`, `a3_t2d5/mjcf/a3_t2d5.xml`; GMR `bb1bbe4`; LAFAN1 `9408460`. Preparation verifies all axes at the training default pose, records mass/limit differences in `third_party/agibot_a3/validation.json`, flattens MJCF includes and rewrites resources to portable relative paths. Joint parameters come from the frozen source spec under `sim2real/config/robots/specs/`. Simulation uses floor-only contacts, friction 1.0, implicitfast, zero passive joint damping/friction, contract gains, effort and armature.

`checkpoints/proteus/proteus_wamseq_62/agibot_a3/` holds one complete `policy.onnx`, the unchanged contract, comparison report and two YAMLs. `policy-sim.yaml` measures world root translation for sim2sim; `policy.yaml` uses the reference anchor for hardware without localization. Both expose `proprioception[10,68]`, `task[6,253]`, `action_history[10,31]`, `history_valid[10]`; outputs are `action[30]` and `next_slot_action[31]`. The ten-step history is oldest first; invalid rows and preceding actions are masked exactly as WAMSeq training. The wrapped graph had **zero error on 64 seed-0 comparisons** against the original actor (IR 8, opset 17). `--release proteus_wamseq_62_full` exports the other released actor.

## Motion library and validation

A3’s profiles transfer calibrated G1 human frames through the neutral FK of the actual standard A3. This includes target-link orientation and position offsets and limb/root scale, followed by IK on the A3 model. The PICO pause/hold reference also populates native MJCF addresses by joint name: A3’s MJCF traversal starts with waist/arms while its Proteus contract starts with legs. The publisher does not depend on the Ultra-specific PICO package.

`checkpoints/agibot_a3/motions/` is one any4hdmi library with multiple named NPZs and a manifest. Sources are actual LAFAN1 mocap segments, retargeted on this robot, not synthetic holds. Input 30 Hz is declared in the manifest; tracking resamples to 50 Hz. Root XY is recentered, root quaternion is WXYZ, and sole geometry supplies floor height. IK joint speed is bounded to 12 rad/s. The independent `pico_hold/` dataset is only for live startup shaping.

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot a3 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml \
  --dataset checkpoints/agibot_a3/motions --output outputs/a3/eval-new
```

Local validation (2026-11-03): seed 0, 50 Hz policy, 0.005 s physics, 1 s initial policy-active hold, each 8 s mocap excerpt to the final frame, then 1 s final hold. A run passes only when complete, finite and every playback frame’s mean error over the 14 contract bodies is at most 0.5 m. All default clips below passed. Metrics use recorded trajectories; root endpoint error here is in a common world frame. The NPZ also records displacements in separate initial root frames, which is a different measure.

| Clip | Mean world body error (cm) | Mean local body error (cm) | World root endpoint error (cm) |
| --- | ---: | ---: | ---: |
| `dance1_subject1` | 5.75 | 5.18 | 9.87 |
| `fightAndSports1_subject4` | 7.36 | 4.93 | 8.59 |
| `walk1_subject1` | 4.17 | 3.46 | 1.60 |

The machine-readable record with source/artifact hashes and commands is `docs/experiments/proteus-a3.json`. Re-running requires a new output directory. A threshold pass is an offline tracking result, not a hardware/live teleop result. T800 dance stress results, if present in the record, are excluded from its default library.

## Hardware entry

This targets the standard A3 `agibot_a3` contract using the public T2.5 model, not A3 Ultra or the T3.0 revision. The vendor has 31 motors; Proteus has 30 policy joints. Head pitch is fixed at zero in simulation and explicitly held at zero by the physical neck command. Vendor mass differs from training by 0.19383539 kg and is retained. Five vendor ranges differ; simulation uses training limits, while hardware intersects physical and training limits (especially both knees). The ROS adapter uses the official `joint_msgs` Command/JointCommand and State/JointState definitions with `/body_drive/{waist,neck,arm,leg}_joint_{state,command}` and `/body_drive/pelvis_imu/data`. Groups are mapped by name. Configure the IMU extrinsic for the installed sensor. The referenced source `sonic_for_a3@0311d71` supplies the wire schema; its SONIC policy and Ultra mechanics are not used.

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
# Release the native controller through the vendor procedure before HAL ownership.
bash scripts/a3/start_proteus.sh checkpoints/agibot_a3/motions/motions/walk1_subject1.npz
```

[Vendor SDK/interface source](https://github.com/AgibotTech/sonic_for_a3). Policy inference is 50 Hz on CPU; a separate ROS timer holds the latest command. Fresh joint/IMU feedback, finite commands and nonnegative gains are required. Stale commands/feedback (>250 ms), errors and zero mode replace stiffness, velocity and feedforward torque with zero and retain damping. The A3 codec uses damping 2 (the `NamedJointCodec` default) and a 0.2 rad per-publish position step bound. Normal shutdown sends damping repeatedly. These are process-level checks; no claim is made about firmware protection after process kill or power loss. IMU quaternion and angular velocity are converted to pelvis WXYZ/body coordinates with an explicit extrinsic. Changing control generation invalidates old commands. Keys: `i` shapes to frame 0 over 10 s, `]` enables policy Hold, Space plays/pauses, `o` damps, Ctrl+C exits. Use `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` to select a deployment config.

## PICO

```bash
# Terminal 1: calibrated PICO body/leg trackers and XRoboToolkit PC Service
bash scripts/a3/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: simulator (open its mjviser URL)
bash scripts/a3/start_pico.sh sim
# Terminal 3: policy, motion + button stream from terminal 1
bash scripts/a3/start_pico.sh policy-sim 127.0.0.1
# Hardware alternative to terminal 3, after SDK/IMU/ownership setup:
bash scripts/a3/start_pico.sh robot PUBLISHER_IP
```

The launcher sets both HF offline flags, selects the correct environment, and creates the neutral hold dataset if needed. Publisher motion uses TCP 28701; buttons use TCP 5592 (hand triggers remain on 5593, with no hand driver added here). Use one robot per default port set. **A** shapes to neutral; release, then **A+B** enables policy Hold; **X** resumes/pauses the live reference; **B alone** damps. Inspect IK in the publisher viewer before control. Disable the simulator elastic band before assessing balance. The saved profile scales live XR dimensions by assumed/actual operator height, and bounds joint motion to 12 rad/s.

The software test replaces the XR device with a known-pose fixture and exercises IK/FK, named ZMQ motion, buttons, pause/reconnect, semantic observations and complete ONNX inference at operator heights 1.6/1.8/2.0 m. ONNX stays in the root environment; IK runs in `venv/pico`. Stream loss holds the last reference; it does not automatically damp if the robot-state channel remains alive. PICO has no future samples, so the realtime buffer fills missing future references from received poses. File-tracking metrics do not establish live-control quality.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m pytest \
  tests/test_proteus_transport.py tests/test_proteus_observations.py \
  tests/test_robot_io.py tests/test_command_sender.py -q
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py -q
```

Large ONNX, vendor assets, motions and trajectories stay outside git. Rebuild with the recipe above; do not assume a remote artifact upload occurred.
