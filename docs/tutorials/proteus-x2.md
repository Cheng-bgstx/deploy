---
title: Proteus WAMSeq on AgiBot Lingxi X2
slug: /tutorials/proteus-x2
---

# Proteus WAMSeq on AgiBot Lingxi X2

This adapter targets the Proteus **`lingxi_x2` 31-joint contract**, with the public
X2 Ultra v1.3 model and the existing sim2real Tracking / integrated sim2sim runtimes.
Robot aliases are `x2`, `lingxi_x2`, `agibot_x2`. The policy is the original shared
WAMSeq actor, specialized by the morphology and signed action mapping in its contract.
There is no fine-tuning. Two full dynamic sim2sim clips were checked; hardware is untested.

## Prepare the model and policy

Use the root Python 3.10 environment, with MuJoCo 3.11, ONNX / CPU ONNX Runtime,
Pillow, joblib and `ffmpeg`. `--no-sync` avoids the existing unrelated ARM GPU
requirement's universal-resolution failure in this checkout.

```bash
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone AgibotTech/agibot_x2_urdf ../agibot_x2_urdf -- --depth 1
uv run --no-sync python scripts/x2/prepare_assets.py \
  --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/x2/export_policy.py --proteus-root ../Proteus-wamseq
```

Reuse existing checkouts instead of cloning them again. Source revisions checked:
Proteus `1e0e038`, official description `575cc6b`. The model is
`X2_URDF-v1.3.0/x2_ultra.xml`; the vendor's newer v1.4 model is a different revision.
See the [official model guide](https://github.com/AgibotTech/agibot_x2_urdf).

`prepare_assets.py` writes `third_party/lingxi_x2/model/scene_tracking.xml`, copies
meshes/license, removes the vendor actuator/sensor blocks, and applies the training
joint limits, armatures and zero passive damping/friction. Integrated simulation
adds 31 named torque actuators and disables self-collisions, as in training.
Every joint axis at the training default pose matches exactly. Vendor total mass
**41.966473 kg** differs from the training spec's **41.80159766 kg** by **0.16487534 kg**;
the vendor inertias are retained. Twelve joint ranges differ and are overridden in
the simulation model. `validation.json` records the differences and hashes. The
AimDK backend separately enforces the intersection of vendor and training limits.

The exporter writes a single complete `policy.onnx`, the original contract,
`comparison.json`, and two YAMLs under `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/`:

| Config | Root translation | Use |
| --- | --- | --- |
| `policy-sim.yaml` | Measured world root (`reference_forcing: false`) | Sim2sim, matching the source MuJoCo observation semantics |
| `policy.yaml` | Reference anchor (`reference_forcing: true`) | SDK deployment without world-position feedback |

Both bind `proprioception[10,68]`, `action_history[10,31]`, `history_valid[10]`,
`task[6,253]`; outputs are `action[31]` and `next_slot_action[31]`. Observation
histories, validity masks, joint signs, scales and gains come from the original
contract. Across 64 seed-0 input sets the wrapped actor's output error was zero;
ONNX is IR 8, opset 17. `--release proteus_wamseq_62_full` exports the other actor.

## Dynamic references and video

The checked references are public X2 poses from
[sonic-x2](https://github.com/meetsitaram/sonic-x2), commit `c995944`. Only its motion
PKLs and explicit joint-column layout are used. Its SONIC policy and deployment
tuning are not used. The converter maps all 31 columns by name, converts source
root quaternion XYZW to WXYZ, preserves root height and recenters first-frame XY.
Input FPS is stored in the any4hdmi manifest; runtime resampling is to 50 Hz.

```bash
gh repo clone meetsitaram/sonic-x2 ../sonic-x2-reference -- --depth 1 --filter=blob:none --sparse
git -C ../sonic-x2-reference sparse-checkout set motions scripts assets/mjcf
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip gangam_dance --output outputs/x2/gangam_dance
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip relaxed_walk --output outputs/x2/relaxed_walk
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/gangam_dance --duration 11.2 \
  --output outputs/x2/x2_wamseq_gangam_dance.mp4 \
  --label 'X2 Gangnam dance reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/relaxed_walk --duration 17 \
  --output outputs/x2/x2_wamseq_relaxed_walk.mp4 \
  --label 'X2 relaxed walk reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_relaxed_walk.mp4
```

Videos use actual integrated physics on the right and kinematic reference on the
left, one shared camera, real speed, 25 fps and 1280×720. Seed is 0; policy rate is
50 Hz and physics timestep 0.005 s. Policy control stays active during the initial
1 s pause and during the final-frame hold. Each video writes its exact `.root.npz`,
`.trajectory.npz`, `.json` and `.metrics.json`, plus a JPEG preview.

Local validation (2026-11-02), computed on the 14 contract tracking bodies during playback:

| Reference | Motion / video duration | Mean world / local body position error | World root endpoint error |
| --- | --- | --- | --- |
| Gangnam dance | 9.2 / 11.2 s | 8.24 / 5.42 cm | 1.00 cm |
| Relaxed walk | 14.82 / 17 s | 4.60 / 4.05 cm | 1.72 cm |

Both completed without falling. Root height ranged 0.608–0.665 m in the dance and
0.636–0.667 m in the walk. World error compares body positions directly; local
error expresses each skeleton in its own pelvis frame. The saved root record also
contains the legacy error between displacements expressed in *separate* initial
root frames (2.46 cm dance, 45.54 cm walk); it differs from world endpoint error
because the initial pelvis orientations differ. The metrics JSON names both measures.

## AimDK deployment

The inline backend uses the documented
[AimDK joint interface](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html)
and [pelvis IMU topic](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/hal/sensor.html).
It subscribes to four joint-state arrays and `/aima/hal/imu/torso/state` (the pelvis
IMU despite its name), maps arrays by group/name and converts ROS XYZW to WXYZ.
An independent ROS timer republishes the latest command at 500 Hz; policy inference
is 50 Hz. Expired feedback/commands (>250 ms), malformed feedback and `zero` mode
send zero position stiffness, velocity and feedforward torque while retaining damping.
Normal shutdown publishes damping five times. The HAL itself has no command-death
watchdog, so this thread does not protect against process kill or computer power loss.

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/setup.bash
# On the robot, release the native controller before taking HAL ownership:
# aima em stop-app mc
bash scripts/x2/start_proteus.sh /path/to/x2/any4hdmi_dataset
```

Set the correct ROS domain/network and verify feedback before taking control.
The script sets offline flags, CPU inference, `robot=x2`, `robot_io=inline` and
keyboard control. Keys: `i` shapes to frame 0 over 10 s, `]` enables policy Hold,
Space plays/pauses, `o` damps, Ctrl+C exits. First hardware tests require support.

`policy.yaml`'s `robot_io_options` configures timing, the IMU topic, quaternion
`imu_to_pelvis_wxyz` and `head_pitch_enabled`. The identity IMU rotation matches
the model; physical deployment must use factory calibration rather than relying
on the topic's `frame_id`. Head pitch is disabled by default because only yaw is
active on the original X2 Ultra. This adapter does not establish compatibility with
other X2 revisions. Do not use `policy-sim.yaml` on hardware without world localization.

Regression: `uv run --no-sync python -m pytest tests/test_x2.py tests/test_oli.py
tests/test_proteus_observations.py tests/test_et1_proteus.py tests/test_robot_io.py
tests/test_command_sender.py -q`; syntax and shell argument checks also passed.
Runtime assets under `third_party/`, `checkpoints/` and `outputs/` are ignored by Git.
