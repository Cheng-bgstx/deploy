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


## PICO whole-body teleoperation

Run the publisher in **`venv/pico`**, and the simulator / policy in the **root
project**. Prepare the X2 model above on both the publisher and policy computers.
Follow [PC teleop setup](../getting-started/teleop-x86-64.md) for XRoboToolkit PC
Service, PICO body/leg-tracker calibration and whole-body streaming. Clone both
SDK repos before bootstrapping `uv sync --project venv/pico --no-install-package
xrobotoolkit-sdk`, then build `bash scripts/setup/setup_xrobot_pybind.sh`.

The bundled X2 IK calibration comes from the community
[TaiE64/AgibotX2-Retargeter](https://github.com/TaiE64/AgibotX2-Retargeter), commit
`bb582d9` (Apache-2.0), rather than an AgiBot-supplied PICO profile. The source
SMPL-X body offsets are converted to XRobot axes with `Ry(pi)`. Its auxiliary
`*_toe_link` targets do not exist in the vendor model: the adapter instead uses
the calibrated LAFAN ankle-roll tasks, with ankle position and foot orientation
from the real XR skeleton. Both IK and FK use the prepared 31-joint model; no G1
joint angles are copied. The X2 solver adds a neutral shoulder/waist posture task,
uses separate offsets for the two IK stages, passes Mink limits explicitly, and
bounds consecutive joint targets to **12 rad/s** at publisher frequency. Live
positions are normalized by calibration height / measured operator height.

The profile is included in Git; its generator records calibration-file SHA256s.
To regenerate it (optional):

```bash
gh repo clone TaiE64/AgibotX2-Retargeter ../AgibotX2-Retargeter -- --depth 1
# --gmr-configs must use the original upstream GMR source frames, not LimX's newer BVH basis.
uv run --no-sync python scripts/x2/build_pico_profile.py \
  --retarget-root ../AgibotX2-Retargeter \
  --gmr-configs ../GMR/general_motion_retargeting/ik_configs
```

Generate the neutral startup reference once on the policy computer. Streaming
tracking still needs a motion file for `shaping_target: motion_first_frame`:

```bash
uv run --no-sync python scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold
```

Open **three terminals** for local sim2sim:

```bash
# 1: publisher PC; set the operator's actual measured height in metres
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70
# 2: root project MuJoCo / mjviser
bash scripts/x2/start_pico.sh sim
# 3: root project tracking with policy-sim.yaml and measured root translation
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
```

Inspect retargeting in the publisher viewer first. **A** shapes to neutral over
10 s; release, then **A+B** enables policy Hold. **X** starts the live reference;
X again returns to neutral Hold at the current XY/yaw. **B alone** selects damping.
Disable the simulator elastic band in its GUI before testing free balance. X pauses
the reference, while the policy remains active. Head targets remain neutral;
hardware head pitch retains its default disabled setting.

For hardware, replace terminal 3 with the following on the X2 policy computer:

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/setup.bash
# Release the native mc controller before taking HAL ownership, as above.
# aima em stop-app mc
bash scripts/x2/start_pico.sh robot PUBLISHER_IP
```

The launcher sets HF offline flags, CPU inference, 50 Hz policy, `robot=x2`,
`robot_io=inline` and hardware `policy.yaml`. `policy-sim` uses ZMQ RobotIO and
`policy-sim.yaml`. Both connect to **TCP 28701** for motion and **TCP 5592** for
PICO buttons on `PUBLISHER_IP`; allow both ports. Trailing tracking CLI flags can
override the policy or neutral `--motion-path`. Use one publisher / simulator per
set of ports; OLI and X2 defaults cannot run concurrently on the same host. Hand
triggers remain on 5593, without an X2 hand driver in this adapter.

On body-stream loss the receiver holds the last reference; it does not automatically
select damping. Reconnect starts an aligned segment. B still works if the button
channel is alive; use the robot's independent stop path for a lost network/process.
The AimDK ownership, IMU calibration and HAL watchdog limitations above still apply.
PICO has no future motion samples: the existing realtime buffer fills unavailable
future references from the received pose, so file-tracking results below do not
establish live teleoperation performance.

Local offline XR tests passed for OLI and X2 at operator heights 1.6, 1.8 and 2.0 m:
source-frame conversion, IK/FK joint order, motion/button messages, X pause/resume,
stream reconnection, named ZMQ reception and X2 target speed bounds. The SDK builds
and imports on this x86 PC. No physical PICO headset or X2 was connected.

```bash
# Test-only dependencies for importing the root motion buffer in the Pico env:
uv pip install --python venv/pico/.venv/bin/python pytest \
  'any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@5d0ae1915c904c64f2480a055aa1693523b9e4b4'
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py tests/test_pico_hand_grip.py -q
```

## Additional LAFAN tracking motions

The shared recipe `scripts/lafan_tracking_suite.json` adds **walk, aiming and
dance2**, taking the first 8 s of subject 1, with original 30 Hz BVH timing and
50 Hz policy resampling. Use the local dataset, or obtain `lafan1/lafan1.zip` from
[Ubisoft's source](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
and extract it into `lafan1/data`, keeping its license with the data.

```bash
gh repo clone limxdynamics/GMR ../GMR-limx -- --depth 1
uv run --no-sync python scripts/lafan_tracking_suite.py --robot x2 \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
# Optional keyboard playback of a generated reference on X2:
# bash scripts/x2/start_proteus.sh outputs/x2/lafan_suite/walk
```

Reuse an existing GMR checkout. The script uses the bundled community LAFAN
calibration with the prepared X2 model, neutral posture task, 12 rad/s joint bound,
and per-frame sole-contact grounding. It explicitly reproduces **upstream Rx(90)
left multiplication without quaternion conjugation**, the convention used by this
calibration; LimX's newer loader uses different frames. This avoids silently mixing
those two BVH conventions. No external X2 retarget patch is needed. No G1 qpos is
replayed on X2. The two head columns stay neutral.

Datasets are `outputs/x2/lafan_suite/{walk,aiming,dance2}`. Each stores source BVH
SHA256, interval, calibration/GMR revision, qpos names and grounding metadata.
Select a new `--output` after changing conversion settings; existing datasets are
reused. Videos `x2_wamseq_lafan_<name>.mp4` include 1 s initial policy Hold and a
final-frame Hold, with exact trajectories and `.metrics.json`. Seed is 0, physics
step 0.005 s, control 50 Hz, measured world root and no elastic band.

Results on 2026-11-02, during playback over the 14 contract tracking bodies:

| LAFAN clip | Mean world / pelvis-local body error | World root endpoint error |
| --- | --- | --- |
| walk1_subject1 | 4.49 / 3.38 cm | 3.72 cm |
| aiming1_subject1 | 3.75 / 3.64 cm | 1.34 cm |
| dance2_subject1 | 4.23 / 3.59 cm | 1.59 cm |

All three 10 s videos finished upright (minimum root height 0.620 / 0.641 / 0.620 m).
These add to the existing relaxed walk and Gangnam dance references. This validates
these short retargeted motion intervals in simulation, with vendor inertia / training
mass differences retained; hardware and live PICO quality still need separate tests.
