---
title: Proteus WAMSeq on AgiBot Lingxi X2
slug: /tutorials/proteus-x2
---

# Proteus WAMSeq on AgiBot Lingxi X2

## Default checkpoint

Preparation, simulation and PICO scripts default to **`proteus_wamseq_62`**, under
`checkpoints/proteus/proteus_wamseq_62/lingxi_x2/`. Use `policy-sim.yaml` for measured-root
simulation; `policy.yaml` is the base of the hardware overlays. The actor uses a ten-step
WAMSeq history and validity mask with the original joint/morphology contract. Copy the
complete policy directory (ONNX, YAMLs and contract JSON) with the code to the deployment
host; existing motion-library paths remain unchanged. Each export records the source and
artifact SHA256s in its `comparison.json`. Other releases (for example `wsq61`) remain
available through `export_policy.py --release`.

This adapter targets the Proteus **`lingxi_x2` 31-joint contract**, with the public
X2 Ultra v1.3 model and the existing sim2real Tracking / integrated sim2sim runtimes.
Robot aliases are `x2`, `lingxi_x2`, `agibot_x2`. The policy is the original shared
WAMSeq actor, specialized by the morphology and signed action mapping in its contract.
There is no fine-tuning.

This page covers model and policy preparation, sim2sim and PICO teleoperation. The route
run on hardware is the onboard one, with the policy on the X2 motion-control board: see
[Proteus on AgiBot Lingxi X2 (onboard)](./proteus-x2-onboard.md). The sim2sim, LAFAN and
runtime-audit results below were recorded on 2026-11-02/03 with an earlier WAMSeq actor and
do not validate `proteus_wamseq_62`; its simulation checks and gantry runs are in the
onboard tutorial.

## Prepare the model and policy

Use the root Python 3.10 environment, with MuJoCo 3.11, ONNX / CPU ONNX Runtime,
Pillow, joblib and `ffmpeg`. `--no-sync` avoids the existing unrelated ARM GPU
requirement's universal-resolution failure in this checkout.

```bash
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone AgibotTech/agibot_x2_urdf ../agibot_x2_urdf -- --depth 1
uv run --no-sync python scripts/x2/prepare_assets.py \
  --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/x2/export_policy.py --release proteus_wamseq_62 --proteus-root ../Proteus-wamseq
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
| `policy.yaml` | Reference anchor (`reference_forcing: true`) | Hardware without world-position feedback; base of the onboard overlays `policy-pc.yaml` / `policy-pc-damped.yaml` |

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

## Hardware deployment

Hardware runs use the onboard route: `scripts/x2/aimdk_bridge.py` owns the AimDK
ROS 2 HAL on the X2 motion-control board and the Tracker runs next to it with the
`policy-pc.yaml` / `policy-pc-damped.yaml` overlays of `policy.yaml`. Network setup
(tagged VLAN 200), deployment, the read-only probe, takeover, key order and safety
notes are in [Proteus on AgiBot Lingxi X2 (onboard)](./proteus-x2-onboard.md) and
`scripts/x2/README.md`.

The inline ROS 2 backend `X2RobotIO` (`sim2real/rl_policy/robot_io/x2.py`, selected
with `--robot-io inline`) is used only by `start_pico.sh robot` below and has never
been run on hardware. It subscribes to the four joint-state groups and
`/aima/hal/imu/torso/state` (the pelvis IMU despite its name), republishes the latest
command at 500 Hz and, on feedback/command timeout (>250 ms), malformed feedback or
`zero` mode, keeps only damping. The HAL itself has no command-death watchdog.

`policy.yaml`'s `robot_io_options` configures timing, the IMU topic, quaternion
`imu_to_pelvis_wxyz` and `head_pitch_enabled` (disabled by default; only head yaw
is active on the original X2 Ultra). The identity IMU rotation matches the model; a
physical robot needs its factory calibration rather than the topic's `frame_id`.
`policy.yaml` also sets `soft_joint_pos_limits`
to the AimDK target clip (training limits narrowed to the vendor limits), which the
`motion_first_frame` Shaping target requires; `policy-sim.yaml` omits them. Do not use
`policy-sim.yaml` on hardware without world localization.

Regression: `uv run --no-sync python -m pytest tests/test_x2.py tests/test_x2_bridge.py
tests/test_oli.py tests/test_proteus_observations.py tests/test_robot_io.py
tests/test_command_sender.py -q`. Runtime assets under `third_party/`, `checkpoints/`
and `outputs/` are ignored by Git.


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

The launcher also has a `robot` mode that replaces terminal 3 on a PC with ROS 2 +
AimDK sourced. It uses the inline `X2RobotIO` path described above and **has never
been run on hardware**; native mc must already be released. Hardware runs use the
onboard route instead.

```bash
# Untested on hardware: inline PC route.
bash scripts/x2/start_pico.sh robot PUBLISHER_IP
```

The launcher sets HF offline flags, CPU inference, 50 Hz policy and `robot=x2`.
`policy-sim` uses ZMQ RobotIO and `policy-sim.yaml`; `robot` uses `robot_io=inline`
and `policy.yaml`. Both connect to **TCP 28701** for motion and **TCP 5592** for
PICO buttons on `PUBLISHER_IP`; allow both ports. Trailing tracking CLI flags can
override the policy or neutral `--motion-path`. Use one publisher / simulator per
set of ports; OLI and X2 defaults cannot run concurrently on the same host. Hand
triggers remain on 5593, without an X2 hand driver in this adapter.

On body-stream loss the receiver holds the last reference; it does not automatically
select damping. Reconnect starts an aligned segment. B still works if the button
channel is alive; use the robot's independent stop path for a lost network/process.
The `X2RobotIO` limitations above (no HAL watchdog, IMU calibration) apply.
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

## Original Proteus runtime audit (2026-11-03)

After discovering the Q1/T800 simulation timestep issue, this branch was checked
against the **unchanged** Proteus `wamseq` runner at `1e0e038`. Both runners use
the same prepared robot model, shared WAMSeq weights and exact 50-Hz reference
arrays; these arrays were packed into a native `MotionLibrary` without retargeting.
Each original run was also invoked through its actual `deploy/sim2sim.py` CLI.
This controls the model revision; it does not recreate unavailable training assets.

Seed is 0, policy rate 50 Hz. Compare the first physical sample for each shared
playback frame 1..T-1 over the contract's 14 tracking bodies. The local runner
retains its 1-s initial policy Hold; the source retains its native start and
reference velocities. The last column changes only the local physics timestep.
These matched-frame diagnostics can differ slightly from historical video metrics
because of sampling phase and, for older OLI experiments, root-reference forcing.
All runs below use `policy-sim.yaml` with measured world-root translation.

| Reference | Local 5 ms (cm) | Original 0.625 ms (cm) | Local 0.625 ms (cm) |
| --- | ---: | ---: | ---: |
| walk | 4.70 | 4.38 | 4.82 |
| aiming | 3.76 | 3.64 | 3.75 |
| dance2 | 4.25 | 4.25 | 4.29 |
| gangam | 8.18 | 8.27 | 8.23 |
| relaxed | 4.87 | 4.60 | 5.06 |

There is **no analogous timestep-induced oscillation or major runtime gap** on
these references. The Gangnam dance remains about 8.2 cm in both
runners; its local/original joint-acceleration RMS is 24.02/24.24 rad/s².
A smaller local step does not improve mean position accuracy on these references.
Keep the existing X2 simulation defaults. The previously recorded 0.16487534-kg
vendor/training mass difference is unchanged.

On 399 shared physical states, proprioception, action history and validity masks
were checked against the original builder. After evaluating both on fresh FK and
unit reference quaternions, maximum task/action differences are
5.66e-07/3.96e-05; action history and masks match exactly.
The unadjusted source body-transform cache lags qpos by one physics step; this
small observation-timing difference is retained in the original rollout.

The machine-readable record is `docs/experiments/proteus-x2-runtime-audit.json`:
source/artifact hashes, exact commands, native CLI results, all three variants and
observation diagnostics. Debug scripts and actual physical-pose traces/videos are
archived under `outputs/proteus_oli_x2_audit/20261103_171600/`. The differential
checker rejects the known bad Q1/T800 captures and accepts all nine OLI/X2
references in both local resolutions. This detects the observed numerical
regression; it is not an absolute tracking-quality threshold. Runtime code,
policy weights and hardware/PICO behavior were not changed by this audit.
