---
title: Proteus WAMSeq on LimX OLI
slug: /tutorials/proteus-oli
---

# Proteus WAMSeq on LimX OLI

## Default checkpoint

The OLI scripts default to `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/`, the actor the
runtime audit below was run on. The **wsq61** actor (`wsq61_wam_61robots_30000.pt`, SHA256
`1d1c65ddf5706b940744d87e2f86f7cea453c9240bcccc12d31923211f694bba`, provenance in
`checkpoints/proteus/wsq61/source/export.json`, records in `docs/experiments/proteus-wsq61.json`) is an
alternative: pass its `policy.yaml` explicitly. The results below do not validate wsq61.
Use `policy.yaml` for hardware and `policy-sim.yaml` for measured-root simulation, and copy the complete
policy directory (ONNX, YAMLs and contract JSON) with the code to the deployment host.

This adapter targets **HU_D04_01 / limx_oli_edu, 31 virtual PR joints**. It uses the
Proteus `wamseq` branch, the existing sim2real Tracking / integrated sim2sim runtimes,
and the official [LimX SDK](https://github.com/limxdynamics/limxsdk-lowlevel).
Local export, observation history, transport mapping, stationary/synthetic arm-wave
sim2sim and one LAFAN1 dance excerpt have been checked. Hardware remains untested.

## Prepare assets and policies

Run from the sim2real root using the existing root `.venv` (Python 3.10,
MuJoCo 3.11, ONNX and CPU ONNX Runtime). These commands use `--no-sync` because
this checkout's existing ARM GPU requirement can fail universal dependency resolution.

```bash
# Keep the original Proteus working tree unchanged.
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone limxdynamics/humanoid-description ../humanoid-description -- --depth 1
gh repo clone limxdynamics/limxsdk-lowlevel ../limxsdk-lowlevel -- --depth 1

uv run --no-sync python scripts/oli/prepare_assets.py \
  --description-root ../humanoid-description --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/export_proteus_onnx.py \
  --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 \
  --contract limx_oli_edu --robot oli

# x86_64; use python3/aarch64 on the onboard ARM computer.
uv pip install --no-deps ../limxsdk-lowlevel/python3/amd64/limxsdk-*.whl
```

`prepare_assets.py` copies the official serial `HU_D04_01_vis.xml` and meshes to
`third_party/limx_oli_edu/model/`. It checks the 31 joints, limits, axes at the
training default pose and total mass against the Proteus spec. Alternatively,
`--asset-root <SCALETRACK_ROBOT_ASSETS>` stages the training asset tree. Do not use
the parallel-linkage `HU_D04_01.xml` for sim2real FK or integrated simulation.

The exporter writes `policy.onnx`, `policy.yaml`, the original robot contract and
`comparison.json` under `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/`.
Use `--release proteus_wamseq_62_full` for the second released WAMSeq actor.
The original `proteus_60` exporter remains the default. Integrated OLI simulation disables
self-collisions and passive joint damping to match the training spec.

| Unbatched ONNX input | Layout |
| --- | --- |
| `proprioception[10,68]` | gravity, base angular velocity, signed slot joint offsets, signed velocities × 0.05 |
| `action_history[10,31]` | raw slot action immediately before each observation |
| `history_valid[10]` | oldest first; only the newest observation valid at reset |
| `task[6,253]` | reference body positions and tangent/normal rotations, absolute/relative in anchor frame, offsets `[0,1,2,3,4,16]` |

The graph freezes the morphology, returns `action[31]` in contract joint order,
and retains `next_slot_action[31]` for feedback. The shared observation core advances
once per policy step; invalid history observations/actions are zeroed. Joint targets
are `default + signed_action_scale * action`, with the original training gains.
`reference_forcing: true` supplies the reference anchor translation because the
hardware SDK provides no world root position. It therefore tests local tracking.

## Motions and local simulation

Convert an OLI packed library clip (`index.json` and `*.npy`) to an any4hdmi dataset:

```bash
uv run --no-sync python scripts/oli/convert_motion.py \
  --library /path/to/libraries/limx_oli_edu/eval --motion-index 0 \
  --output outputs/oli/clip0
```

This preserves the WXYZ articulation-root pose, maps joints by name and checks
sampled FK against the stored reference bodies. It requires 50 Hz data and a fresh
output directory. An existing OLI any4hdmi dataset can be used directly.

Without a motion library, generate a **synthetic stationary smoke reference**:

```bash
uv run --no-sync python scripts/oli/convert_motion.py \
  --standing-smoke --duration 4 --output outputs/oli/wamseq_smoke
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1
uv run --no-sync python -m sim2real.sim_env.integrated_sim2sim \
  --robot oli --policy-config checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml \
  --motion-path outputs/oli/wamseq_smoke --headless --initial-pause-s 1 \
  --max-runtime-s 6 --seed 0 --root-trajectory-output outputs/oli/root_smoke_hold.npz
```

The policy is active during the initial pause and holds the final frame after the
motion finishes. Add `--run-once` to exit at that frame. The stationary clip checks
the integration; it does not measure dynamic tracking performance.

To record the actual simulation beside the reference (MuJoCo EGL, `ffmpeg` on
PATH and Pillow required):

```bash
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/wamseq_smoke \
  --output outputs/oli/oli_wamseq_standing_sim2sim.mp4 \
  --label 'Synthetic stationary reference | smoke test only'
```

The default is a six-second, 25 fps, real-speed video: one second of initial hold,
four seconds of reference playback, then final-frame hold. It also writes a JPEG
preview and JSON metadata. Use an OLI motion dataset to record dynamic tracking.

A synthetic dynamic arm-wave test is also available:

```bash
uv run --no-sync python scripts/oli/make_wave_motion.py --output outputs/oli/wave_motion
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/wave_motion --duration 10 \
  --output outputs/oli/oli_wamseq_wave_sim2sim.mp4 \
  --label 'Synthetic dynamic reference | raise arm - wave - lower arm'
```

This creates an eight-second reference (raise left arm, wave, lower arm), with one
second each of initial/final hold in the video. It is generated from joint curves,
not captured or retargeted human motion. Seed-0 integrated simulation with the same
settings saved `outputs/oli/wave_root.npz` and `outputs/oli/wave_trajectory.npz`;
the playback root height stayed at 0.9096–0.9119 m. Mean world-frame body-position
error across all recorded model bodies was 0.0363 m. This validates this synthetic
upper-body motion only; broader motion-library performance and hardware remain untested.
The metrics came from `integrated_sim2sim` with the same policy/motion, `--headless
--initial-pause-s 1 --max-runtime-s 10 --seed 0 --root-trajectory-output
outputs/oli/wave_root.npz --trajectory-output outputs/oli/wave_trajectory.npz`.

## Retarget real LAFAN1 motion using official LimX GMR

[LimX GMR](https://github.com/limxdynamics/GMR/blob/master/README.zh-CN.md) includes
`bvh_lafan1_to_oli_edu.json` for the same serial HU_D04_01 model. The public
[Ubisoft LAFAN1 repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
provides the human BVH source. A G1-retargeted clip cannot be copied directly to OLI.
The example dance in `limxdynamics/luna-beyondmimic` targets the 27-joint Luna,
not the 31-joint OLI.

```bash
gh repo clone limxdynamics/GMR ../GMR-limx -- --depth 1
uv pip install mink daqp loop-rate-limiters
uv run --no-sync python scripts/oli/retarget_lafan.py \
  --gmr-root ../GMR-limx \
  --bvh ../ubisoft-laforge-animation-dataset/lafan1/data/dance1_subject1.bvh \
  --duration 8 --output outputs/oli/lafan_dance1
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/lafan_dance1 --duration 10 \
  --output outputs/oli/oli_wamseq_lafan_dance1_sim2sim.mp4 \
  --label 'LAFAN1 dance1_subject1 | official LimX GMR retarget'
```

The headless adapter calls the official GMR APIs, uses the staged robot model,
grounds the lowest physical sole box each frame and recenters initial root XY.
It preserves the BVH frame interval in the any4hdmi manifest; the runtime resamples
to 50 Hz. The raw GMR qpos, source hashes, IK config hash, commit and floor shifts
are retained beside the dataset. This run used GMR **86e9546** and BVH frames 0–239
(approximately eight seconds). The shared video camera follows the midpoint of
the reference and robot roots, so it retains visible drift.

Seed-0 integrated simulation (1 s initial hold, 10 s total) stayed upright:
playback root height 0.8526–0.9189 m. Mean world-frame position error across all
recorded model bodies was **0.2433 m**, and the worst frame mean was **0.5238 m**;
this excerpt has visible translation drift. These are world-space diagnostics,
not the training's local-body success metric. Exact records are
`outputs/oli/lafan_dance1_{root,trajectory}.npz`. They were written with
`integrated_sim2sim --robot oli --policy-config
checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml --motion-path
outputs/oli/lafan_dance1 --headless --initial-pause-s 1 --max-runtime-s 10 --seed 0
--root-trajectory-output outputs/oli/lafan_dance1_root.npz --trajectory-output
outputs/oli/lafan_dance1_trajectory.npz`.

## Run through the SDK

Stage `checkpoints/proteus/`, `third_party/limx_oli_edu/` and the motion dataset on
the deployment computer; these runtime artifacts are ignored by Git. Install its
architecture's SDK wheel in the root environment. Release the official stand/walk/
mimic controllers before starting this controller; two command publishers must not
own the robot simultaneously. Suspend the robot for initial hardware tests.

```bash
# 127.0.0.1 for the official SDK simulator, or the real robot's IP.
bash scripts/oli/start_proteus.sh /path/to/oli/any4hdmi_dataset 127.0.0.1
```

The launcher sets `ROBOT_TYPE=HU_D04_01`, offline environment flags, CPU inference,
50 Hz policy control and `--robot-io inline`. The generated YAML sets
`shaping_target: motion_first_frame`.
Keyboard: `i` interpolates to frame 0 over 10 seconds, `]` enables the policy in
Hold, Space starts/pauses the motion, `o` enters damping, and Ctrl+C exits. This entry
uses the keyboard controller; the Unitree joystick controller is not an OLI remote.

`OliRobotIO` subscribes to RobotState and IMU, keeps SDK WXYZ quaternions and body
angular velocity, maps the SDK's head-pitch/head-yaw order into the MJCF's
head-yaw/head-pitch order, and publishes hybrid commands at 1 kHz. Every command
sets `parallel_solve_required=True`; SDK PR→AB conversion is required for the ankles
and waist ([official mapping guide](https://github.com/limxdynamics/humanoid-rl-deploy-python/blob/main/doc/parallel_joint_mapping_en.md)).
Missing names use the documented HU_D04_01 array order. Malformed data is rejected;
feedback/command age above 250 ms and the `zero` mode reduce commands to damping.
The publisher sends damping on orderly shutdown. This is a Python publisher, not a
hard real-time controller; validate timing and SDK/firmware compatibility on hardware.

## Local validation log (2026-11-02)

Sources: Proteus `origin/wamseq` **1e0e038**, official description **d7d6a01**,
SDK **0572427** (wheel 4.1.2), official Python deploy **f799b88**.
The prepared model has 52.547359 kg total mass and zero axis/limit difference from
the training spec. Both WAMSeq actors match their wrapped ONNX outputs exactly over
64 seed-0 random input sets (including binary validity); ONNX IR 8, opset 17.

The commands above used `../Proteus-wamseq`, the official description checkout,
`outputs/oli/wamseq_smoke` (200 stationary frames), seed 0, 1 s initial pause and
6 s total simulation. `outputs/oli/root_smoke_hold.npz` records relative final root
error **0.00713 m** and root height **0.9096–0.9105 m** during playback. This log is
for the `proteus_wamseq_62` actor, not the full actor or physical hardware.

A seed-0 comparison against the source `ObservationBuilder` and `ActingWindow`
over 3 episodes × 30 randomized steps gave zero proprioception/window error and
maximum task error 1.33e-15 (`outputs/oli/observation_comparison.json`).
The 40 targeted regression tests below passed; syntax and launcher argument checks passed.

Regression checks:

```bash
uv run --no-sync python -m pytest tests/test_oli.py tests/test_proteus_observations.py \
  tests/test_et1_proteus.py tests/test_robot_io.py tests/test_command_sender.py \
  -q
uv run --no-sync python -m py_compile scripts/export_proteus_onnx.py scripts/oli/*.py \
  sim2real/config/robots/oli.py sim2real/rl_policy/robot_io/oli.py
bash -n scripts/oli/start_proteus.sh
```

The extended SONIC artifact test was also attempted; its three release cases cannot
run here because `checkpoints/sonic/{release/g1,release/smpl,v1_1/g1}` are absent.


## PICO whole-body teleoperation

Use the existing isolated `venv/pico` environment on the publisher PC, and the root
project for the policy and simulator. Prepare the OLI model and actor above on
both computers that need them. Install XRoboToolkit PC Service / Python SDK using
[Teleop setup](../getting-started/teleop-x86-64.md). Clone the two SDK repos before
`uv sync --project venv/pico`; on a fresh machine bootstrap with
`uv sync --project venv/pico --no-install-package xrobotoolkit-sdk`, then run
`bash scripts/setup/setup_xrobot_pybind.sh`. Wear the leg trackers, complete PICO
whole-body calibration, and enable body + controller streaming to the publisher PC.

The bundled `xrobot_to_oli.json` is derived from LimX GMR's calibrated
`smplx_to_oli_edu.json` at `86e9546`. It preserves OLI task weights, scales and link
position offsets, changes source names to XRobot's names, and composes every
orientation offset with `Ry(pi)`. This source-axis conversion follows the matching
G1 SMPL-X / XRobot profiles; it does not copy G1 joints. The live publisher normalizes source positions by calibration height / measured operator height.
The publisher registers
the prepared **HU_D04_01** model for both IK and FK, and publishes 31 named joints
and all model bodies in WXYZ. To regenerate the profile:

```bash
uv run --no-sync python scripts/build_xrobot_profile.py \
  --smplx-config ../GMR-limx/general_motion_retargeting/ik_configs/smplx_to_oli_edu.json \
  --output sim2real/teleop/configs/xrobot_to_oli.json
uv run --no-sync python scripts/make_hold_motion.py --robot oli --output outputs/oli/pico_hold
```

Generate the neutral reference once on the policy computer: streamed tracking
still needs a file for startup shaping (`shaping_target: motion_first_frame`).
The publisher initially sends this same RobotCfg default pose. Use **three terminals**
for local sim2sim:

```bash
# 1: publisher PC; substitute the operator's measured height in metres
bash scripts/oli/start_pico.sh publisher --actual-human-height 1.70
# 2: root project simulation; open its mjviser URL
bash scripts/oli/start_pico.sh sim
# 3: root project tracking; use policy-sim.yaml / measured world root
bash scripts/oli/start_pico.sh policy-sim 127.0.0.1
```

Inspect IK in the publisher viewer first. Press **A** to shape to neutral (10 s),
release the buttons, then **A+B** together to enable policy Hold. Press **X** to
start live reference updates; X again returns the publisher to neutral Hold at
the current XY/yaw. **B alone** selects damping. Disable the simulator's elastic
band in its GUI before evaluating free balance. X pauses reference generation;
it does not disable the policy. A/B control is independent of the motion channel.

For a separate publisher PC and OLI computer, replace the third command with:

```bash
# Policy / SDK computer: PUBLISHER_IP is the PC; ROBOT_IP is the LimX controller
bash scripts/oli/start_pico.sh robot PUBLISHER_IP ROBOT_IP
```

The launcher sets `ROBOT_TYPE=HU_D04_01`, offline HF flags, CPU inference, 50 Hz
policy, `robot_io=inline`, hardware `policy.yaml`, and both network endpoints:
motion **TCP 28701** and PICO buttons **TCP 5592**. Allow both ports between the
computers. `policy-sim` uses `robot_io=zmq` against BaseSimulator instead. Additional
tracking CLI flags can follow the positional IP arguments, including a custom
`--policy-config` or `--motion-path`. Run one publisher / simulator per set of ports.
Hand triggers remain on 5593; no OLI hand driver is added here.

If body streaming stops, the publisher stops sending live poses and the receiver
holds its last reference; this is not an automatic fall-back to damping. Reconnect
starts a new aligned segment. Use B while the button link works, or the SDK's
independent stop path if the network/process fails. Hardware ownership and IMU
requirements in the deployment section still apply.

Local validation covers robot-specific IK, FK joint order, paused/live/reconnect
markers, named ZMQ motion reception and existing button mapping with an offline XR
fixture. The Python SDK also builds/imports on this x86 PC. No PICO headset or OLI
hardware was connected for this validation; live tracking quality still needs
operator calibration and a supported first test.

## Additional LAFAN tracking motions

The checked recipe `scripts/lafan_tracking_suite.json` adds **walk, aiming and
dance2**, each the first 8 s of subject 1 at source 30 Hz. These are real human
mocap, retargeted directly to OLI by official LimX GMR, then resampled to 50 Hz.
Use the existing local LAFAN dataset or obtain `lafan1/lafan1.zip` from
[Ubisoft's source repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
and extract the BVHs into `lafan1/data`. Keep the dataset's license with the data.

```bash
# Export again to produce policy-sim.yaml as well as hardware policy.yaml
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python scripts/export_proteus_onnx.py \
  --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 --contract limx_oli_edu --robot oli
uv run --no-sync python scripts/lafan_tracking_suite.py --robot oli \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
# Play any generated reference on hardware through the existing keyboard launcher:
# bash scripts/oli/start_proteus.sh outputs/oli/lafan_suite/walk ROBOT_IP
```

Datasets live at `outputs/oli/lafan_suite/{walk,aiming,dance2}`. Each has the BVH
SHA256, interval, GMR commit / config hash, qpos names and sole-grounding metadata.
The suite reuses existing datasets; select a new `--output` directory after changing
retarget settings. Videos are `oli_wamseq_lafan_<name>.mp4` in the same parent,
with root/whole-body trajectories, metrics JSON and exact policy/motion paths.
Each 10 s clip includes 1 s initial policy Hold and a final-frame Hold. Simulation
uses seed 0, 50 Hz control, 0.005 s physics, measured root position, no elastic band.

Results on 2026-11-02 over the 14 contract tracking bodies during playback:

| LAFAN clip | Mean world / pelvis-local body error | World root endpoint error |
| --- | --- | --- |
| walk1_subject1 | 13.29 / 16.61 cm | 13.26 cm |
| aiming1_subject1 | 12.35 / 17.29 cm | 11.05 cm |
| dance2_subject1 | 13.74 / 16.91 cm | 18.11 cm |

All three completed upright (minimum root height 0.864 / 0.894 / 0.879 m).
There is visible pose/yaw error; these clips are validation references, not a claim
of hardware readiness. `policy-sim.yaml` disables reference forcing to match the
training simulation; keep `policy.yaml` on hardware without world localization.

```bash
# The SDK-free integration test also imports root motion-buffer dependencies:
uv pip install --python venv/pico/.venv/bin/python pytest \
  'any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@5d0ae1915c904c64f2480a055aa1693523b9e4b4'
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py tests/test_pico_hand_grip.py -q
```

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
| walk | 12.91 | 12.74 | 12.73 |
| aiming | 12.27 | 12.21 | 12.23 |
| dance2 | 13.55 | 12.78 | 12.51 |
| dance1 | 12.09 | 11.56 | 12.25 |

There is **no analogous timestep-induced oscillation or major runtime gap** on
these references. The original runner also has 11.56–12.78 cm
mean body error. A smaller local step changes position accuracy only modestly.
Local/original walk joint-acceleration RMS is 17.07/18.40 rad/s², unlike Q1's
87.11/33.83 rad/s² symptom. Keep the existing OLI simulation defaults. Remaining
tracking quality cannot be assigned to weights, reference retargeting or model
revision from this comparison alone.

On 399 shared physical states, proprioception, action history and validity masks
were checked against the original builder. After evaluating both on fresh FK and
unit reference quaternions, maximum task/action differences are
6.56e-07/9.54e-06; action history and masks match exactly.
The unadjusted source body-transform cache lags qpos by one physics step; this
small observation-timing difference is retained in the original rollout.

The machine-readable record is `docs/experiments/proteus-oli-runtime-audit.json`:
source/artifact hashes, exact commands, native CLI results, all three variants and
observation diagnostics. Debug scripts and actual physical-pose traces/videos are
archived under `outputs/proteus_oli_x2_audit/20261103_171600/`. The differential
checker rejects the known bad Q1/T800 captures and accepts all nine OLI/X2
references in both local resolutions. This detects the observed numerical
regression; it is not an absolute tracking-quality threshold. Runtime code,
policy weights and hardware/PICO behavior were not changed by this audit.
