---
title: Proteus WAMSeq on LimX OLI
slug: /tutorials/proteus-oli
---

# Proteus WAMSeq on LimX OLI

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
