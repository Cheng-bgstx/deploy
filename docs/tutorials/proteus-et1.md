---
title: Proteus on Galbot ET1
slug: /tutorials/proteus-et1
---

# Proteus on Galbot ET1

This tutorial deploys the multi-robot
[Proteus](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleTrack#multi-robot-training-proteus)
actor on the Galbot ET1 with sim2real's own runtime instead of et1-dep's Loco.
It covers what is adapted and why, how to generate the artifacts, how to check
them in sim2sim, and how to run them on the robot.

:::warning
The ET1 is not one of the 60 robots Proteus was trained on. Its contract is
derived from the Unitree R1's (see below), and the result has been checked in
sim2sim only. Treat the first hardware runs as experiments: robot in a hoist,
short Motions first.
:::

## What Is Adapted

The ET1 is derived from the Unitree R1: the same 26 joints and, in Proteus's
reference pose, the same joint axes. The vendor model weighs 31.8 kg against
the R1's 28.9 kg, and some waist, arm and head limits are narrower. Its low
level is Unitree HG DDS: et1-dep's `libhdep_bridge_galbot_et1` publishes
`rt/lowcmd` and reads `rt/lowstate` (`unitree_hg` messages) and releases the
built-in controller through the MotionSwitcher. sim2real therefore drives it
through the same Unitree Bridge as the G1, with its own motor slots.

| Piece | Where | Notes |
| --- | --- | --- |
| Robot config `et1` (alias `galbot_et1`) | `sim2real/config/robots/et1.py` | Joint order, limits, efforts, HG motor slots (slots 14, 20, 21, 27 and 28 are empty). Armatures come from the gains of the et1-dep Walker/Tracker (`kp = armature * (2π·10 Hz)²`). |
| MJCF | `third_party/galbot_et1/galbot_et1_26dof_unitree_frames.xml` | et1-dep's vendor model with the elbow and wrist bodies re-expressed in the Unitree R1 frames (the vendor model turns them by 90°). The physics is unchanged; Galbot's own retargeted Motions use these frames too. Written by `scripts/et1/prepare_mjcf.py`. |
| Proteus contract | `checkpoints/proteus/contracts/galbot_et1.json` | Built by `scripts/build_proteus_contract.py` with the R1 as template, see below. |
| Policy | `checkpoints/proteus/proteus_60/galbot_et1/` | The released actor wrapped by `scripts/export_proteus_onnx.py`, like the G1 policy. |
| Motions | `checkpoints/galbot_et1/motions/` | et1-dep Motions converted to an any4hdmi dataset by `scripts/et1/convert_motions.py`. |
| Bridge | `scripts/g1/real_bridge.py --robot et1` | Maps the 26 joints to their HG motor slots and copies `mode_machine` from `rt/lowstate`. |

The contract runs Proteus's own description code
(`ScaleTrack/scripts/multi_robot/build_robot_spec.py`) and morphology descriptor
(`supermimic/morphology.py`) on the ET1 MJCF:

- slots, signs and tracked bodies come from the ET1 geometry and must match the
  R1's (they do: all 24 driven joints keep the R1 slot and sign +1);
- the default pose and the standing hip height `h` are the R1's, because Proteus
  derives both from retargeted walking clips that do not exist for the ET1;
- stiffness, damping and velocity limits are the G1 values of each slot
  rescaled to the ET1's mass and `h`, as for every released robot, so the
  ET1's kp/kd are the R1's times the mass ratio (about 1.1);
- efforts come from the MJCF actuators and armatures from the robot config;
- the head joints are locked: no slot, held at 0 with et1-dep Loco's
  uncovered-joint gains (kp 15.71, kd 1.00), as Loco does;
- the ankle tracked body is `ankle_roll_link` (the R1 tracks a toe link the ET1
  model does not have).

## Prerequisites

Place the et1-dep delivery checkout (Galbot's ET1 deployment package with Loco)
and a Proteus checkout next to this repo (`../et1-dep`, `../Proteus`). The commands below run from the
sim2real root with the root environment:

```bash
uv sync --extra inference-cpu
```

## Generate The Artifacts

1. MJCF (meshes and the DDS joint table are copied next to it):

   ```bash
   uv run python scripts/et1/prepare_mjcf.py \
     --source ../et1-dep/storage/humanoid_description/galbot_et1
   ```

   It checks body positions, masses and the mass matrix against the vendor
   model and prints the largest differences (about 1e-12).

2. Contract and policy:

   ```bash
   uv run python scripts/build_proteus_contract.py \
     --robot et1 --name galbot_et1 --template unitree_r1
   uv run python scripts/export_proteus_onnx.py \
     --contract galbot_et1 --robot et1 --contract-dir checkpoints/proteus/contracts
   ```

   The first command prints every joint's slot, default, gains and action scale
   and writes `galbot_et1.yaml` next to the contract for review. The second
   checks the wrapper against the released actor (`comparison.json`).

3. Motions. Both et1-dep layouts are accepted: an `.npz`, or a directory with
   `meta.json` and `.npy` arrays. Name a source with `=<name>` when its file name
   is not a plain identifier. The arrays are untrusted data, so run the script
   with `python -I` and absolute paths:

   ```bash
   E=$PWD/../et1-dep/storage/motions/galbot_et1
   uv run --frozen python -I scripts/et1/convert_motions.py \
     --output-dir "$PWD/checkpoints/galbot_et1/motions" \
     --motions "$E/1010test/2-太极大师-20000轮.npz=taichi" \
       "$E/liedown_getup_data/0814getup_liedown/liedown_002/liedown_002_yanwei"
   ```

   Every converted frame's forward kinematics must reproduce the stored body
   positions and orientations; a Motion retargeted to another model is
   rejected.

## Check In Sim2sim

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz
```

Add `--headless --run-once --trajectory-output <file>.npz` to record the robot
and Reference bodies. The split-process setup works too: `sim2real/sim_env/base_sim.py
--robot et1` with `sim2real/rl_policy/tracking.py --robot et1`.

Results on `taichi` (6894 frames, 138 s), mean over the 14 tracked bodies:

| `proteus.reference_forcing` | Global body error | Root-relative body error | Max tilt | Root drift |
| --- | ---: | ---: | ---: | ---: |
| `true` (local, as on the robot) | 3.51 m | 0.038 m | 12.8° | 5.1 m backwards |
| `false` (global, sim only) | 0.12 m (max 0.20 m) | 0.033 m | 20.1° | none |

- The robot never falls, and in global mode no frame exceeds Proteus's 0.5 m
  failure threshold.
- Local tracking drifts (about 3.7 cm/s backwards on this Motion). Proteus
  never sees a root position error in this mode; the G1 port drifts in local
  mode too.
- This Motion's feet sit about 3.5 cm below the floor (median ankle height
  0.02 to 0.03 m against 0.06 m on the robot), so the robot stands up to
  about 8 cm higher than the Reference pelvis. Local tracking is unaffected.

## Run On The Robot

The policy and the Bridge run either on the ET1's onboard computer or on a
machine on the robot network (`192.168.123.x`). Copy `third_party/galbot_et1/`,
`checkpoints/proteus/proteus_60/galbot_et1/` and `checkpoints/galbot_et1/` to
that checkout; the Motion dataset finds the MJCF at its relative path. Install
the Unitree SDK:

```bash
uv sync --extra inference-cpu --extra robot-g1
```

Before you start:

- Hang the robot in the hoist.
- Stop Loco and every et1-dep Controller (`./dist/cpp/loco/aarch64/tmux.sh stop`).
  Only one process may publish `rt/lowcmd`.
- The Bridge releases the built-in controller through the MotionSwitcher, like
  Loco does.

Terminal 1, the Bridge (set `--interface` to the NIC on the robot network):

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run scripts/g1/real_bridge.py \
  --robot et1 --interface eth0
```

Terminal 2, the policy:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run sim2real/rl_policy/tracking.py \
  --robot et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz
```

As soon as the policy process runs, every joint holds its measured position
with the policy's gains (E-stop). Then, on the keyboard:

1. `i`: Shaping to the contract's default pose over 10 s.
2. Lower the hoist until the feet carry the robot, with slack left in the rope.
3. `]`: the Tracker takes over in Hold, on the Motion's first frame.
4. `Space`: Go. `Space` again pauses, and the Tracker keeps the last frame when
   the Motion ends.
5. `o`: E-stop at any time. The joints follow their measured positions and the
   robot goes compliant, so keep the hoist ready.

Limits of this path:

- `--robot-io inline` is not available for the ET1: `unitree_interface` only
  knows the G1. Use the Bridge.
- et1-dep reads the ET1 remote from `rt/lowstate` (`wireless_remote`).
  `--controller joystick` reads `rt/wirelesscontroller`, which has not been
  checked on the ET1. Use the keyboard.
- The policy observes the pelvis IMU of `rt/lowstate` (the MJCF `imu` site is on
  `pelvis_link`).
