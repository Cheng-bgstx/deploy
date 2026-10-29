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
| Bridge | `scripts/g1/real_bridge.py --robot et1` | The G1 safety Bridge with the 26 joints mapped to their HG motor slots; `mode_machine` comes from the first `rt/lowstate`, and a change latches damping. |
| Deployment overlay | `checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml` | Written by `scripts/g1/prepare_proteus_deployment.py --robot et1 --limit-inset 0.01`, see [Run On The Robot](#run-on-the-robot). |
| Launcher | `scripts/et1/start_proteus.sh` | Starts the Bridge and Tracking on one machine, like the G1's `scripts/g1/start_proteus_pc.sh`. |

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

The robot side follows the [Proteus G1 PC deployment](../proteus_g1_pc.md): the
Bridge emits no motor command until a fresh Tracker and a physical A on the
robot remote, rejects any command outside the joint limits (±0.02 rad), and
latches damping on a fault or when state or commands stop for longer than the
I/O window. The G1 lessons apply to the ET1 as well.

### Deployment Overlay

```bash
uv run python scripts/g1/prepare_proteus_deployment.py --robot et1 --limit-inset 0.01 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml
```

This writes `policy-pc.yaml` next to the policy and leaves the exported actor
contract unchanged. It sets the waist roll PD to 100/2.5 (yaw unchanged), shapes
to the contract's default pose, and moves to the Motion's first frame over 2 s
before Hold. It clips targets 1% inside the joint limits and limits them to
15 rad/s. The ET1's ankle roll range is only ±0.26 rad, hence 1% rather than the
G1's 5%.

### Rehearse

The rehearsal runs the actual Tracking controller and the Bridge's
`BridgeSafety` in MuJoCo, without DDS. The root is held as by an ideal gantry
during Shaping, then lowered. It presses A, A, Start, Start (pause), Start and
Select:

```bash
PYTHONPATH=$PWD SIM2REAL_ORT_NUM_THREADS=1 uv run python scripts/g1/validate_proteus_deployment.py \
  --robot et1 --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz --walking-s 30
```

Rehearsal results on `taichi`:

| Policy | Result |
| --- | --- |
| `policy.yaml` (no overlay) | While the gantry lowers in Hold, ankle targets leave the joint limits (pitch 0.63 > 0.576, roll −0.45 < −0.26); the Bridge latches damping and the robot falls. |
| Overlay with the G1's 5% margins | No Bridge fault, but the ankle targets sit on the soft limits in the deep stances and the robot falls at Motion second 24. |
| 5% margins and the waist PD of `policy.yaml` | Falls at Motion second 16. |
| `policy-pc.yaml` (1% margins), 60 s of Motion | Passes: no fault, no rejected command, max tilt 19°, relative body error 8.6 cm, 2.4 m drift. |
| Clipping at the joint limits, the whole Motion | Passes over 132 s. |

The gantry model has no cable dynamics; the rehearsal does not certify the
hardware.

### Machine And Safety

The policy and the Bridge run either on the ET1's onboard computer or on a
machine on the robot network (`192.168.123.x`). Copy `third_party/galbot_et1/`,
`checkpoints/proteus/proteus_60/galbot_et1/` and `checkpoints/galbot_et1/` to
that checkout; the Motion dataset finds the MJCF at its relative path. Install
the Unitree SDK:

```bash
uv sync --extra inference-cpu --extra robot-g1
```

- Hang the robot in the hoist and keep the hardware stop at hand.
- Stop Loco and every et1-dep Controller on the robot
  (`./dist/cpp/loco/aarch64/tmux.sh stop`). Only one process may publish
  `rt/lowcmd`; the launcher refuses to start when it sees them on its own machine.
- Operate with the ET1 remote. The Bridge arms only on an A it reads from
  `rt/lowstate` (`wireless_remote`), so the keyboard cannot start a session.

### Read-Only Bring-Up

```bash
ET1_INTERFACE=<nic> PROTEUS_READ_ONLY=1 scripts/et1/start_proteus.sh
```

The Bridge reads `rt/lowstate` and runs the Tracker, but never writes a motor
command or releases the built-in controller (30 s, `PROTEUS_READ_ONLY_S`). In
`outputs/proteus-et1/<run>/bridge.log`, check:

- `et1 mode_machine <n> from LowState`, then `Tracker and LowState ready`;
- press A: `Bridge state: arming`; press Select: `Bridge state: Select`. This
  confirms that the remote layout is the Unitree one the Bridge decodes
  (keys at byte 2, A bit 8, Select bit 3, Start bit 2);
- `bridge-state.json` (written on exit): the joint positions match the robot's
  pose in `sim2real/config/robots/et1.py` joint order.

### Protected Session

```bash
ET1_INTERFACE=<nic> scripts/et1/start_proteus.sh
```

The launcher limits ONNX to one thread, gives both processes a 250 ms I/O window
(`PROTEUS_IO_TIMEOUT_S`) and records to `outputs/proteus-et1/<run>/`. It plays
the first 8 s of `taichi` (`PROTEUS_MOTION_PATH`, `PROTEUS_PLAYBACK_LIMIT_S`).
Wait for `Tracker and LowState ready`, then:

| Remote | Behavior |
| --- | --- |
| First A | Release the built-in controller and shape to the default pose over 10 s, in the hoist. |
| Second A, after Shaping | Move to the Motion's first frame over 2 s, reset the histories and enter Hold. Wait for `Hold ready`. |
| Lower the hoist | Only now let the feet carry the robot, still protected. Lowering before Hold failed on the G1. |
| Start | Play / pause the Motion. At the playback limit the Tracker holds the last frame. |
| Select | Software stop with priority: targets follow the measured joints with the gains kept; the robot is no longer actively balanced. |

After Select or a fault, restart with A, A, Start. Ctrl+C or SIGTERM stops both
processes, and the Bridge sends a short damping sequence. A killed process, a
lost cable or a dead PC is not an emergency stop: the robot-side timeout and the
hardware stop must be validated on site.

Limits of this path:

- Not tested on hardware yet.
- `--robot-io inline` is not available for the ET1: `unitree_interface` only
  knows the G1.
- Local tracking drifts (2.4 m over 60 s in the rehearsal), and the head stays
  locked.
- The policy observes the pelvis IMU of `rt/lowstate` (the MJCF `imu` site is on
  `pelvis_link`).
