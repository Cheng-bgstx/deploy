---
title: Proteus WAMSeq on Primebot Q1
slug: /tutorials/proteus-q1
---

# Proteus WAMSeq on Primebot Q1

## Default checkpoint (2026-11-04)

The launchers now use **wsq61**, `wsq61_wam_61robots_30000.pt` from
[ModelScope Hzzzz001/Proteus](https://modelscope.cn/datasets/Hzzzz001/Proteus).
The current policy is `checkpoints/proteus/wsq61/primebot_q1/policy.yaml`;
use `policy-sim.yaml` for measured-root simulation.
The checkpoint SHA256 is `1d1c65ddf5706b940744d87e2f86f7cea453c9240bcccc12d31923211f694bba`.
The original joint/morphology contract is retained; the actor uses a ten-step WAMSeq window.
Existing artifacts remain available. The recipes and performance measurements below
refer to the earlier release unless explicitly marked wsq61; those measurements do
not validate wsq61. Local wsq61 export provenance and per-robot comparisons are in
`checkpoints/proteus/wsq61/source/export.json` and `<robot>/comparison.json`.

Branch `feat/proteus-wamseq-primebot-q1` targets **`primebot_q1` (22 policy joints)**. Robot aliases are `q1` and `primebot_q1`. It specializes the released shared actor; no training or fine-tuning is required. Model, named file tracking, PICO IK/ZMQ and inline hardware I/O are included. **Software, MuJoCo and read-only physical feedback are tested. An initial hardware Shaping trial produced crossed legs; the command-order correction has offline coverage but has not passed another hardware trial. Policy takeover and live headset operation remain unvalidated.**

## Prepare

```bash
# From this branch's repository/worktree root. Reuse existing source checkouts.
sources_root="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
git -C "$sources_root/Proteus" worktree add --detach "$sources_root/Proteus-wamseq" origin/wamseq
gh repo clone PrimebotTech/primebot-q1-urdf "$sources_root/primebot-q1-urdf" -- --depth 1
gh repo clone YanjieZe/GMR "$sources_root/GMR" -- --depth 1
gh repo clone ubisoft/ubisoft-laforge-animation-dataset "$sources_root/ubisoft-laforge-animation-dataset" -- --depth 1
uv run --no-sync python -m scripts.q1.prepare --sources-root "$sources_root"
```

`--sources-root /path/to/projects` overrides the common checkout parent (also works from git worktrees). `--proteus-root` selects an existing `wamseq` checkout. `--skip-motions` prepares only model/policy/PICO hold. Use the established root environment for inference and `venv/pico` for teleop. The preparation script launches the offline IK subprocess in `venv/pico`; it does not install robot SDKs. Set up those environments using the existing Getting Started pages before running this recipe. `--no-sync` reuses the local environments.

The default `--release wsq61` writes `checkpoints/proteus/wsq61/primebot_q1/`, which the launchers read. The artifact description and evaluation table below were measured on `proteus_wamseq_62`; prepare it with `--release proteus_wamseq_62` and select it with `PROTEUS_POLICY_CONFIG` (launchers) or `--policy-config` (evaluation).

Pinned sources checked: [Proteus wamseq](https://github.com/Hxxxz0/Proteus/tree/1e0e0381bdec73652f8aa90b871f2980586bb52a) `1e0e038`, [vendor model](https://github.com/PrimebotTech/primebot-q1-urdf) `9657817`, `xml/q1_v3.xml`; GMR `bb1bbe4`; LAFAN1 `9408460`. Preparation verifies all axes at the training default pose, records mass/limit differences in `third_party/primebot_q1/validation.json`, flattens MJCF includes and rewrites resources to portable relative paths. Joint parameters come from the frozen source spec under `sim2real/config/robots/specs/`. Simulation uses floor-only contacts, friction 1.0, implicitfast, zero passive joint damping/friction, contract gains, effort and armature.

`checkpoints/proteus/proteus_wamseq_62/primebot_q1/` holds one complete `policy.onnx`, the unchanged contract, comparison report and two YAMLs. `policy-sim.yaml` measures world root translation for sim2sim; `policy.yaml` uses the reference anchor for hardware without localization. Both expose `proprioception[10,68]`, `task[6,253]`, `action_history[10,31]`, `history_valid[10]`; outputs are `action[22]` and `next_slot_action[31]`. The ten-step history is oldest first; invalid rows and preceding actions are masked exactly as WAMSeq training. The wrapped graph had **zero error on 64 seed-0 comparisons** against the original actor (IR 8, opset 17). `--release proteus_wamseq_62_full` exports the other released actor.

## Motion library and validation

Q1 has no public vendor GMR profile in the inspected sources. Its saved BVH and XRobot tables are generated from the calibrated G1 human frames and neutral FK of the Q1 itself, including target link axes, root/limb scales and position residuals. All trajectories are solved on the Q1 model; G1 joint trajectories are never replayed. The profiles contain hashes and the regeneration method.

`checkpoints/primebot_q1/motions/` is one any4hdmi library with multiple named NPZs and a manifest. Sources are actual LAFAN1 mocap segments, retargeted on this robot, not synthetic holds. Input 30 Hz is declared in the manifest; tracking resamples to 50 Hz. Root XY is recentered, root quaternion is WXYZ, and sole geometry supplies floor height. IK joint speed is bounded to 12 rad/s. The independent `pico_hold/` dataset is only for live startup shaping.

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot q1 \
  --policy-config checkpoints/proteus/wsq61/primebot_q1/policy-sim.yaml \
  --dataset checkpoints/primebot_q1/motions --output outputs/q1/eval-new
```

Local validation (2026-11-03): seed 0, 50 Hz policy, 0.000625 s physics, 1 s initial policy-active hold, each 8 s mocap excerpt to the final frame, then 1 s final hold. A run passes only when complete, finite and every playback frame’s mean error over the 14 contract bodies is at most 0.5 m. All default clips below passed. Metrics use recorded trajectories; root endpoint error here is in a common world frame. The NPZ also records displacements in separate initial root frames, which is a different measure.

| Clip | Mean world body error (cm) | Mean local body error (cm) | World root endpoint error (cm) |
| --- | ---: | ---: | ---: |
| `dance1_subject1` | 4.88 | 2.31 | 8.46 |
| `fightAndSports1_subject4` | 4.06 | 2.63 | 4.39 |
| `walk1_subject1` | 4.46 | 1.81 | 1.74 |

The machine-readable record with source/artifact hashes and commands is `docs/experiments/proteus-q1.json`. Re-running requires a new output directory. A threshold pass is an offline tracking result, not a hardware/live teleop result. T800 dance stress results, if present in the record, are excluded from its default library.

## Simulation timestep correction

The original Proteus `deploy/sim2sim.py` was run unchanged on the same WAMSeq
weights, verified robot model and exact 50-Hz reference arrays. Q1 and A3 have
similar mean position errors across the runners; T800's old 5-ms external-PD
simulation was materially worse. On matched 50-Hz playback samples, T800's body
error was 12.72 cm locally versus 5.62 cm in the original runner. Q1's joint
acceleration RMS was 87.11 versus 33.83 rad/s², despite similar position error.

The original uses 0.625-ms physics steps and implicit joint servos. MuJoCo's
`implicitfast` integrator does not make a velocity-feedback torque computed
outside the integrator implicit. Changing only the local physics timestep to
0.625 ms reduced T800's matched-sample error to 5.84 cm and Q1's acceleration RMS
to 28.42 rad/s². Removing the initial Hold alone did not resolve the T800 gap;
using implicit servos at 5 ms did. The experiment record preserves each probe.

`evaluate.py` now defaults to **0.000625 s for Q1/T800** (override with `--sim-dt`).
The PICO simulator launcher passes `--sim-dt 0.000625 --decimation 32`; policy
inference and viewer updates remain 50 Hz. For direct integrated-simulation
commands, pass `--sim-dt 0.000625` explicitly. Both the initial and final Hold are
retained. The table above was re-run with the corrected timestep.

The regression `tests/test_proteus_sim_resolution.py` exercises the actual
checkpoint and walk reference. It failed before the correction and passes after
it, catching Q1 acceleration RMS >=60 rad/s² or T800 mean body error >=8 cm.
The source repository and ONNX weights were not changed. This correction concerns
simulation numerical stability and does not establish physical-robot performance.

## Hardware entry

The Q1 HAL uses 22 named joints in `aimdk_msgs/JointStateArray` / `JointCommandArray` on `/aima/hal/joint/state` and `/aima/hal/joint/command`. The public V3 model matches the training contract; physical zero offsets and limits still need checking on each robot. On 2026-11-04, the local robot supplied `sensor_msgs/Imu` on **`/aima/hal/imu/state`** and all 22 joint names matched. A 501-tick read-only wsq61 probe completed without stale feedback or non-finite values (observation plus ONNX mean 3.66 ms). No motor commands were sent.

The official V3 model explicitly places `imu_0` directly on **`pelvis`**, with no rotation relative to that body: [site definition](https://github.com/PrimebotTech/primebot-q1-urdf/blob/9657817ba375f41b0e5d2b62d8d6b72b757cdf4f/xml/q1_v3_mc.xml#L40-L41). Its [orientation, gyro and accelerometer sensors](https://github.com/PrimebotTech/primebot-q1-urdf/blob/9657817ba375f41b0e5d2b62d8d6b72b757cdf4f/xml/q1_v3_mc.xml#L270-L276) all reference this site. Loading the vendor MJCF confirms body `pelvis`, local position `[0, 0, 0]` and quaternion `[1, 0, 0, 0]`. Accordingly, use `imu_to_pelvis_wxyz: [1, 0, 0, 0]`; this matches the current local configuration and requires no waist-joint compensation. The public SDK text does not separately specify whether the live ROS IMU frame follows that model frame, so hardware/driver consistency remains to be verified. This gap does not make the model's IMU body unknown. Do not cancel the robot's current lean to manufacture an upright reading.

Prepare a separate hardware overlay; the exported actor, contract, default pose and PD gains remain unchanged:

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Official V3 model: imu_0 is aligned with pelvis.
uv run --no-sync python -m scripts.q1.prepare_deployment \
  --imu-topic /aima/hal/imu/state --imu-to-pelvis-wxyz 1 0 0 0 --manual-policy-entry
# On the inspected PC, source outputs/q1/source_env.sh for its wired DDS settings.
bash scripts/q1/start_proteus.sh checkpoints/primebot_q1/motions/motions/walk1_subject1.npz
```

`start_proteus.sh` and `start_pico.sh robot` select `checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml` and exit with a hint if it has not been prepared. `PROTEUS_POLICY_CONFIG` overrides it. The overlay supplies the joint limits required by `shaping_target: motion_first_frame` (`soft_joint_pos_limits`, or `shaping_joint_pos_limits` with `--policy-limit-mode pass|feedforward`) and a 15 rad/s policy target rate bound. The command above explicitly selects **manual policy entry**, as requested for the local robot: `]` enters paused policy Hold directly, without requiring Shaping completion, joint error below 0.10 rad, or speed below 0.5 rad/s. `i` remains an optional ten-second Shaping command. Startup remains disarmed; feedback checks, timeout damping, target limits and `o` stop remain active. This is operator-controlled commissioning, not hardware validation.

**Policy targets beyond the joint range (experimental).** By default the overlay clips every policy target to the model joint range twice: `soft_joint_pos_limits` in the Tracker and the position clamp in `NamedJointCodec`. Proteus trains without such a clip. At low gains (ankle kp 8.4 N·m/rad) the actor aims past a limit to obtain torque. In the 2026-11-05 walk1 recordings, the clip removed about 35% of the commanded ankle-pitch PD torque while walking and 60–73% during the stumble at 7.0–7.5 s. `--policy-limit-mode` selects how the codec handles such targets in policy mode. Shaping targets are always clipped.
- `clip` (default): send the range bound.
- `pass`: send the target unchanged. **On Q1, use it only with motions that start with the arms down.** On 2026-11-06, entering policy Hold with walk1 made the right arm swing violently: the measured right shoulder roll reached -2.28 rad against the model limit of -1.54. The vendor lists shoulder roll as a continuous joint, so nothing stops it at the model limit. walk1's first 3 s are the LAFAN T-pose, with the arms held at the shoulder-roll limit (-1.43/+1.52 against ±1.54). In the three clipped walk1 recordings, the actor's arm targets exceeded the range in 100% of those frames, by 1.4–1.7 rad. Once the reference arms were down they exceeded it in only 1–4% of frames, while leg targets exceeded it in 83–91% of walking frames. Widening the shoulder-roll limit in the contract's `morph` (an actor input) to 2.49 rad did not stop the swing on hardware.
- `feedforward`: send the range bound plus effort kp·(target − bound), faded linearly to zero over the last 0.1 rad before the limit. In range this equals the unclipped PD torque; at the limit it equals `clip`.

With `pass` or `feedforward`, the overlay writes the joint range as `shaping_joint_pos_limits` and sets `robot_io_options.policy_limit_mode`. The velocity and effort bounds, the 15 rad/s rate bound and the per-publish step bound remain.

A codec-in-the-loop simulation widened the shoulder-roll range to ±3 rad and ran a gantry Hold, release, playback and a 3 s final hold. Under `feedforward`, walk1, dance1 and fightAndSports1 finished from both initial poses without falling; `clip` fell in 4 of 6 runs and `pass` in 1. From the hardware initial pose, with the weld holding the pelvis at the recorded 29° tilt, the right shoulder roll still swung to -2.42 rad in the Hold under `feedforward`, as under `pass`. On the real gantry the pelvis settles to about 7°, and from an upright pelvis no variant reproduced the hardware arm swing in simulation. The 2026-11-04 gravity-feedforward test also suggests that the Q1 hardware path ignores `effort`, in which case `feedforward` acts as `clip`. Neither option is validated for hardware; the local overlay uses `clip`.

The local experimental export `checkpoints/proteus/wsq61_shoulder249/primebot_q1` is the same actor with the shoulder-roll abduction limit in `morph` widened from 1.54 to 2.49 rad, the median of the 61 training robots, plus a `pass` overlay. In simulation it completed all six walk1 runs without falling, against one fall with 1.54. On hardware the right arm still swung in the walk1 Hold.

`checkpoints/primebot_q1/motions_walk1_stand` holds walk1 retargeted from LAFAN 3.2–10.7 s with the same IK profile. It starts standing with the arms down, walks at about 0.45 m/s and decelerates to a stop; at the stop the right heel is raised. In the same codec simulation from an upright start, the supported Hold kept the right shoulder roll within [-0.36, -0.19] rad in every mode, with no arm target out of range. Playback with release at 15.0 or 16.0 s: `clip` fell in 2 of 2 runs, `pass` in 0 of 2. `scripts.q1.validate_deployment` with `policy-pc-pass.yaml` passed. It has not been tested on hardware.

Gravity feedforward is **off by default**. The latest hardware recording contained nonzero commanded feedforward, but reported efforts tracked the PD term alone and shoulder sag remained. Offline SDK/CDR serialization preserved `effort`; where the hardware path or reporting differs is not yet established. `--enable-shaping-gravity` explicitly enables the experimental, bounded model feedforward (two-second ramp, 10% of joint effort limits, Shaping only). Omitting `--manual-policy-entry` restores the Shaping/pose/speed entry gates. Neither option changes the exported actor or PD gains.

### Hardware command-order incident (2026-11-04)

A real `i`-only Shaping attempt produced crossed legs; the operator damped and exited without entering policy. The adapter serialized legs-first policy order, while the observed full-body HAL feedback is arms-first. The vendor's `publish_full_command` example preserves feedback order. Several post-stop joint readings were close to the targets occupying their array slots instead of their named targets (left shoulder yaw: measured 0.0545, wrong-slot target 0.0515, intended target -0.5864 rad). This strongly suggests slot-based consumption in the hardware path; firmware behavior has not been independently verified.

Q1 now preserves the latest validated full-body feedback order for **every command field**, including damping commands. Observations and policy outputs still use name-based mapping internally. The new regression feeds that measured layout into the real serializer and a slot-consuming receiver. The offline rehearsal now includes serialization too; the earlier rehearsal applied named codec values directly to MuJoCo and missed this interface fault. Passing the revised rehearsal does **not** establish that the physical fault is fixed. Do not repeat full-body Shaping as the next commissioning step: first verify a measured-pose hold and small isolated joint responses with support, known controller ownership and a stop available.

### Optional measured-pose hold and joint commissioning

Keep the vendor controller in Advanced / low-level development mode and stop other command programs. With actuation damped, ensure the gantry carries the robot's weight, the legs are uncrossed and the feet cannot constrain the test. These low gains cannot support standing. Keep the vendor stop accessible; a host keyboard cannot stop a robot across a broken network link.

```bash
cd /home/galbot/projects/sim2real  # inspected PC
source outputs/q1/source_env.sh
uv run --no-sync python -m scripts.q1.commission_joints --joint left_elbow_joint
```

This entry point uses the production Q1 transport and feedback-order serializer without loading a policy. It starts disarmed and sends no motor commands until **`h`**. Wait for fresh feedback and `publishers=1`, then:

1. Press **`h`** to capture and hold the measured joint positions. Stiffness ramps from zero to 5 over two seconds; damping is 2. It does not move toward the motion's first frame. Observe for at least three seconds. Any unexpected twist is a reason to press **`o`** immediately.
2. If the supported pose remains steady, press **`j`**. Only the selected joint target changes: `+0.03 → 0 → -0.03 → 0` rad relative to the captured pose over eight seconds (about ±1.7°). Left/right refer to the robot. Confirm physical joint identity and direction as well as encoder feedback. Other joints retain their captured targets.
3. The script damps and exits one second after the cycle; **`o`**, **Space**, **`q`**, **Esc** or **Ctrl+C** stop it earlier. **Space means stop only in this diagnostic; in the policy launcher Space starts/pauses reference playback.**

Arming is refused if joints are moving or another command publisher is visible. During the test, stale feedback, a changed wire layout/publisher count, displacement over 0.10 rad or joint speed over 0.5 rad/s stops the session. The armed deadline is 25 seconds. A probe is refused near the selected joint's model limits or if the low-gain hold drifts; do not increase gains to force it through. Publisher discovery does not replace releasing the vendor controller.

Each run saves targets, measured positions/velocities, wire order and stop reason under `outputs/q1/commission/`. `--read-only --duration-s 15` checks feedback without arming. Finishing a cycle does not automatically mark hardware validation passed. Inspect the log and physical response before testing another joint (`--joint right_elbow_joint`, then the waist/hip joints implicated in the incident). Complete joint identity/direction checks before resuming full-body Shaping and policy entry.

### Read-only IMU consistency check

```bash
source outputs/q1/source_env.sh  # inspected PC only; elsewhere source the ROS/vendor environment
uv run --no-sync python -m scripts.q1.inspect_imu --duration-s 30
```

This creates only a subscriber and uses the official model's aligned pelvis frame by default. It is an optional hardware/driver consistency check, not a prerequisite for choosing the documented model transform. The live frame label `q1_imu_state` does not itself specify a transform, and no usable IMU-to-pelvis TF arrived during the eight-second local check; neither observation overrides the vendor model definition.

For a sensor rigidly attached to the pelvis, verify the physical axes against the model: x forward, y left, z up. With actuation stopped and the robot securely supported, known small rotations should primarily appear on the corresponding gyro axis: right lean (left side up) on +x, forward lean on +y, left turn on +z. In a physically level pelvis pose, correctly transformed roll/pitch should be near zero and stationary specific force near `[0, 0, +9.81]` m/s². Gravity alone cannot identify the mounting yaw; check physical axis markings or more than one known rotation axis. Conventions: [ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst), [IMU REP 145](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst).

The captured raw IMU reported about 18.9° pitch; quaternion-predicted and measured gravity directions differed by 0.25°. This is internal consistency, not mounting calibration. The inspector's `assumed_pelvis_*` values use the supplied `--imu-to-pelvis-wxyz` and never mark it verified or change deployment configuration. Do not subtract that 18.9° merely because the torso looks upright in a photograph.

The manual-entry operator sequence (the preparation command above) is:

1. Start with the gantry carrying the robot and the vendor high-level controller released, in Advanced / low-level development mode. Before the first **`i` or `]`**, Q1 emits no motor commands, including on exit.
2. Optionally press **`i`** for ten-second Shaping. With gravity feedforward disabled, static sag may remain. The program does not require this step in manual-entry mode.
3. Press **`]`** to enter **policy Hold** directly. Histories reset, target rate limiting starts from measured joint positions, and the reference remains at frame 0. Hold is active policy control, not damping; limbs can move even while reference playback is paused.
4. Keep gantry protection for the first Hold trial. **`o`** enters damping; **Ctrl+C** exits. **Space plays/pauses reference motion; it is not a motor stop.** Start playback only after supported Hold and foot contact have been assessed. Manual mode also permits `]` after damping without another `i`.

[Vendor interface source](https://github.com/PrimebotTech/primebot-q1-sdk). Policy inference runs at 50 Hz; the ROS timer requests 1000 Hz command publication. Once armed, stale commands/feedback (>250 ms), errors and zero mode use zero stiffness, velocity and feedforward torque with damping 5. The per-publish position step bound is 0.2 rad. Normal shutdown sends damping repeatedly after arming. These are process checks, not verified firmware protection after process kill or power loss. Developer mode and an empty ROS command-publisher list do not by themselves prove the vendor internal controller has released ownership.

The actual Tracking startup, command codec and Shaping-to-Hold transition can be rehearsed without connecting to ROS:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.q1.validate_deployment
# Optional: --initial-feedback outputs/q1/preflight-20261104/ros-readonly.json
# Or: --initial-tracking-record <local-tracking-record.npz>
# Add --no-use-recorded-orientation to compare against the motion's root orientation.
```

The earlier root-reset rehearsal was invalid for supported gravity: resetting a floating root's pose every substep does not supply support forces, and missed the shoulder sag. The rehearsal now uses a mocap weld carrying body weight, the real command serializer, and 0.625 ms physics steps. Replaying the 2026-11-04 hardware joint pose and approximately 19° pelvis pitch reproduced a 0.305 rad pre-entry error without compensation; compensation reduced that error to 0.002 rad. That tilted run entered Hold but failed the free-support-release criteria (minimum height 0.280 m, upright cosine 0.776). With the same initial joints and the reference root orientation, it passed: Hold at 10.4 s, zero first-target jump, minimum free root height 0.405 m and upright cosine 0.993. These are offline results; the weld still idealizes cable mechanics and does not certify hardware release.

The corresponding hardware recording showed sustained shoulder sag (left roll 0.340 rad at kp 4.2032), with measured effort close to the model's gravity torque. Both ankle pitch errors increased after lowering before policy entry; right ankle error reached 0.351 rad while pelvis pitch increased to about 27°. A later attempt with feedforward still showed shoulder sag and never entered policy. The operator subsequently chose manual entry with feedforward disabled, bypassing the optional 0.10 rad entry gate; the earlier simulation results do not validate this new manual-entry trial.

## PICO

```bash
# Terminal 1: calibrated PICO body/leg trackers and XRoboToolkit PC Service
bash scripts/q1/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: simulator (open its mjviser URL)
bash scripts/q1/start_pico.sh sim
# Terminal 3: policy, motion + button stream from terminal 1
bash scripts/q1/start_pico.sh policy-sim 127.0.0.1
# Hardware alternative to terminal 3, after SDK/IMU/ownership setup:
bash scripts/q1/start_pico.sh robot PUBLISHER_IP
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
