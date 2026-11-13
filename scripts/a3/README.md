# AgiBot Expedition A3 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

Proteus WAMSeq on the standard AgiBot Expedition A3 (`agibot_a3` contract, public T2.5 model; not A3 Ultra
or the T3.0 revision). The released shared actor is exported to ONNX unchanged; no training or fine-tuning.
The Tracker (`sim2real/rl_policy/tracking.py`) runs CPU ONNX inference at 50 Hz and talks to the robot
through the official `joint_msgs` ROS 2 interface; a separate 500 Hz ROS timer republishes the latest
command. **Software and MuJoCo are tested; the physical robot and a live PICO headset are not.**

## Status

| Item | Value |
| --- | --- |
| Hardware | Untested. Software tests and MuJoCo evaluation only (`docs/experiments/proteus-a3.json`: `hardware_tested: false`, `physical_pico_tested: false`) |
| `--robot` | `a3` (alias `agibot_a3`) — `sim2real/config/robots/__init__.py` |
| Joints | 30 policy joints; the vendor robot has 31 motors. `head_pitch_joint` is not in the contract and is held at 0 |
| Robot I/O | `--robot-io inline` → `A3RobotIO` (ROS 2, `joint_msgs` Command/JointCommand, State/JointState). Topics `/body_drive/{waist,neck,arm,leg}_joint_{state,command}`, IMU `/body_drive/pelvis_imu/data` (`sensor_msgs/Imu`) |
| Tracker runs on | Any host on the robot's ROS 2 graph with ROS Humble and the vendor `joint_msgs` workspace sourced; the launchers do not choose a host |
| Default policy | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml` (hardware, reference-anchored root) and `policy-sim.yaml` (sim2sim, measured world root) |
| Default motion | `checkpoints/agibot_a3/motions/motions/walk1_subject1.npz`; PICO hold dataset `checkpoints/agibot_a3/pico_hold` |

## Files

`scripts/a3/`:

| File | Purpose |
| --- | --- |
| `prepare.py` | One-shot asset build: stages and verifies the vendor MJCF (fixing `head_pitch_joint`), exports the WAMSeq ONNX + YAMLs, writes `xrobot_to_a3.json` / `bvh_lafan1_to_a3.json`, builds the PICO hold, retargets three LAFAN1 clips. Flags `--sources-root`, `--proteus-root`, `--release` (default `proteus_wamseq_62`), `--skip-motions`. |
| `start_proteus.sh` | Hardware file playback: `tracking --robot a3 --robot-io inline --controller keyboard --motion-backend npz`. Optional first argument is the motion NPZ. |
| `start_pico.sh` | PICO teleop, one mode per terminal: `publisher`, `sim`, `policy-sim [publisher-ip]`, `robot [publisher-ip]`. |
| `__init__.py` | Makes `python -m scripts.a3.prepare` importable. |

Related files elsewhere:

- `sim2real/config/robots/a3.py` + `sim2real/config/robots/specs/agibot_a3.yaml` (via `proteus_spec.py`) — joint order, limits, gains, armature, MJCF path.
- `sim2real/rl_policy/robot_io/a3.py` — joint groups, topics, knee physical limits, fixed head pitch.
- `sim2real/rl_policy/robot_io/ros2_named.py` — shared named-joint codec: IMU extrinsic, watchdog, limit clip, 0.2 rad step bound, damping on stop.
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_a3.json` — live PICO IK (12 rad/s joint-speed bound).
- `scripts/proteus/` — shared `prepare_assets.py`, `build_retarget_profiles.py`, `retarget_lafan.py`, `evaluate.py`; plus `scripts/export_proteus_onnx.py`, `scripts/make_hold_motion.py`.
- Tests: `tests/test_proteus_transport.py`, `tests/test_proteus_observations.py`, `tests/test_robot_io.py`, `tests/test_command_sender.py`, `tests/test_pico_retarget_profiles.py`.
- Record: `docs/experiments/proteus-a3.json`.

## Setup

1. Environments: root project (`uv`) for inference/sim, `venv/pico` for the PICO publisher and offline IK.
   Launchers use `--no-sync`, so both must already be installed (see Getting Started).
2. Sources next to the repo (default parent of the git common dir): `Proteus-wamseq` (worktree of
   `Proteus` at `origin/wamseq`), `A3-A3U-robot-model`, `GMR` (YanjieZe), `ubisoft-laforge-animation-dataset`.
   Exact clone commands are in the tutorial.
3. Build the runtime artifacts (not in git):
   ```bash
   uv run --no-sync python -m scripts.a3.prepare --sources-root "$sources_root"
   ```
   Outputs: `third_party/agibot_a3/` (model + `validation.json`), `checkpoints/proteus/proteus_wamseq_62/agibot_a3/`,
   `checkpoints/agibot_a3/{motions,pico_hold}`.
4. Hardware only: install the vendor ROS 2 workspace that provides `joint_msgs`
   ([sonic_for_a3](https://github.com/AgibotTech/sonic_for_a3) supplies the wire schema) and source it with
   `/opt/ros/humble/setup.bash`. `prepare.py` does not install robot SDKs.
5. Every launcher exports `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1` and `SIM2REAL_ORT_NUM_THREADS=1`
   (if unset). Set the same in any shell where you run Python by hand.

## Adjustments

| Tunable | Current value | Where |
| --- | --- | --- |
| IMU topic | `/body_drive/pelvis_imu/data` | `robot_io_options.imu_topic` in `policy.yaml` (written by `scripts/export_proteus_onnx.py`); default in `robot_io/a3.py` |
| IMU → pelvis extrinsic | `imu_to_pelvis_wxyz: [1, 0, 0, 0]` — **set it for the installed sensor** | `robot_io_options` in `policy.yaml` |
| Command publish rate | `publish_hz: 500` | `robot_io_options` |
| Feedback/command timeout | `timeout_s: 0.25` | `robot_io_options` (codec default in `ros2_named.py`) |
| Stop damping / step bound | `damping` 2.0, `max_position_step` 0.2 rad per publish | `NamedJointCodec` defaults in `ros2_named.py`; overridable via `robot_io_options` |
| Policy-mode limit handling | `policy_limit_mode: clip` (also `pass`, `feedforward`) | `NamedJointCodec`; overridable via `robot_io_options` |
| Knee limits on hardware | training ∩ physical: upper 2.49582 rad (training 2.53073) | `physical_limits` in `robot_io/a3.py` |
| Fixed head pitch | `head_pitch_joint`: q=0, dq=0, τ=0, kp=3, kd=0.3 | `fixed_joints` in `robot_io/a3.py`; removed from the sim model by `prepare.py` |
| Shaping target | `shaping_target: motion_first_frame` | `policy.yaml` |
| Ankle gains | kp 64.6461, kd 4.708 in the frozen spec | `specs/agibot_a3.yaml`; commanded `joint_kp`/`joint_kd` come from the exported contract in `policy.yaml` |
| Sim timestep | 0.005 s, decimation 4 (base_sim/`evaluate.py` defaults) | `start_pico.sh sim`, `scripts/proteus/evaluate.py` |
| Policy config | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | both launchers |

Vendor mass differs from training by about 0.194 kg and is retained; five vendor joint ranges differ
(simulation uses training limits). Details in `third_party/agibot_a3/validation.json` after `prepare.py`.

## Usage

Offline evaluation (MuJoCo, measured root):

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot a3 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml \
  --dataset checkpoints/agibot_a3/motions --output outputs/a3/eval-new
```

PICO teleop in simulation (publisher on the same PC):

```bash
# Terminal 1: XRoboToolkit PC Service + calibrated PICO body/leg trackers
bash scripts/a3/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: MuJoCo simulator (open its mjviser URL; untick Elastic Band before judging balance)
bash scripts/a3/start_pico.sh sim
# Terminal 3: Tracker, reading motion (TCP 28701) and buttons (TCP 5592) from terminal 1
bash scripts/a3/start_pico.sh policy-sim 127.0.0.1
```

PICO buttons: **A** shapes to neutral; release, then **A+B** enables policy Hold; **X** pauses/resumes the
live reference (publisher side); **B** alone damps. Inspect IK in the publisher viewer before control.

Hardware (after releasing the native controller through the vendor procedure):

```bash
# Terminal 1
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
bash scripts/a3/start_proteus.sh checkpoints/agibot_a3/motions/motions/walk1_subject1.npz
# Or PICO teleop instead: terminal 1 runs `start_pico.sh publisher`, then
bash scripts/a3/start_pico.sh robot PUBLISHER_IP
```

Keyboard (`start_proteus.sh`): **i** shapes to frame 0 over 10 s, **]** enters policy Hold, **Space**
plays/pauses, **o** damps, **Ctrl+C** exits. Extra flags are forwarded to `tracking`, e.g. `--record`.

Shutdown: press **o** (or **B** on PICO), then Ctrl+C. On close the transport publishes damping several times.

## Safety notes & known issues

- Never run on the physical A3 without gantry support and an E-stop in reach: no hardware run has been done.
- Feedback or commands older than 250 ms, errors, nonfinite values or zero mode → zero stiffness,
  velocity and feed-forward torque with damping kept. These are process-level checks; there is no claim
  of firmware protection after a process kill or power loss.
- The IMU extrinsic defaults to identity; a wrong mount silently corrupts the gravity/orientation observation.
- PICO stream loss holds the last reference and does **not** damp while robot state stays healthy.
  Use one robot per default port set (28701/5592; hand triggers stay on 5593, no hand driver here).
- `policy.yaml` uses the reference anchor (no localization); `policy-sim.yaml` needs a measured world root and
  is for simulation only.
- File-tracking metrics (all three default clips pass at 0.5 m) do not establish live-control quality.

## Detailed docs

- English: [`docs/tutorials/proteus-a3.md`](../../docs/tutorials/proteus-a3.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-a3.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-a3.md)
- PICO setup: `docs/tutorials/pico-teleoperation.md`
