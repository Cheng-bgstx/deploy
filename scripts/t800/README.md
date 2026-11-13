# EngineAI T800 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

Proteus WAMSeq on the standard 25-motor EngineAI T800 (`engineai_t800` contract; not either T800 Pro
contract). The released shared actor is exported to ONNX unchanged; no training or fine-tuning. The Tracker
(`sim2real/rl_policy/tracking.py`) runs CPU ONNX inference at 50 Hz and drives the robot through EngineAI's
`joint_bridge` ROS 2 interface (`interface_protocol`); a separate 500 Hz ROS timer republishes the latest
command with `parallel_parser_type=1` (RL_PARSER). **Software and MuJoCo are tested; the physical robot and a
live PICO headset are not.**

## Status

| Item | Value |
| --- | --- |
| Hardware | Untested. Software tests and MuJoCo evaluation only (`docs/experiments/proteus-t800.json`: `hardware_tested: false`, `physical_pico_tested: false`) |
| `--robot` | `t800` (alias `engineai_t800`) — `sim2real/config/robots/__init__.py` |
| Joints | 25 (SDK order `J00_HIP_PITCH_L` … `J24_HEAD_YAW`, mapped to policy order by name) |
| Robot I/O | `--robot-io inline` → `T800RobotIO` (`interface_protocol` JointState/JointCommand/ImuInfo/MotionState). Topics `/hardware/joint_state`, `/hardware/joint_command`, `/hardware/imu_info`, `/motion/motion_state` |
| Tracker runs on | Any host on the robot's ROS 2 graph with ROS Humble and the vendor workspace sourced; the launchers do not choose a host |
| Default policy (launchers) | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml` (hardware) and `policy-sim.yaml` (sim) |
| Validated release | `checkpoints/proteus/proteus_wamseq_62/engineai_t800/` — the 2026-11-03 evaluation table and experiment record refer to it, not to wsq61 |
| Default motion | `checkpoints/engineai_t800/motions/motions/walk1_subject1.npz`; PICO hold `checkpoints/engineai_t800/pico_hold` |

## Files

`scripts/t800/`:

| File | Purpose |
| --- | --- |
| `prepare.py` | One-shot asset build: stages and verifies `t800_mocap.xml` from `GMR-engineai`, exports ONNX + YAMLs, converts EngineAI's `smplx_to_t800.json` to `xrobot_to_t800.json` and copies `bvh_lafan1_to_t800.json`, builds the PICO hold, retargets four LAFAN1 clips. Flags `--sources-root`, `--proteus-root`, `--release` (default `wsq61`), `--skip-motions`. |
| `start_proteus.sh` | Hardware file playback: `tracking --robot t800 --robot-io inline --controller keyboard --motion-backend npz`. Optional first argument is the motion NPZ. |
| `start_pico.sh` | PICO teleop: `publisher`, `sim` (adds `--sim-dt 0.000625 --decimation 32`), `policy-sim [publisher-ip]`, `robot [publisher-ip]`. |
| `__init__.py` | Makes `python -m scripts.t800.prepare` importable. |

Related files elsewhere:

- `sim2real/config/robots/t800.py` + `sim2real/config/robots/specs/engineai_t800.yaml` (via `proteus_spec.py`).
- `sim2real/rl_policy/robot_io/t800.py` — SDK joint names; `T800Codec` inherits RL_PARSER.
- `sim2real/rl_policy/robot_io/engineai.py` — shared EngineAI transport (also PM01): `joint_bridge` mode ownership, 1 s mode heartbeat, publishes only with a command subscriber present.
- `sim2real/rl_policy/robot_io/ros2_named.py` — IMU extrinsic, 250 ms watchdog, limit clip, 0.2 rad step bound, damping on stop.
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_t800.json` — live PICO IK.
- `scripts/proteus/` (`prepare_assets.py`, `retarget_lafan.py`, `evaluate.py`), `scripts/build_xrobot_profile.py`, `scripts/export_proteus_onnx.py`, `scripts/make_hold_motion.py`.
- Tests: `tests/test_proteus_transport.py`, `tests/test_proteus_sim_resolution.py`, `tests/test_pm01.py` (`test_pm01_uses_classic_parser_while_t800_keeps_rl_parser`), `tests/test_pico_retarget_profiles.py`.
- Record: `docs/experiments/proteus-t800.json`.

## Setup

1. Environments: root project (`uv`) for inference/sim, `venv/pico` for the publisher and offline IK.
   Launchers use `--no-sync`; install both first.
2. Sources next to the repo: `Proteus-wamseq`, `GMR-engineai` (engineai-robotics/GMR), `GMR` (YanjieZe),
   `ubisoft-laforge-animation-dataset`. Clone commands are in the tutorial.
3. Build the runtime artifacts (not in git). The default release matches the launchers:
   ```bash
   uv run --no-sync python -m scripts.t800.prepare --sources-root "$sources_root"
   ```
   The tutorial's recipe passes `--release proteus_wamseq_62`; if you build that release, point the launchers
   at it with `PROTEUS_POLICY_CONFIG`. Outputs go to `third_party/engineai_t800/`,
   `checkpoints/proteus/<release>/engineai_t800/`, `checkpoints/engineai_t800/{motions,pico_hold}`.
4. Hardware only: firmware that implements the documented `joint_bridge` command subscription
   ([engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace)). The inspected
   native SDK revision `335c60e` has telemetry but no joint-command subscriber, so it cannot run this adapter.
   Source `/opt/ros/humble/setup.bash` and the vendor workspace that provides `interface_protocol`.
5. Launchers export `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1` and `SIM2REAL_ORT_NUM_THREADS=1` (if unset).

## Adjustments

| Tunable | Current value | Where |
| --- | --- | --- |
| IMU topic | `/hardware/imu_info` (`ImuInfo`) | `robot_io_options.imu_topic` in `policy.yaml` (written by `scripts/export_proteus_onnx.py`); default in `robot_io/engineai.py` |
| IMU → pelvis extrinsic | `imu_to_pelvis_wxyz: [1, 0, 0, 0]` | `robot_io_options` |
| Command publish rate | `publish_hz: 500` | `robot_io_options` |
| Feedback/command timeout | `timeout_s: 0.25`; mode heartbeat 1 s | `robot_io_options`; heartbeat in `EngineAICodec._fresh` |
| Stop damping / step bound | `damping` 2.0, `max_position_step` 0.2 rad per publish | `NamedJointCodec` defaults; overridable via `robot_io_options` |
| Ankle parser | `parallel_parser_type = 1` (RL_PARSER: SDK converts serial pitch/roll ankle PD) | `EngineAICodec` in `robot_io/engineai.py` |
| Ankle gains | kp 107.2198, kd 8.2957, effort 160 Nm in the frozen spec | `specs/engineai_t800.yaml`; the commanded `joint_kp`/`joint_kd` in `policy.yaml` come from the exported release contract |
| Joint limits | Training limits only; the vendor model matches training, no physical-limit override | `robot_io/t800.py` (none passed) |
| Sim timestep | **0.000625 s**, decimation 32 (50 Hz policy) | `start_pico.sh sim`; `scripts/proteus/evaluate.py` default for T800 (override `--sim-dt`) |
| Policy config | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | both launchers |

The 0.625 ms step matters: T800 in the old 5 ms external-PD loop tracked materially worse (12.72 vs 5.62 cm
matched-sample body error); `tests/test_proteus_sim_resolution.py` guards it.

## Usage

Offline evaluation (uses 0.000625 s automatically):

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot t800 \
  --policy-config checkpoints/proteus/wsq61/engineai_t800/policy-sim.yaml \
  --dataset checkpoints/engineai_t800/motions --output outputs/t800/eval-new
```

PICO teleop in simulation:

```bash
# Terminal 1: XRoboToolkit PC Service + calibrated PICO body/leg trackers
bash scripts/t800/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: MuJoCo simulator (open its mjviser URL; untick Elastic Band before judging balance)
bash scripts/t800/start_pico.sh sim
# Terminal 3: Tracker, reading motion (TCP 28701) and buttons (TCP 5592) from terminal 1
bash scripts/t800/start_pico.sh policy-sim 127.0.0.1
```

PICO buttons: **A** shapes to neutral; release, then **A+B** enables policy Hold; **X** pauses/resumes the
live reference; **B** alone damps.

Hardware:

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
# Enter joint_bridge with the vendor controller and check /motion/motion_state first.
bash scripts/t800/start_proteus.sh checkpoints/engineai_t800/motions/motions/walk1_subject1.npz
# Or PICO teleop instead (publisher running elsewhere):
bash scripts/t800/start_pico.sh robot PUBLISHER_IP
```

Keyboard: **i** shapes to frame 0 over 10 s, **]** enters policy Hold, **Space** plays/pauses, **o** damps,
**Ctrl+C** exits. Extra flags go to `tracking` (e.g. `--record`).

Shutdown: **o** (or **B**), then Ctrl+C; the transport repeats damping on close. The launchers never switch
robot modes — leave `joint_bridge` with the vendor controller.

## Safety notes & known issues

- No hardware run exists. Use a gantry and keep the E-stop in reach for any first attempt.
- The transport publishes only while `/motion/motion_state` reports `joint_bridge` (fresh within 1 s) and a
  command subscriber exists; otherwise it sends nothing. An explicit mode change hands ownership back to firmware.
- Stale commands/feedback (>250 ms), errors or zero mode → zero stiffness, velocity and feed-forward with damping
  2. Process-level only: no firmware protection is claimed after a kill or power loss.
- The wsq61 default is not covered by the published evaluation table (that used `proteus_wamseq_62`).
- PM01 shares this transport but overrides the parser to classic (`0`) after hardware ankle oscillation; T800
  keeps RL_PARSER untested on hardware. Review `scripts/pm01/README.md` before a first T800 run.
- PICO stream loss holds the last reference and does not damp while robot state stays healthy.

## Detailed docs

- English: [`docs/tutorials/proteus-t800.md`](../../docs/tutorials/proteus-t800.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-t800.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-t800.md)
- PICO setup: `docs/tutorials/pico-teleoperation.md`
