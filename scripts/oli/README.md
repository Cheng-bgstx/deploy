# LimX OLI (HU_D04_01) launchers

`scripts/oli/` runs the Proteus WAMSeq Tracker on the LimX OLI EDU humanoid
(HU_D04_01, Proteus contract `limx_oli_edu`) through the official LimX Python SDK.
It also stages the robot model, converts and retargets motions, and records sim2sim
videos. The generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot
checklist) is in [`../README.md`](../README.md); this page covers only what is
specific to OLI. [中文](README_zh.md)

## Status

| Item | Value |
| --- | --- |
| Hardware | **Untested.** Local export, observation history, SDK transport mapping, stationary / synthetic arm-wave sim2sim, one LAFAN1 dance excerpt and the LAFAN walk / aiming / dance2 suite were checked in simulation only |
| `--robot` aliases | `oli`, `limx_oli_edu`, `hu_d04_01` (`sim2real/config/robots/__init__.py`) |
| Joints | 31 virtual pitch/roll (PR) joints, serial model `HU_D04_01_vis.xml` |
| Robot I/O | `--robot-io inline --robot-ip <IP>` → `OliRobotIO` (LimX `limxsdk` wheel); requires `ROBOT_TYPE=HU_D04_01` |
| Where the Tracker runs | Any computer with the LimX SDK wheel for its architecture that reaches the robot IP (`127.0.0.1` for the SDK simulator). The tutorial names the x86_64 and onboard ARM wheels; neither was run against hardware |
| Default checkpoint | `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/` (the actor used by the 2026-11-03 runtime audit) |
| Policy YAMLs | `policy.yaml`: hardware, `reference_forcing: true` (SDK gives no world root position). `policy-sim.yaml`: measured-root simulation, `reference_forcing: false` |
| Alternative actor | wsq61 is **not** the OLI default and is not validated by the OLI results; export it and pass its `policy.yaml` explicitly |

## Files

| File | Purpose |
| --- | --- |
| `start_proteus.sh` | Hardware / SDK-simulator run of one any4hdmi dataset with the keyboard controller |
| `start_pico.sh` | PICO teleop: `publisher`, `sim`, `policy-sim` or `robot` mode |
| `prepare_assets.py` | Copies the official serial HU_D04_01 model to `third_party/limx_oli_edu/model/` and checks joints, limits, axes and mass against the Proteus spec |
| `convert_motion.py` | Proteus packed OLI library clip → any4hdmi dataset (FK-checked), or `--standing-smoke` stationary clip |
| `make_wave_motion.py` | Synthetic arm raise / wave / lower reference for a dynamic smoke test |
| `retarget_lafan.py` | LAFAN1 BVH → OLI with LimX's official GMR, headless, with sole grounding |
| `render_sim2sim.py` | Integrated sim2sim video beside the reference (MuJoCo EGL + `ffmpeg`); also used by `scripts/x2/render_sim2sim.py` |

Related files elsewhere:

- Robot config: `sim2real/config/robots/oli.py` (`OLI_CFG`, `OLI_SDK_JOINT_NAMES`, `OLI_MJCF_PATH`).
- Robot I/O: `sim2real/rl_policy/robot_io/oli.py`; dispatch in `sim2real/rl_policy/robot_io/factory.py`.
- Simulation model tweaks (floor-only contacts, zero passive damping): `sim2real/sim_env/utils/mjcf.py`.
- Policy export: `scripts/export_proteus_onnx.py` (writes `policy.yaml` and `policy-sim.yaml`).
- Teleop: `sim2real/teleop/pico_retarget_pub.py` (GMR, robot `limx_oli_edu`), profile
  `sim2real/teleop/configs/xrobot_to_oli.json`, regenerated with `scripts/build_xrobot_profile.py`.
- Shared helpers: `scripts/make_hold_motion.py`, `scripts/lafan_tracking_suite.py` (+ `.json`),
  `scripts/summarize_proteus_sim2sim.py`.
- Tests: `tests/test_oli.py`, `tests/test_pico_retarget_profiles.py`.
- Records: `docs/experiments/proteus-oli-runtime-audit.json`.

## Setup

Run everything from the repository root. The Tracker, simulator and asset tools use the
root `uv` project; the PICO publisher uses `venv/pico`. The tutorial commands use
`uv run --no-sync`.

1. External checkouts (siblings of the repo): Proteus `wamseq` worktree
   (`../Proteus-wamseq`), `limxdynamics/humanoid-description`,
   `limxdynamics/limxsdk-lowlevel`, and for LAFAN/teleop profiles `limxdynamics/GMR`
   (`../GMR-limx`).
2. Stage the model and export the actor (both are Git-ignored runtime artifacts):

   ```bash
   uv run --no-sync python scripts/oli/prepare_assets.py \
     --description-root ../humanoid-description --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/export_proteus_onnx.py \
     --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 \
     --contract limx_oli_edu --robot oli
   ```

   Use the serial `HU_D04_01_vis.xml`, not the parallel-linkage `HU_D04_01.xml`.
   For wsq61, export with `--release wsq61`; the output goes to
   `checkpoints/proteus/wsq61/limx_oli_edu/`.
3. Install the SDK wheel into the root environment (x86_64 shown; the onboard ARM
   computer uses `python3/aarch64`):
   `uv pip install --no-deps ../limxsdk-lowlevel/python3/amd64/limxsdk-*.whl`
4. Copy `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/` (ONNX, YAMLs, contract JSON),
   `third_party/limx_oli_edu/` and the motion datasets to the deployment computer.
5. Both launchers export `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`; set them in
   your shell too for any manual command.
6. PICO only: XRoboToolkit PC Service and Python SDK per
   `docs/getting-started/teleop-x86-64.md`, then
   `uv sync --project venv/pico --no-install-package xrobotoolkit-sdk` and
   `bash scripts/setup/setup_xrobot_pybind.sh`. Create the neutral startup reference on
   the policy computer:
   `uv run --no-sync python scripts/make_hold_motion.py --robot oli --output outputs/oli/pico_hold`

## Adjustments

| Tunable | Current value | Set in |
| --- | --- | --- |
| Joint limits, kp/kd, armature, velocity / effort limits, default pose | Proteus `limx_oli_edu` spec (e.g. knee `-0.087..2.583` rad, root height `0.9282` m) | `sim2real/config/robots/oli.py` `_JOINTS`, `default_qpos` |
| SDK joint order | MJCF order except head: SDK sends `head_pitch` before `head_yaw` | `OLI_SDK_JOINT_NAMES` in `oli.py` |
| Command rate / watchdog | 1000 Hz publisher thread, 250 ms feedback / command timeout → damping | `OliRobotIO.__init__` defaults (`publish_hz`, `timeout_s`); the factory passes only `robot_ip`, so `robot_io_options` is not used for OLI |
| PR → AB conversion | `parallel_solve_required=True` on every joint (ankles and waist need it) | `OliRobotIO._publish` |
| IMU | SDK WXYZ quaternion and gyro used as the pelvis frame, no extrinsic rotation | `OliRobotIO._on_imu` |
| Shaping target | `shaping_target: motion_first_frame`; no soft / shaping limits are written for OLI, so frame 0 is used unclipped | `scripts/export_proteus_onnx.py` `policy_yaml` |
| Root translation | `reference_forcing: true` (hardware) / `false` (`policy-sim.yaml`) | `scripts/export_proteus_onnx.py` |
| Simulation | Floor contacts only, zero passive joint damping; `sim_dt` 0.005 s, policy 50 Hz (shared defaults; the audit found no reason to change them) | `sim2real/sim_env/utils/mjcf.py`, `IntegratedSim2SimArgs` |
| Inference | CPU ONNX Runtime, `SIM2REAL_ORT_NUM_THREADS` default 1 | both launchers |
| Robot / publisher IPs | `127.0.0.1` by default | launcher positional arguments |

## Usage

### Sim2sim rehearsal

```bash
# Stationary smoke clip, or convert a packed library clip with --library DIR --motion-index N
uv run --no-sync python scripts/oli/convert_motion.py --standing-smoke --duration 4 --output outputs/oli/wamseq_smoke
uv run --no-sync python scripts/oli/make_wave_motion.py --output outputs/oli/wave_motion
uv run --no-sync python scripts/oli/render_sim2sim.py --motion-path outputs/oli/wave_motion \
  --duration 10 --output outputs/oli/oli_wamseq_wave_sim2sim.mp4
# Real LAFAN1 motion via LimX GMR
uv run --no-sync python scripts/oli/retarget_lafan.py --gmr-root ../GMR-limx \
  --bvh ../ubisoft-laforge-animation-dataset/lafan1/data/dance1_subject1.bvh \
  --duration 8 --output outputs/oli/lafan_dance1
uv run --no-sync python scripts/lafan_tracking_suite.py --robot oli \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
```

`render_sim2sim.py` defaults to `proteus_wamseq_62/limx_oli_edu/policy-sim.yaml`; override
with `--policy-config`. Output directories must be new: the converters refuse to overwrite.

### Hardware (or LimX SDK simulator) run

Release the official stand / walk / mimic controllers first, and suspend the robot.

```bash
# start_proteus.sh <motion> [robot-ip=127.0.0.1] [policy.yaml]
bash scripts/oli/start_proteus.sh /path/to/oli/any4hdmi_dataset ROBOT_IP
```

Keyboard: `i` Shaping to frame 0 over 10 s, `]` policy Hold, Space play / pause,
`o` damping, Ctrl+C exit. The Unitree joystick controller is not used for OLI.

### PICO teleop

```bash
bash scripts/oli/start_pico.sh publisher --actual-human-height 1.70   # publisher PC (venv/pico)
bash scripts/oli/start_pico.sh sim                                     # root project simulator
bash scripts/oli/start_pico.sh policy-sim 127.0.0.1                    # Tracker, ZMQ robot I/O
# Hardware instead of the last two: on the policy / SDK computer
bash scripts/oli/start_pico.sh robot PUBLISHER_IP ROBOT_IP
```

Buttons: **A** Shaping to neutral (10 s), **A+B** policy Hold, **X** start / pause live
reference (pause returns the publisher to neutral at the current XY / yaw), **B** alone
damping. Open TCP 28701 (motion) and 5592 (buttons) between the computers. Extra Tracker
flags (e.g. `--policy-config`, `--motion-path`) go after the positional IPs. Disable the
simulator's elastic band in its GUI before judging free balance.

### Shutdown

`o` (keyboard) or **B** (PICO) selects damping; Ctrl+C exits. On orderly close
`OliRobotIO` publishes one damping command. Keep the SDK's independent stop path ready.

## Safety notes & known issues

- **Hardware is untested.** No OLI or PICO headset was connected during validation.
- Two command publishers must not own the robot: release the official controllers first.
- The 1 kHz publisher is a Python thread, not a hard real-time controller; validate
  timing and SDK / firmware compatibility on the robot.
- Damping on timeout / `zero` mode holds measured position with `kp=0` and keeps `kd`;
  it does not protect against process kill or power loss.
- If the PICO body stream stops, the receiver holds the last reference; it does not
  fall back to damping automatically.
- `policy-sim.yaml` needs a measured world root; do not use it on hardware.
- Simulated LAFAN walk / aiming / dance2 tracking has about 12–14 cm mean world-frame
  error over the 14 tracking bodies, with visible pose / yaw drift; this is not a claim
  of hardware readiness.

## Detailed docs

- English: [`docs/tutorials/proteus-oli.md`](../../docs/tutorials/proteus-oli.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-oli.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-oli.md)
- Generic workflow: [`../README.md`](../README.md)
