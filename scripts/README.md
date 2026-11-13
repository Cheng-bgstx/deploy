# Deployment scripts

[中文](README_zh.md)

This folder holds the launchers and tools that put a policy on a real robot. Every robot uses the same
runtime (`sim2real/`) and the same workflow; only the robot I/O, the deployment overlay and the safety
details differ. This page describes the shared workflow and how to bring up a new robot. Each robot
folder has a README (`./<robot>/README.md`) with its specific commands.

Terms follow [`GLOSSARY.md`](../GLOSSARY.md): **Tracker**, **Reference**, **Motion**, **File Playback**,
**Planner**, **Shaping**, **Hold**, **Go**, **Final Hold**, **E-stop**, **Controller**, **Bridge**.

> **Safety.** Treat every hardware run as dangerous. Keep the robot on a gantry or hoist until Hold is
> stable, keep a hardware emergency stop within reach of a second person, and never run a robot whose
> hardware status below is "untested" without the vendor's safety procedure and supervision. A software
> E-stop, a killed process or a lost network link is **not** a hardware emergency stop.

## Layout

### Robot folders

| Folder | Robot | `--robot` names | Contents |
| --- | --- | --- | --- |
| [`g1/`](./g1/README.md) | Unitree G1 | `g1` | Proteus on an external PC or the robot's Orin: SDK2 Bridge, validator, onboard deploy |
| [`et1/`](./et1/README.md) | Galbot ET1 | `galbot_et1`, `et1` | Onboard HDEP Bridge, keyboard operator, Galbot tracking-policy import |
| [`pm01/`](./pm01/README.md) | EngineAI PM01 | `pm01`, `pm_v2`, `engineai_pm01` | ROS 2 launchers (PC and onboard), read-only check, sway monitor, bounded trial relay tools |
| [`q1/`](./q1/README.md) | Primebot Q1 | `q1`, `primebot_q1` | Overlay builder, offline validator, joint commissioning, IMU inspection |
| [`a3/`](./a3/README.md) | AgiBot A3 | `a3`, `agibot_a3` | Asset preparation, File Playback and PICO launchers |
| [`t800/`](./t800/README.md) | EngineAI T800 | `t800`, `engineai_t800` | Asset preparation, File Playback and PICO launchers |
| [`x2/`](./x2/README.md) | AgiBot Lingxi X2 | `x2`, `lingxi_x2`, `agibot_x2` | Onboard AimDK Bridge (the route run on hardware), damped overlay, probes; PC/ROS 2 launchers |
| [`oli/`](./oli/README.md) | LimX OLI | `oli`, `limx_oli_edu`, `hu_d04_01` | Assets, motion conversion and retargeting, sim2sim rendering |

The names come from `sim2real/config/robots/__init__.py`. The Unitree H2 (`h2`, `unitree_h2`) has a
robot config and a PICO retargeting profile but no robot I/O backend and no folder here.

### Other folders

| Folder | Contents |
| --- | --- |
| [`dh116s/`](./dh116s/README.md) | DH116S dexterous hands driven by PICO trigger grip (`control.py`, `keyboard_control.py`) |
| `scalecar/` | G1 vehicle-entry task tooling: Vehicle Scene build, Scalecar motion conversion, Reference Token comparison. The G1 launchers are in `g1/scalecar/` |
| `proteus/` | Proteus helpers shared by robots: `prepare_assets.py` (stage and verify an MJCF), `retarget_lafan.py`, `build_retarget_profiles.py`, `evaluate.py` (sim2sim over a motion library) |
| `setup/` | G1 network helpers (hotspot, laptop gateway) and the XRoboToolkit pybind setup |

### Shared tools (`scripts/*.py`)

| Script | Purpose |
| --- | --- |
| `export_proteus_onnx.py` | Wrap a released Proteus actor for one robot: `policy.onnx`, `policy.yaml`, `policy-sim.yaml`, `comparison.json` |
| `build_proteus_contract.py` | Build a Proteus contract for a robot outside the release, from its MJCF and a released template robot |
| `prepare_proteus_test_motions.py` | Convert the Proteus ScaleBridge test clips (G1, ET1) into Motion datasets |
| `lafan_tracking_suite.py` | Build robot-specific LAFAN References and optionally run sim2sim |
| `make_hold_motion.py` | Write the neutral startup Motion used to shape before a live PICO stream |
| `summarize_proteus_sim2sim.py` | Summarize trajectories recorded while rendering a Proteus sim2sim video |
| `build_xrobot_profile.py` | Convert a calibrated SMPL-X GMR profile to PICO/XRobot body frames |
| `export_scalecar_tracker_onnx.py`, `export_scalebfm_onnx.py` | Exports for the Scalecar Tracker and ScaleBFM |
| `test_policy_inference.py` | Benchmark policy inference backends |
| `record_motion.py`, `view_motion.py`, `process_qpos_data.py` | Record, view and smooth Motions |
| `compute_tracking_metrics.py` | Progress, root and body tracking errors from `--trajectory-output` rollouts |

## Runtime architecture

```text
 Reference Source                Tracker (50 Hz)                 Robot I/O                     Robot
 ----------------                ---------------                 ---------                     -----
 File Playback  --motion-backend npz ---+
 Planner        --motion-backend planner+-> sim2real/rl_policy/ -> --robot-io inline ---------> vendor SDK / ROS 2
 PICO stream    --motion-backend zmq ---+   tracking.py            (backend in the Tracker)
   (sim2real/teleop/pico_retarget_pub.py)   policy ONNX            --robot-io zmq ---> Bridge -> vendor SDK
                                            stage machine          (ZMQ on 127.0.0.1)  BridgeSafety
                                                  ^
                                      Controller: keyboard / joystick / bridge_joystick / pico
```

- **Tracker.** `sim2real/rl_policy/tracking.py` (built on `base_policy.py`) runs one control step every
  `1 / --rl-rate` s (50 Hz): read state, update the Reference, build observations
  (`sim2real/rl_policy/observations/`), run the ONNX policy (`--inference-backend onnx-cpu`, `onnx-gpu`
  or `tensorrt`), post-process the joint targets (`sim2real/rl_policy/utils/joint_targets.py`), and send
  position targets with the YAML PD gains.
- **Inline robot I/O** (`--robot-io inline`). `sim2real/rl_policy/robot_io/factory.py` builds the robot's
  backend inside the Tracker process. Used by A3, Q1, T800, PM01 (ROS 2, `ros2_named.py` /
  `engineai.py`), X2 (`x2.py`, AimDK ROS 2) and OLI (`oli.py`, LimX SDK). These backends run their own
  publish loop and command watchdog and fall back to damping when feedback or commands go stale.
- **ZMQ robot I/O** (`--robot-io zmq`). The Tracker exchanges LowState/LowCmd over local ZMQ
  (`robot_io/zmq.py`) with a separate Bridge process. The G1 uses `scripts/g1/real_bridge.py` (SDK2),
  the ET1 uses `scripts/et1/hdep_bridge.py` (HDEP; the ET1 has no inline mode), and the X2 onboard route
  uses `scripts/x2/aimdk_bridge.py` (AimDK HAL). All three arbitrate commands with
  `robot_io/bridge_safety.py` (BridgeSafety): startup emits no motor command, takeover needs a ready
  Tracker plus a fresh physical A press, Select and faults latch a stop independently of the Tracker.
  Sim2sim with `sim2real/sim_env/base_sim.py` also uses `--robot-io zmq`.
- **Where it runs.** The Tracker runs on whichever machine reaches the robot's SDK: an external PC
  (Host) on the robot LAN, or the robot's own computer. The G1 launchers support both
  (`g1/start_proteus_pc.sh`, `g1/onboard.sh`; only the PC path is hardware-tested); the ET1 runs only onboard. Pin the Tracker and the Bridge
  to separate cores and keep ONNX to few threads (`SIM2REAL_ORT_NUM_THREADS`), as the launchers do.

## Deployment workflow

### 1. Adapt and export the policy

Proteus actors are wrapped, not retrained. `export_proteus_onnx.py` reads a release under
`<proteus-root>/ScaleTrack/checkpoints/<release>` (or `--checkpoint-dir`) and writes
`checkpoints/proteus/<release>/<contract>/`:

```bash
uv run python scripts/export_proteus_onnx.py --release wsq61 --contract engineai_pm01 --robot pm01
# A robot outside the release: build its contract first, then point the export at it.
uv run python scripts/build_proteus_contract.py --robot et1 --name galbot_et1 --template unitree_r1
uv run python scripts/export_proteus_onnx.py --release proteus_60 --contract galbot_et1 --robot et1 \
  --contract-dir checkpoints/proteus/contracts
```

| File | Use |
| --- | --- |
| `policy.yaml` | Hardware. `proteus.reference_forcing: true`: the root is placed at the Reference pelvis, because the robot has no root position. Regenerate instead of editing |
| `policy-sim.yaml` | Simulation only. Measured world root (`reference_forcing: false`). Do not use on hardware without localization |
| `policy-pc.yaml` | Hardware deployment overlay, written by a robot's prepare script (step 2). Not produced by the export |
| `comparison.json` | Wrapper vs released actor on random inputs; the export fails above `1e-6` |

Most robot folders wrap this in a prepare script (`python -m scripts.a3.prepare`, `scripts.t800.prepare`,
`scripts.pm01.prepare`, `scripts.q1.prepare`, `scripts/x2/export_policy.py`,
`scripts/oli/prepare_assets.py`) that also stages the MJCF and Motions. Non-Proteus policy families are
adapted with the `.agents/skills/adapt-policy-to-sim2real` skill.

### 2. Deployment overlay

Keep the exported `policy.yaml` unchanged and write hardware changes to a separate overlay
(`g1/prepare_proteus_deployment.py --robot g1|galbot_et1`, `python -m scripts.q1.prepare_deployment`,
`et1/prepare_galbot_track_policy.py`). Keys the runtime reads:

| Key | Behaviour in code |
| --- | --- |
| `joint_kp`, `joint_kd` | PD gains sent with every command in every stage (`utils/command_sender.py`). Change them in the overlay, e.g. the G1 waist roll/pitch at kp 100, kd 2.5 |
| `soft_joint_pos_limits` | Name (regex) → `[lower, upper]`. Clips Tracker targets; must cover every policy joint. A zero-width range `[d, d]` locks a joint at `d` (the ET1 head) |
| `max_target_velocity_rad_s` | Rate limit on Tracker targets, `max × control period` per step; restarts from the measured pose on every entry to Hold |
| `shaping_target` | `default_joint_pos` (default) or `motion_first_frame`: where Shaping interpolates to |
| `shaping_joint_pos_limits` | Clip only for pose targets before policy control (Shaping, `reference_entry_s`); never clips Tracker targets |
| `reference_entry_s` | File Playback only: the Hold request first interpolates to the Motion's first frame over this time, then enters Hold (G1 and ET1 overlays: 2 s) |
| `policy_entry_max_joint_error_rad`, `policy_entry_max_joint_velocity_rad_s` | Optional Hold gate: Hold is refused until Shaping is complete and every joint is within the error (and, with the velocity key, settled) |
| `shaping_gravity_compensation` | Optional feedforward during Shaping (Q1, experimental) |
| `robot_io_options` | Keyword arguments for the inline backend: topics, `publish_hz`, `timeout_s`, IMU extrinsic `imu_to_pelvis_wxyz`, and for `ros2_named.py` robots `policy_limit_mode` |

Shaping-target clipping uses `shaping_joint_pos_limits`, else `soft_joint_pos_limits`; with neither, the
Motion's first frame is used unclipped. `policy_limit_mode` (ROS 2 named-joint backends) decides what
happens to Tracker targets outside the joint range: `clip` (default) sends the bound, `pass` sends them
unchanged, `feedforward` sends the bound plus `kp × excess` as effort, faded out near the limit. Shaping
targets are always clipped by that backend. Locked joints come in three forms: no contract slot
(`build_proteus_contract.py --lock`), a zero-width soft limit, or a backend's fixed motor (the A3 head
pitch in `robot_io/a3.py`). The exporter already sets `shaping_target: motion_first_frame` and default
`robot_io_options` for OLI, X2, A3, T800, PM01 and Q1.

### 3. Sim2sim rehearsal

Run the same YAML, Motion and Controller flow in MuJoCo before any hardware run.

```bash
# Policy against a Motion library, measured root; fails on unfinished, non-finite or > 0.5 m body error
uv run --no-sync python -m scripts.proteus.evaluate --robot a3 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml \
  --dataset checkpoints/agibot_a3/motions --output outputs/a3/eval
# Single Motion in the integrated simulator (viewer; --headless for batch)
uv run sim2real/sim_env/integrated_sim2sim.py --robot g1 --policy-config <policy.yaml> --motion-path <motion>
# Deployment rehearsal of the real Tracker, stage machine and BridgeSafety (G1, ET1)
uv run python scripts/g1/validate_proteus_deployment.py --robot g1 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy-pc.yaml --npz-playback-limit-s 3
```

`validate_proteus_deployment.py` holds the root fixed during Shaping (an ideal gantry), lowers it, then
presses the Controller sequence: early Start and early second A must be ignored, then Shaping, Hold,
play, pause, resume and Select. It writes `<output>.json` and `<output>.npz`; `passed` requires finite
state, no Bridge fault, the expected stage sequence, and a free root that stays above 75 % of standing
height and upright. `--scenario walk-hold` also checks forward progress, lateral error and drift in the
Final Hold. Q1 has its own offline rehearsal, `scripts/q1/validate_deployment.py`. A pass is simulation
evidence only: it does not model gantry cables, the robot's communication timeout or the hardware stop.
For live PICO, each robot's `start_pico.sh sim` + `start_pico.sh policy-sim` runs the same stream against
`base_sim.py`.

### 4. Hardware preflight

- [ ] `checkpoints/` and `third_party/` are present (runtime artifacts, not in git; see
      [`docs/reference/artifacts.md`](../docs/reference/artifacts.md)). Inference uses the root project; teleop uses `venv/pico`.
- [ ] `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` are exported in the shell and persisted in the
      robot user's `~/.bashrc`. The launchers export both; add them to every manual command.
- [ ] Network: the robot LAN is reachable from the machine that runs the Tracker, with the right interface
      (`G1_INTERFACE`, `--robot-interface`, `--robot-ip`). See
      [`docs/getting-started/network-configuration.md`](../docs/getting-started/network-configuration.md).
- [ ] Vendor SDK or ROS 2 environment is sourced as the robot README says (e.g. ROS 2 Humble and AimDK
      for X2, `scripts/pm01/env.sh` for PM01).
- [ ] Exactly one motor writer: the vendor locomotion or motion-control stack is stopped or has released
      ownership. The G1 and ET1 launchers refuse to start next to a running controller.
- [ ] Read-only feedback probe passed: G1 and ET1 `PROTEUS_READ_ONLY=1`, PM01 `scripts/pm01/check_robot.py`,
      X2 `scripts/x2/probe.sh` (onboard) or `scripts/x2/check_feedback.py` (PC), Q1 `scripts/q1/inspect_imu.py`. Check joint order and signs and
      that the IMU reports gravity along the expected axis.
- [ ] Robot hangs on the gantry or hoist with the feet clear; the operator knows when to lower it.
- [ ] Hardware emergency stop tested and in a second person's hand.
- [ ] No benchmark, simulation or other heavy process runs on the Tracker's machine.

### 5. Operating stages

```text
 start --> E-stop (zero) --Shaping--> Shaping --Hold--> Hold --Go--> playing --end--> Final Hold
              ^                            (10 s)          (Reference frozen)        (last frame)
              +---------------- E-stop from any stage; restart with Shaping ---------------+
```

| Stage | Robot | Enter | Leave |
| --- | --- | --- | --- |
| E-stop (`zero`) | Tracker targets follow the measured joints. The ROS 2, X2 and OLI backends send damping only (kp = 0). BridgeSafety: Select keeps the last kp/kd on measured positions; faults latch damping | Startup; E-stop input; a Bridge stop or fault | Shaping |
| Shaping (`init`) | Interpolates from the pose at entry to `shaping_target` over 10 s, then holds it | Shaping input | Hold, E-stop, or Shaping again (restarts) |
| Hold (`policy`, paused) | Tracker in control; histories reset; Reference frozen at the Motion's first frame (with the Planner, until the first Chunk) | Hold input (joystick: only after Shaping completes) | Go, E-stop, Shaping |
| Go (playing) | Reference advances; File Playback toggles play/pause on each Go input; the Planner accepts the first Go only | Go input while in Hold | End of Motion, E-stop, Shaping |
| Final Hold | Reference stays on the last frame (or at `--npz-playback-limit-s`); Go cannot extend it. The Motion must end standing | End of File Playback | E-stop, Shaping (new session) |

Controller mapping (from `sim2real/rl_policy/control_mode.py`, `controllers/` and `tracking.py`):

| `--controller` | Shaping | Hold | Go | E-stop |
| --- | --- | --- | --- | --- |
| `keyboard` (Tracker terminal) | `i` | `]` | `space` | `o` |
| `keyboard` with `--keyboard-layout et1` (PM01 launchers) | `a` | `a` again | `s` | `space` or `x` |
| `joystick`, Unitree HG family (G1) | `A` | `A` again, after Shaping completes | `Start` | `Select` |
| `bridge_joystick` (handset via the Bridge: G1 onboard, ET1) | `A` | `A` again, after Shaping completes | `Start` | `Select` (ET1 Bridge also: `L1` + `Down`) |
| `pico` (right controller) | `A` alone | `A` + `B` together (release first) | `X` on the publisher (pause/resume stream) | `B` alone |
| ET1 / X2 onboard `scripts/<robot>/operator.sh` (keys to the Bridge) | `a` | `a` again | `s` | `space` or `x` |

Notes:

- `Select` is edge-triggered and wins over simultaneous presses. On a BridgeSafety robot it latches in the
  Bridge itself, so it acts even if the Tracker hangs. After any stop, restart with Shaping → Hold → Go.
- `keyboard` and `pico` do not wait for Shaping to finish before Hold unless the overlay sets
  `policy_entry_max_joint_error_rad`. Wait for the log line `Shaping interpolation complete`.
- With the PICO stream the Tracker never pauses the Reference: it follows what the publisher sends, which
  is a neutral pose until `X` starts the live stream. Stream loss holds the last Reference.
- The Unitree `go` DDS family (legacy layout) maps `A` / `R1` / `B` / `R2`; no robot in this repo uses it.
- With a G1 or ET1 on the gantry: Shaping, then Hold, wait for `Hold ready`, lower the gantry, then Go.

### 6. Record, review and shut down

- `--record --record-output <file>.npz` saves every control step when the Tracker exits. The launchers put
  it in a run directory with the logs: `outputs/proteus-pc/<time>/` (G1: `tracking.npz`, `tracker.log`,
  `bridge.log`, `bridge-state.json`, `lowstate-monitor.log`), `<root>/logs/run-<time>/` (ET1),
  `outputs/real_runs/<time>/` (Scalecar). Without `--record-output` the file is
  `./policy_tracking_record_<policy>_<time>.npz`.
- Load a recording:

  ```python
  import numpy as np
  rec = np.load("tracking.npz", allow_pickle=True)["data"].item()  # schema policy_tracking_record_v1
  rec["robot"]["joint_pos"], rec["policy"]["q_target"], rec["runtime"]["control_mode"], rec["runtime"]["motion_t"]
  ```

- Shut down with E-stop first while the robot is supported, then Ctrl+C or SIGTERM. The Tracker saves the
  recording and closes the Controller and robot I/O; the G1 and ET1 Bridges send damping on an orderly
  exit. On the G1 Orin use `scripts/g1/onboard.sh stop` and `scripts/g1/onboard.sh logs`. Stop the launcher
  before rebooting the robot.

## Bringing up a new robot

1. **Robot config.** Add `sim2real/config/robots/<robot>.py` with a `RobotCfg` (joint and body names,
   limits, armature, `safe_joint_kp`/`kd`, `mjcf_path`, `default_qpos`). Proteus robots can load a frozen
   spec with `from_proteus_spec()` from `sim2real/config/robots/specs/<contract>.yaml`. Register the name
   and aliases in `_ROBOT_CFGS` in `sim2real/config/robots/__init__.py`.
2. **MJCF and assets.** Stage the vendor model with `scripts/proteus/prepare_assets.py` into
   `third_party/<contract>/model` and fix every joint-axis mismatch it reports. `third_party/` and
   `checkpoints/` are runtime artifacts, not in git: upload converted artifacts to the shared Drive with
   `rclone` as `AGENTS.md` describes.
3. **Robot I/O.** Add `sim2real/rl_policy/robot_io/<robot>.py` and dispatch it in
   `robot_io/factory.py`. Reuse before writing new code:
   - `ros2_named.py` (`NamedRosRobotIO`) for ROS 2 joint groups addressed by name (A3, Q1);
   - `engineai.py` for EngineAI's `interface_protocol` with the `joint_bridge` mode handshake (T800, PM01);
   - `--robot-io zmq` plus a Bridge using `BridgeSafety` when the SDK cannot live in the Tracker process,
     or when you need a stop that survives a Tracker hang (G1, ET1).
   Every backend must map joints by name, refuse stale feedback, send damping on command timeout, and
   accept the `zero` / `init` / `policy` modes from `set_control_mode`.
4. **Proteus contract and export.** Use the robot's released contract, or build one with
   `build_proteus_contract.py --template <released robot>`; then export (step 1). Add the robot to the
   name sets in `policy_yaml()` of `export_proteus_onnx.py` if it needs `shaping_target` or default
   `robot_io_options`.
5. **Deployment overlay.** Write a prepare script that derives `policy-pc.yaml` (or document the keys) from
   the exported `policy.yaml`; never edit the export in place.
6. **PICO retarget profile.** Add `sim2real/teleop/configs/xrobot_to_<robot>.json` (built with
   `scripts/proteus/build_retarget_profiles.py` or `scripts/build_xrobot_profile.py`) and add the robot to
   the GMR target map and robot sets in `sim2real/teleop/pico_retarget_pub.py`. Create the startup Motion
   with `python -m scripts.make_hold_motion --robot <robot> --output <dir>`.
7. **Tests.** Follow `tests/test_proteus_transport.py` (feedback mapping, limits, watchdog, damping, factory
   options), `tests/test_pico_retarget_profiles.py` (profile validity, FK and ZMQ buffer) and
   `tests/test_g1_bridge_safety.py` (takeover, Select, timeouts) for a Bridge. Add the robot to their
   parameter lists, and whitelist new test files in `.gitignore` (`/tests/*` is ignored by default).
8. **Robot folder.** Create `scripts/<robot>/` with `start_proteus.sh` (and `start_pico.sh` if PICO is
   supported) exporting the HF variables, plus `README.md` and `README_zh.md` with the sections
   **Status, Files, Setup, Adjustments, Usage, Safety notes**. Add the robot to the tables on this page.
9. **Docs.** Keep English and Chinese in sync: this page and `README_zh.md`, the robot README pair, the
   top-level `README.md` / `README_zh.md`, and `docs/` with
   `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`.
10. **Verify.** `uv run python -m py_compile <files>` for runtime files,
    `uv run --project venv/pico python -m py_compile <files>` for teleop files, the tests, the sim2sim
    rehearsal, a read-only hardware probe, and only then a gantry-protected Shaping trial.

## Robot status

| Robot | Hardware status | Default policy (launcher) | README |
| --- | --- | --- | --- |
| Unitree G1 | Hardware-tested from an external PC (`start_proteus_pc.sh`). Onboard Orin: read-only timing runs only, not yet hardware-tested | `checkpoints/proteus/proteus_wamseq_62/unitree_g1` (`policy-pc.yaml`) | [g1](./g1/README.md) |
| Galbot ET1 | Hardware-tested onboard: runs on 2026-10-29 and 2026-11-11 (proteus_60; WAM-62 stood and started a walk) | `checkpoints/proteus/proteus_60/galbot_et1` (`policy-pc.yaml`); `proteus_wamseq_62/galbot_et1` via `PROTEUS_POLICY_DIR` | [et1](./et1/README.md) |
| EngineAI PM01 | First gantry-protected hardware walk on 2026-11-05; onboard gantry run on 2026-11-13 | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml` | [pm01](./pm01/README.md) |
| Primebot Q1 | Read-only feedback tested; first Shaping trial crossed the legs; the fix is verified offline only; policy takeover unvalidated | `checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml` | [q1](./q1/README.md) |
| AgiBot A3 | Simulation tested; hardware untested | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml` | [a3](./a3/README.md) |
| EngineAI T800 | Simulation tested; hardware untested | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml` | [t800](./t800/README.md) |
| AgiBot Lingxi X2 | Onboard gantry trials on 2026-11-12 (Shaping worked; arm/hip oscillation, damped overlay fixed in simulation only) | `checkpoints/proteus/proteus_wamseq_62/lingxi_x2` (`policy-pc-damped.yaml`, `start_proteus_onboard.sh`) | [x2](./x2/README.md) |
| LimX OLI | Simulation tested; hardware untested | `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml` | [oli](./oli/README.md) |
| Unitree H2 | Robot model and PICO retargeting only; no hardware I/O | — | [`docs/reference/h2.md`](../docs/reference/h2.md) |

Default policies are the ones the robot's `start_proteus.sh` (G1: `start_proteus_pc.sh`, ET1:
`start_proteus_onboard.sh`) uses; `PROTEUS_POLICY_CONFIG` or `PROTEUS_POLICY_DIR` overrides them where
the launcher supports it.
