# AgiBot Lingxi X2 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

The X2 runs the Proteus Tracker on its own motion-control board (RK3588, aarch64, CPython 3.10), operated
from a PC over SSH, in the same layout as the Galbot ET1: `aimdk_bridge.py` owns the AimDK ROS 2 HAL and
serves the local ZMQ LowState/LowCmd contract, arbitrated by `BridgeSafety`; the shared Tracker
(`sim2real/rl_policy/tracking.py`) runs CPU ONNX inference at 50 Hz next to it; the ET1 keyboard operator
in a second SSH session sends `a`, `a`, `s` and space. This onboard route is the one run on hardware and the
only hardware route documented here.

## Status

| Item | Value |
| --- | --- |
| Hardware | **Gantry trials on 2026-11-12** on an X2 Basic (firmware v0.8.2.16) with `proteus_wamseq_62` and `policy-pc-damped.yaml`: takeover, shaping and Hold (stood 23 s and 42 s), `21_wave_right` without arm or waist oscillation, `15_walk_short` played in full and stopped standing. The rope carried most of the weight in every run: **not yet a validated free-standing run.** Details: [onboard tutorial](../../docs/tutorials/proteus-x2-onboard.md) |
| `--robot` aliases | `x2`, `lingxi_x2`, `agibot_x2` (`sim2real/config/robots/__init__.py`) |
| Joints | 31 (12 leg, 3 waist, 14 arm, 2 head); head locked by the overlay |
| Robot I/O | `--robot-io zmq` on 127.0.0.1 ⇄ `aimdk_bridge.py` (500 Hz) ⇄ `/aima/hal/joint/{leg,waist,arm,head}/{state,command}`, `/aima/hal/imu/torso/state` (the pelvis IMU) |
| Controller | `--controller bridge_joystick` (operator keys republished by the bridge in the Unitree key layout) |
| Tracker runs on | The X2 board: bridge on core 5, Tracking on core 7 (`isolcpus=5-7`) |
| Default policy | `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc-damped.yaml` (0 falls in 45 perturbed sims) |
| Other policies | `proteus_60`, `proteus_60_bigdata` (same layout); `wsq61` via `export_policy.py --release wsq61` or explicit paths |
| Default motion | `outputs/x2/wave/21_wave_right` (standing right-arm wave) |

## Files

Onboard route (`scripts/x2/`):

| File | Purpose |
| --- | --- |
| `deploy_onboard.sh` | PC → board: rsyncs code, the X2 actors with their overlays, the model, motions, HF assets and the aarch64 site-packages. Never stops mc or starts the controller. |
| `env.sh` | Sourced by the board scripts: board ROS 2 env, `aimdk_msgs`, `PYTHONPATH`, HF offline, one ORT/BLAS thread. |
| `start_proteus_onboard.sh` | Board launcher: refuses a second supervisor or controller, starts bridge + Tracker pinned to CPUs, logs under `logs/run-<UTC>/`. |
| `aimdk_bridge.py` | AimDK HAL ⇄ ZMQ bridge, `BridgeSafety` (`max_kd=25`); the ready operator A runs the takeover. |
| `release_mc.sh` | Takeover command: `aima em stop-app mc`, then waits until `mc_main` has exited. |
| `operator.sh` | Runs `scripts/et1/operator_keyboard.py` on the board (`ssh -t`). |
| `probe.sh` / `state_probe.py` | Read-only joint/IMU feedback rates and values plus ONNX timing on core 7. |
| `watch.sh` / `state_now.py` | Read-only live line (pelvis tilt, leg error, knee torque) from the bridge's LowState, on cores 0-3. |
| `imu_compare.py` | Read-only: compares the torso and chest IMU topics with the waist joints. |
| `make_damped_config.py` | Writes `policy-pc-damped.yaml` (arm kd ×5, leg/waist kd ×2, kp unchanged). |
| `make_wave_motion.py` / `make_short_walk.py` | Synthetic arm wave; clip 11 cut to a 1.4 m walk that ends standing. |
| `eval_sim2sim.py` | Headless sim2sim under perturbations and command delay (`--cmd-delay-ms`, `--sim-dt`). |
| `bridge_sim_test.py` | Rehearsal: the real bridge and Tracking processes against MuJoCo with a virtual gantry. |
| `render_delay_compare.py` | Side-by-side sim2sim videos of several overlays under command delay. |

Model, simulation and teleop files: `prepare_assets.py` (stages `third_party/lingxi_x2/model/scene_tracking.xml`),
`export_policy.py` (`--release`, default `proteus_wamseq_62`; also writes `policy-sim.yaml`), `retarget_lafan.py`,
`convert_reference.py`, `render_sim2sim.py`, `summarize_sim2sim.py`, `build_pico_profile.py`, `start_pico.sh`
(PICO teleop, see below).

Related files elsewhere:

- `sim2real/config/robots/x2.py` — joints, limits, gains, `X2_AIMDK_GROUPS`, `X2_VENDOR_LIMITS`, `x2_deploy_joint_limits`.
- `sim2real/rl_policy/robot_io/x2.py` — `X2StateCommand` codec (`head_present`, shared with the bridge), `X2RobotIO` (inline PC path, used only by `start_pico.sh robot`; never run on hardware).
- `sim2real/rl_policy/robot_io/bridge_safety.py` — command arbitration shared with the G1/ET1 bridges.
- `scripts/export_proteus_onnx.py`, `scripts/g1/prepare_proteus_deployment.py` (`--robot x2`), `scripts/make_hold_motion.py`.
- Tests: `tests/test_x2.py`, `tests/test_x2_bridge.py`, `tests/test_pico_retarget_profiles.py`.

## Setup

One-time, on the PC (root `uv` project):

1. Network: the robot's `10.0.1.0/24` is **tagged VLAN 200** (untagged IP gets no reply). Give the PC
   `10.0.1.2/24` on a VLAN interface (the `nmcli` commands are in the tutorial), then
   `ssh-copy-id agi@10.0.1.40` (motion-control board soc0, user `agi`).
2. Model and actor (`../agibot_x2_urdf`, a Proteus `wamseq` checkout):

   ```bash
   uv run --no-sync python scripts/x2/prepare_assets.py --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/export_proteus_onnx.py --contract lingxi_x2 --robot x2 \
     --checkpoint-dir ../Proteus-wamseq/ScaleTrack/checkpoints/proteus_wamseq_62 \
     --output-root checkpoints/proteus/proteus_wamseq_62
   uv run --no-sync python scripts/g1/prepare_proteus_deployment.py \
     --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.yaml --robot x2   # -> policy-pc.yaml
   uv run --no-sync python scripts/x2/make_damped_config.py checkpoints/proteus/proteus_wamseq_62/lingxi_x2
   ```

3. References: `scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold`,
   `scripts/x2/make_wave_motion.py --output outputs/x2/wave/21_wave_right`, and optionally the LAFAN windows
   (`retarget_lafan.py`, see the tutorial) and `make_short_walk.py --output outputs/x2/lafan/15_walk_short`.
4. aarch64 / CPython 3.10 site-packages from `uv.lock`, built once with the commands in the header of
   `deploy_onboard.sh` into `outputs/x2-onboard-env/site` (`X2_ENV_CACHE`; the ET1 set from the same `uv.lock`
   can be reused). It must hold `.lock-<first 16 hex of sha256(uv.lock)>`.
5. HF caches `models--elijahgalahad--h2_model` and `models--elijahgalahad--g1_xmls` in `~/.cache/huggingface/hub`.

`deploy_onboard.sh` refuses to run unless every release in `X2_RELEASES` (default
`proteus_wamseq_62 proteus_60_bigdata proteus_60`) has `policy.onnx`, `policy-pc.yaml` and
`policy-pc-damped.yaml`, and the model, `pico_hold` and `21_wave_right` exist. Set
`X2_RELEASES=proteus_wamseq_62` to ship only the default actor.

On the board: layout `proteus_x2/{repo,site-packages,hf,logs}` in `agi`'s home. `env.sh` sources
`/agibot/software/entry/cfg/basic_env.sh` and `aimdk_msgs` from `/agibot/software/hal_ethercat`
(`X2_AIMDK_PKG`), and exports `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=<root>/hf`.

## Adjustments

| What | Current value | Set in |
| --- | --- | --- |
| Overlay | `policy-pc.yaml`: shapes to the standing pose, 2 s reference entry, targets inside training ∩ vendor range minus 5%, 15 rad/s target slew, head locked | `scripts/g1/prepare_proteus_deployment.py` |
| Robot gains | `policy-pc-damped.yaml`: arm kd ×5 (1.1 → 5.5), leg/waist kd ×2 (knee 7.7 → 15.3), kp unchanged | `make_damped_config.py` (`--kd-scale`, `--leg-kd-scale`) |
| Bridge kd ceiling | `max_kd=25.0` (`BridgeSafety` default 10) | `aimdk_bridge.py` |
| Bridge rate / clamp | 500 Hz; targets clipped to training ∩ vendor range | `aimdk_bridge.py`, `x2_deploy_joint_limits` |
| CPU pinning | `PROTEUS_BRIDGE_CPU=5`, `PROTEUS_TRACKER_CPU=7`; core 4 runs the EtherCAT HAL, core 6 AgiBot DDS threads | `start_proteus_onboard.sh` |
| I/O timeout | `PROTEUS_IO_TIMEOUT_S=0.2` → `SIM2REAL_IO_TIMEOUT_S` and bridge `--timeout-s` | `start_proteus_onboard.sh` |
| Operator key freshness | 0.3 s (`OPERATOR_FRESH_S`) | `aimdk_bridge.py` |
| Takeover | `--takeover-cmd` (`bash release_mc.sh`), `--takeover-timeout-s 70` | `aimdk_bridge.py` |
| Head group | `X2_NO_HEAD=0` (default): the bridge subscribes to and commands the head group. This X2 Basic's HAL publishes a head group that reads constant 0, so the default works. `X2_NO_HEAD=1` → `--no-head` (no head topics subscribed or commanded) only if `probe.sh` prints `head : no messages` | `start_proteus_onboard.sh` |
| Board host / target | `X2_HOST=agi@10.0.1.40`, `X2_TARGET=proteus_x2` | `deploy_onboard.sh` |

Board launcher selection:

- `PROTEUS_POLICY_DIR` (default `checkpoints/proteus/proteus_wamseq_62/lingxi_x2`) and `PROTEUS_POLICY_FILE`
  (default `policy-pc-damped.yaml`; `policy-pc.yaml` has the training gains). The YAML must set
  `proteus.reference_forcing: true`.
- `PROTEUS_MOTION` — `outputs/x2/wave/21_wave_right` (default), `outputs/x2/pico_hold` (standing only),
  `outputs/x2/lafan/15_walk_short` (short forward walk), `outputs/x2/lafan/1x_*` (longer LAFAN windows).
- `PROTEUS_PLAYBACK_LIMIT_S` — stop the reference clock there and hold.
- `PROTEUS_READ_ONLY=1` (+ optional `PROTEUS_MAX_RUNTIME_S`) — dry run: no command publisher, no takeover.
- `PROTEUS_RUN_DIR`, `PROTEUS_ROOT` — log directory / install root.

## Usage

1. Rehearse on the PC with the real bridge and Tracking processes. `--lower-before-hold` follows the
   hardware key order; `--gantry-gravity 0.22,0.10,-0.97` reproduces the robot hanging tilted:

   ```bash
   uv run --no-sync python scripts/x2/bridge_sim_test.py --lower-before-hold \
     --motion outputs/x2/wave/21_wave_right --play-s 12
   uv run --no-sync python scripts/x2/eval_sim2sim.py --sim-dt 0.001 --cmd-delay-ms 40 \
     --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc-damped.yaml \
     --motion-path outputs/x2/lafan/15_walk_short --perturbation push60 --out outputs/x2/eval/walk_push.json
   ```

2. Deploy (touches only `X2_TARGET` on the board): `bash scripts/x2/deploy_onboard.sh`
3. Read-only probe, with mc still running. **Compare the joint readings with the robot**; bring the limbs
   close to standing by hand before a takeover:

   ```bash
   ssh agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 proteus_x2/repo/scripts/x2/probe.sh
   ```

4. Dry run with mc running (the bridge creates no command publisher):

   ```bash
   ssh -t agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PROTEUS_READ_ONLY=1 \
     proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

5. Gantry run. Terminal 1 (launcher), terminal 2 (operator), terminal 3 (optional monitor):

   ```bash
   ssh -t agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/operator.sh
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/watch.sh
   ```

   | Step | Key | Effect |
   | --- | --- | --- |
   | Wait | — | `bridge.log`: `Tracker and LowState ready` |
   | 1 | `a` | Takeover (`release_mc.sh`) and shaping to the standing pose |
   | — | — | Lower the gantry until both feet are flat and the robot stands upright, rope still taut (`watch.sh`: `UPRIGHT`) |
   | 2 | `a` | 2 s transition to the reference; the policy takes over (Hold) |
   | — | — | Slacken the rope until both knees read about +10 to +20 Nm |
   | 3 | `s` | Play |
   | Stop | space or `x` | Latched stop (damping) |

   If `mc_main` was running (e.g. after a board reboot), the first `a` releases mc, the bridge latches damping
   with "takeover took … s; mc is released, restart the launcher and press a again". Ctrl-C, relaunch and press
   `a`: with mc stopped the takeover is immediate. The launcher warns when `mc_main` is running.

6. Shutdown: keep the robot on the gantry, Ctrl-C the launcher; the bridge latches and sends damping for 0.2 s.
   Restart the native controller with `aima em start-app mc`. `proteus_x2/logs/run-<UTC>/` holds `bridge.log`,
   `tracker.log`, `bridge-state.json` and `tracking.npz`.

### PICO teleop

```bash
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70   # venv/pico
bash scripts/x2/start_pico.sh sim
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
```

`start_pico.sh robot PUBLISHER_IP` replaces the last two with the inline PC path (`--robot-io inline`,
`X2RobotIO`, `policy.yaml`, ROS 2 + AimDK sourced, mc released). It is **untested on hardware**.

Buttons: **A** Shaping to neutral, **A+B** policy Hold, **X** start / pause live reference, **B** alone damping.
Open TCP 28701 and 5592; OLI and X2 PICO share these ports, so run one per host. Needs `outputs/x2/pico_hold`.

## Safety notes & known issues

- **No HAL watchdog.** After a SIGKILL of the bridge or a power loss the last command stays in force. Keep the
  gantry and the physical stop within reach; the keyboard alone cannot stop the robot.
- **Never press the second `a` while the robot hangs.** On 2026-11-12 it hung tilted (13° forward, 7° sideways)
  with its feet off the floor; the policy stepped hard, flailed and spun on the rope until space. Use the order above.
- **Slow takeover.** `aima em stop-app mc` took more than 20 s on the X2 Basic; `release_mc.sh` judges success by
  `mc_main` exiting (waits up to 60 s).
- **Check the readings before `a`.** Passive on the gantry, the legs once read hip yaw ±1.7 rad. Shaping never
  commands a joint further out than it has been since `a`; a rejected command latches with the joint in `bridge.log`.
- **Damping.** With the training gains the arms oscillated at 4.4 Hz (→ arm kd ×5); then the hips rocked at 3.7 Hz
  (→ leg/waist kd ×2, found in sim2sim with 50 ms command delay; on the robot the legs then held 0.01–0.03 rad/s
  over 23 s of Hold). The robot's delay is estimated at 40–50 ms, so expect a possible recovery step.
- **Load on the feet.** The rope carried the weight (knees about 0 Nm); `15_walk_short` covered about 0.7 m of
  1.33 m. Walking with full weight on the feet is untested.
- **GC stall.** A garbage collection stalled the RK3588 loop for about 600 ms; the Tracker now calls `gc.freeze()`.
- **No 1000 Hz ROS subscribers during a run** (joint state 1000 Hz, IMU 500 Hz): use `watch.sh`, which reads
  the bridge's LowState. Never pin anything to core 4 (EtherCAT HAL).
- **Head.** `start_proteus_onboard.sh` defaults to `X2_NO_HEAD=0`. This X2 Basic's HAL publishes a head group that
  reads constant 0, so the default works. Set `X2_NO_HEAD=1` only if `probe.sh` prints `head : no messages`.
- **Simulation load.** Run at most 4 sims in parallel on a 31 GB PC.
- A stop, a command or state timeout over 0.2 s, or an invalid command latches damping; Ctrl-C sends damping.

## Further reading

- Onboard tutorial: [docs/tutorials/proteus-x2-onboard.md](../../docs/tutorials/proteus-x2-onboard.md) ·
  [中文](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2-onboard.md)
- Model, sim2sim and PICO teleop: [docs/tutorials/proteus-x2.md](../../docs/tutorials/proteus-x2.md) ·
  [中文](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md)
- Generic workflow: [../README.md](../README.md)
