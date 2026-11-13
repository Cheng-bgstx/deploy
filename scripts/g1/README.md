# Unitree G1 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

Launchers and tools for running Proteus tracking policies on the Unitree G1 (29 DoF). Two processes do the work: the
**Bridge** (`real_bridge.py`, 200 Hz, Python `unitree_sdk2py` over DDS, arbitrated by `BridgeSafety`) and the
**Tracker** (`sim2real/rl_policy/tracking.py`, 50 Hz, CPU ONNX), linked by the local ZMQ low_state / low_cmd contract.
Both run either on an external PC wired to the robot (`start_proteus_pc.sh`) or on the G1's own Orin NX
(`onboard.sh` → `start_proteus_onboard.sh`); the remote flow is the same: A → A → Start, Select to stop.

## Status

| Item | Value |
| --- | --- |
| Hardware status | External PC: hardware-tested with the default wsq62 actor ([README](../../README.md), [PC doc](../../docs/proteus_g1_pc.md); hardware incidents recorded 2026-11-04). Onboard Orin: read-only timing runs on 2026-11-05; [the onboard doc](../../docs/proteus_g1_onboard.md) still says the onboard path has not been tested on hardware. |
| `--robot` alias | `g1` (`G1_CFG.name`, `sim2real/config/robots/__init__.py`) |
| Joints | 29 (6 per leg, 3 waist, 7 per arm) |
| Robot I/O | `--robot-io zmq` + `scripts/g1/real_bridge.py` (`unitree_sdk2py`, CycloneDDS 0.10.2, `hg` LowState/LowCmd). Inline `--robot-io inline` (`unitree_interface`, aarch64 only) exists but the Proteus launchers do not use it. |
| Tracker runs on | PC (`start_proteus_pc.sh`, interface `enp5s0`) or onboard Orin (`start_proteus_onboard.sh`, interface `eth0`) |
| Default policy | `checkpoints/proteus/proteus_wamseq_62/unitree_g1/` (`policy.onnx`, `policy.yaml`, generated `policy-pc.yaml`); override with `PROTEUS_POLICY_DIR`. Alternatives: `checkpoints/proteus/wsq61/unitree_g1` (not covered by hardware results), `checkpoints/proteus/proteus_60/unitree_g1` (older actor). |
| Default motion | `PROTEUS_CLIP=walk_straight_short` → `checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short` |

## Files

| File | Purpose |
| --- | --- |
| `start_proteus_pc.sh` | PC launcher: preflight, then Bridge + LowState gap monitor + Tracker, CPU-pinned; logs to `outputs/proteus-pc/<time>/`. |
| `start_proteus_onboard.sh` | Orin launcher: sets the offline Python/CycloneDDS/HF environment and core pinning, then runs `start_proteus_pc.sh`. |
| `onboard.sh` | PC-side control of the onboard run: `deploy`, `start`, `status`, `tail`, `stop`, `logs`. |
| `deploy_onboard.sh` | Push CPython 3.10, offline aarch64 packages, code, policies, motions and HF cache to the Orin; never starts a controller. |
| `onboard_build_cyclonedds.sh` | Run on the Orin by `deploy_onboard.sh`: builds CycloneDDS 0.10.2 and its Python binding offline. |
| `prepare_proteus_deployment.py` | Writes the `policy-pc.yaml` deployment overlay next to an exported `policy.yaml` (also used for ET1 with `--robot`). |
| `prepare_proteus_motion.py` | Cuts a test clip (`--clip`) from the pinned `elijahgalahad/any4hdmi-g1-lafan` dataset; refuses to overwrite. |
| `validate_proteus_deployment.py` | Headless MuJoCo rehearsal of the real Tracker + `BridgeSafety` (no DDS). |
| `real_bridge.py` | SDK2 ⇄ ZMQ Bridge with deferred takeover, stop arbitration, `--hold-through-gaps`, `--read-only`. |
| `real_bridge_cpp.py` | Same ZMQ contract over the C++ `unitree_interface` binding (aarch64). |
| `lowstate_gap_monitor.py` | Read-only LowState receiver that logs reception gaps over 30 ms; never publishes. |
| `joint_response_test.py` | Leg joint sine/step response on a hanging robot (`--target real`) or in MuJoCo (`--target sim`). |
| `scalecar/start_bridge.sh`, `scalecar/start_tracker.sh` | Scalecar vehicle-entry Bridge and Tracker (Planner, joystick) launchers. |
| `depth_stream/` | RealSense depth capture program for the Orin (`build.sh` → `build/scalecar-depth`), copied from Scalecar_v2 (`SOURCES.json`). |

Related files elsewhere:

| Area | Path |
| --- | --- |
| Robot config | `sim2real/config/robots/g1.py` (joint limits, `safe_joint_kp/kd`, armature, MJCF `hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml`) |
| Robot I/O | `sim2real/rl_policy/robot_io/zmq.py`, `bridge_safety.py`, `g1.py` (inline), `g1_debug_mode.py` (MotionSwitcher release) |
| Controllers | `sim2real/rl_policy/controllers/unitree_joystick.py` (`joystick`), `bridge_joystick.py` (onboard), `keyboard.py` (sim) |
| Teleop | `sim2real/teleop/pico_retarget_pub.py` (GMR `xrobot_to_g1` from `venv/pico`), [docs/pico_to_g1.md](../../docs/pico_to_g1.md) |
| Scalecar tooling | `scripts/scalecar/` (scene build, motion conversion), `docs/adr/` |
| Tests | `tests/test_g1_bridge_safety.py`, `tests/test_planner_real_robot.py` |
| Setup skill | `.agents/skills/configure-g1-sim2real/` (`scripts/inspect_host.sh`, `scripts/verify_install.py`) |

## Setup

1. **Root environment (policy, Bridge, sim).** On the PC:

   ```bash
   uv sync --extra inference-cpu --extra robot-g1
   ```

   The launchers use `.venv/bin/python` (override with `SIM2REAL_PYTHON`). If the `unitree_sdk2py` build cannot find
   CycloneDDS, follow [Root Project](../../docs/getting-started/root-project.md). Teleop uses `venv/pico`
   (`uv sync --project venv/pico`).
2. **Network.** PC Ethernet on `192.168.123.x` to the robot LAN (find the name with `ip -br link`; launcher default
   `enp5s0`, override with `G1_INTERFACE`). The Orin is `192.168.123.164` (user `unitree`); `deploy_onboard.sh` and
   `onboard.sh` expect an SSH alias `g1` in `~/.ssh/config` (override with `G1_HOST`). See
   [Network Configuration](../../docs/getting-started/network-configuration.md).
3. **Artifacts (not in git).** `checkpoints/proteus/...` and `third_party/` come from the shared Google Drive folder
   (`gdrive:projects/simple-tracking/sim2real`, [Download Artifacts](../../docs/artifacts.md)). Cache the G1 MJCF and the
   motion dataset once with network access; `prepare_proteus_motion.py` downloads the pinned source clip.
4. **Offline Hugging Face.** In every shell, and in `~/.bashrc` on the robot:

   ```bash
   export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
   ```

   The launchers also export both. `start_proteus_onboard.sh` points `HF_HOME` at the pushed cache.
5. **Onboard only.** `deploy_onboard.sh` needs an x86 `uv` binary (`UV`, default
   `outputs/g1-onboard-env/dl/uv/uv-x86_64-unknown-linux-gnu/uv`). Everything lands in
   `/home/unitree/proteus_g1_onboard` (`G1_TARGET`); no root needed on the Orin.

## Adjustments

**Policy overlay** — `prepare_proteus_deployment.py` writes `policy-pc.yaml`; `policy.yaml` stays untouched. The launcher
generates it only when missing, so delete or regenerate it after changing the script.

| Key in `policy-pc.yaml` | G1 value | Effect |
| --- | --- | --- |
| `joint_kp` / `joint_kd` waist roll, pitch | 100 / 2.5 (yaw stays 100 / 2 from `policy.yaml`) | Trained waist gains caused high-frequency waist tremor on hardware. |
| `soft_joint_pos_limits` | joint limits ± `PROTEUS_TARGET_MARGIN_RAD` = 1.0 rad | Proteus trains unclipped; ankles push targets up to ~0.8 rad past range. Other robots get a 5% inset. |
| `max_target_velocity_rad_s` | 15.0 | Target slew limit. |
| `shaping_target` | `default_joint_pos` | Shaping goes to the arms-down default stand (10 s, fixed in `base_policy.py`). |
| `reference_entry_s` | 2.0 | Second A interpolates to the clip's first frame, then Hold. |
| locked joints (`--locked-joints`) | none for G1 | Zero-width soft limit at the default pose (ET1 locks its head). |

`proteus.reference_forcing` must stay `true` (the launcher asserts it). No `robot_io_options` or IMU extrinsics are set
for G1.

**Bridge / safety** (`real_bridge.py`, `sim2real/rl_policy/robot_io/bridge_safety.py`):

| Setting | Default | Notes |
| --- | --- | --- |
| `--timeout-s` / `PROTEUS_IO_TIMEOUT_S` | 0.1 standalone, 0.25 in launcher | Same window goes to the Tracker as `SIM2REAL_IO_TIMEOUT_S`; range (0, 1]. |
| `--hold-through-gaps` / `PROTEUS_HOLD_THROUGH_GAPS` | off standalone, `1` in launcher | Keep last PD targets through gaps; gaps over 1 s (`MAX_HOLD_GAP_NS`) or a restarted tick still latch damping. `0` restores latched damping. |
| `--target-margin-rad` | 1.0 | Must cover the policy's soft limits; accepted range [0, 1.5]. |
| Command bounds | kp ≤ 400, kd ≤ 10 | Commands outside latch a fault. |
| Fault damping | `safe_joint_kd` in `g1.py` | Applied with kp = 0 after a fault. |

**Launcher environment** (`start_proteus_pc.sh` unless noted):

| Variable | Default (PC / onboard) | Purpose |
| --- | --- | --- |
| `PROTEUS_POLICY_DIR` | `checkpoints/proteus/proteus_wamseq_62/unitree_g1` | Policy directory. |
| `PROTEUS_CLIP` | `walk_straight_short` | Also `walk_forward_to_stand`, `dance_in_place`, `dance_steps`, `jump_once`. |
| `PROTEUS_MOTION_PATH`, `PROTEUS_PLAYBACK_LIMIT_S` | clip dir, clip length | Override reference / playback limit. |
| `PROTEUS_READ_ONLY`, `PROTEUS_READ_ONLY_S` | `0`, `5` | Bridge `--read-only --max-runtime-s`: real feedback and inference, no motor command. |
| `PROTEUS_CONTROLLER` | `joystick` / `bridge_joystick` | Onboard uses the Bridge-published handset to save a core. |
| `PROTEUS_BRIDGE_CPUS`, `PROTEUS_TRACKER_CPUS`, `PROTEUS_MONITOR_CPUS` | `0-3,6-15`, `4,5` / `1`, `2-5`, `6` | `taskset` pinning. |
| `SIM2REAL_ORT_NUM_THREADS` | `1` / `4` | ONNX threads. |
| `G1_INTERFACE` | `enp5s0` / `eth0` | DDS interface. |
| `PROTEUS_RUN_DIR` | `outputs/proteus-pc/<time>` | Log directory. |
| `G1_HOST`, `G1_TARGET`, `G1_POLICY_DIRS`, `G1_RSYNC_BWLIMIT` (deploy) | `g1`, `/home/unitree/proteus_g1_onboard`, wsq62 + ht500, `20000` KiB/s | Onboard deploy. |

## Usage

**1. Prepare (once per policy / clip; the launcher does both when missing).**

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
uv run python scripts/g1/prepare_proteus_deployment.py \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy.yaml
uv run python scripts/g1/prepare_proteus_motion.py --clip walk_straight_short
```

**2. Sim2sim rehearsal.** Headless, with the real Tracker and `BridgeSafety`:

```bash
uv run python scripts/g1/validate_proteus_deployment.py --scenario walk-hold \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy-pc.yaml \
  --motion-path checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short \
  --npz-playback-limit-s 5.1
```

Add `--push-velocity-m-s`, `--action-delay-ms`, `--state-gap-s` or `--joint-friction <yaml>` to stress it.
Interactive alternative: `uv run sim2real/sim_env/base_sim.py` in one terminal and
`uv run sim2real/rl_policy/tracking.py --robot g1 --robot-io zmq --controller keyboard --policy-config <policy-pc.yaml> --motion-path <clip>`
in another (keys `i`, `]`, space, `o`).

**3a. Hardware from the PC.** Robot on the gantry, operator on the remote and the hardware stop.

```bash
G1_INTERFACE=enp5s0 PROTEUS_READ_ONLY=1 scripts/g1/start_proteus_pc.sh   # read-only bring-up
G1_INTERFACE=enp5s0 scripts/g1/start_proteus_pc.sh                       # protected session
```

**3b. Hardware on the Orin.**

```bash
scripts/g1/onboard.sh deploy
PROTEUS_READ_ONLY=1 PROTEUS_READ_ONLY_S=60 scripts/g1/onboard.sh start
scripts/g1/onboard.sh start      # protected session; refuses while a PC controller runs
scripts/g1/onboard.sh tail       # Ctrl+C leaves the robot running
scripts/g1/onboard.sh status
```

**4. Operator sequence** (wait for `Tracker and LowState ready` in `bridge.log` first):

| Remote (sim keyboard) | Result |
| --- | --- |
| A (`i`) | Release high-level mode; Shaping to the arms-down stand over 10 s. Wait for `Joystick: Shaping complete`. |
| A again (`]`) | 2 s transition to the clip's first frame, then Hold. Wait for `Hold ready`. |
| Lower the gantry | Feet on the floor while Hold is active; keep protection. |
| Start (space) | Play / pause the clip; at the playback limit it stays in Final Hold. |
| Select (`o`) | Latched software stop: targets follow measured joints, PD gains kept, no active posture support. |

After Select or a fault restart with A → A → Start.

**5. Shutdown.** PC: Ctrl+C (or SIGTERM) on the launcher; the Bridge sends a brief damping sequence. Onboard:
`scripts/g1/onboard.sh stop`, then `scripts/g1/onboard.sh logs` to copy runs to `outputs/proteus-onboard/`.

**Additional tasks**

- *Scalecar vehicle entry*: `bash scripts/g1/scalecar/start_bridge.sh` and `bash scripts/g1/scalecar/start_tracker.sh`
  (overrides `SCALECAR_PYTHON`, `SCALECAR_POLICY_CONFIG`, `SCALECAR_MOTION_PATH`, `G1_SSH_HOST`, `G1_DEPTH_EXECUTABLE`);
  artifacts and depth build in [Scalecar Vehicle Entry](../../docs/tutorials/scalecar-vehicle-entry.md). Not yet run on hardware.
- *Joint response*: `uv run python scripts/g1/joint_response_test.py --target sim`; `--target real --interface enp5s0`
  only with the robot hanging clear and no Bridge/Tracker running.
- *PICO teleop*: [Pico Teleoperation](../../docs/tutorials/pico-teleoperation.md).
- *Tests*: `uv run --with pytest python -m pytest tests/test_g1_bridge_safety.py tests/test_planner_real_robot.py`.

## Safety notes & known issues

- **No response after A (motors disabled).** If the Bridge reports active but the robot does not move and every
  LowState `motor_state[i].motorstate` reads `0x40000000` (tau_est ≈ 0), the motors are not enabled. Stop the launcher
  first, then reboot the robot; relaunch once motorstate reads 0. Re-enabling under a live session snaps to the target.
- **Robot reboot mid-session.** Stop the launcher before rebooting: a 117 s reboot once let the Bridge resume old
  targets for ~3 s. Gaps over 1 s or a backwards tick now latch.
- **Reception gaps.** A 343 ms LowState gap during walking latched damping and the robot fell (2026-11-04). Holding
  through gaps helps standing Hold; blind intervals over ~0.3 s while walking still usually fall in simulation.
  Compare `bridge.log` with `lowstate-monitor.log` to tell robot/network from Bridge-process gaps.
- **"Tracker and LowState ready" never appears.** Check the NIC actually moves packets (`ip -s link show <iface>`
  counters) before blaming the robot; keep the PC from sleeping during tests.
- One writer per robot: never run PC and onboard sessions at once; the launchers refuse a second controller.
- Select and a killed process are not a hardware E-stop; Select drops active posture support. Keep the gantry
  through Shaping and Hold and the hardware stop in reach.
- Local reference forcing has no root localisation and drifts (about 0.5 m sideways over a walk).
- Environment changes apply at the next launch only. The Orin clock runs ~12 min behind the PC; run directories use
  Orin time.

## Detailed docs

- [Proteus G1 from an external PC](../../docs/proteus_g1_pc.md)
- [Proteus G1 on the robot's Orin](../../docs/proteus_g1_onboard.md)
- [Robot I/O](../../docs/robot_io.md) · [Root Project](../../docs/getting-started/root-project.md) ·
  [Network Configuration](../../docs/getting-started/network-configuration.md)
- [Scalecar Vehicle Entry](../../docs/tutorials/scalecar-vehicle-entry.md) · [ADRs](../../docs/adr/)
- [Pico Teleoperation](../../docs/tutorials/pico-teleoperation.md) · [PICO to G1 mapping](../../docs/pico_to_g1.md)
