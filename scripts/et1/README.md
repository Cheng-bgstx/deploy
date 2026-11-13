# Galbot ET1 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

The Galbot ET1 runs the whole deployment on its own board (aarch64, CPython 3.10): an HDEP bridge
(`hdep_bridge.py`, 200 Hz) talks to the robot through a small C++ shim and serves the same local ZMQ
LowState/LowCmd contract as the G1 bridge, and the shared Tracker (`sim2real/rl_policy/tracking.py`)
runs CPU ONNX inference at 50 Hz next to it. It plays Proteus actors (proteus_60, WAM-62) and Galbot's
own et1-dep tracking policies. The supported route is this HDEP onboard stack; a sim-only Unitree-HG-DDS
ET1 route from another branch was intentionally not merged.

## Status

| Item | Value |
| --- | --- |
| Hardware | Tested on the robot. After the takeover and GC-stall fixes (commit `8da006e`), every takeover completed and proteus_60 `11_slow_close` played to the end and held standing; WAM-62 stood (`01_idle`); `shanshenbu` opening stand held, the full dance over-rotates after ~8 s. Details: [hardware status](../../docs/reference/proteus-et1-onboard.md#hardware-status) |
| `--robot` | `galbot_et1` (alias `et1`) — `sim2real/config/robots/__init__.py` |
| Joints | 26: 6 per leg, waist roll/yaw, 5 per arm, head pitch/yaw |
| Robot I/O | `--robot-io zmq` on 127.0.0.1 ⇄ `hdep_bridge.py` ⇄ HDEP (`libhdep_bridge_galbot_et1.so.2.2.0`) |
| Controller | `--controller bridge_joystick` (handset republished by the bridge, Unitree key layout) |
| Tracker runs on | The ET1 board, pinned to CPUs 5,6; bridge on CPU 7 |
| Default policy | `checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml` |
| Other policies | `checkpoints/proteus/proteus_wamseq_62/galbot_et1/policy-pc.yaml`, `checkpoints/galbot_track/galbot_et1/<name>/policy-pc.yaml` |
| Default motion | `checkpoints/proteus/proteus_60/galbot_et1/motions/01_idle` |

## Files

`scripts/et1/`:

| File | Purpose |
| --- | --- |
| `deploy_onboard.sh` | PC → board: resolves aarch64/CPython 3.10 wheels once, rsyncs code, checkpoints, MJCF and HF assets, builds the shim and probe on the board. Never starts the controller. |
| `start_proteus_onboard.sh` | Board launcher: checks for other controllers, starts bridge + Tracker pinned to CPUs, writes logs under `logs/run-<UTC>/`. |
| `hdep_bridge.py` | HDEP ⇄ ZMQ bridge at 200 Hz, arbitrated by `BridgeSafety`; deferred takeover via `enter_debug_mode()`. |
| `hdep_shim.cpp` | C ABI over HDEP for ctypes (`hdep_py` exists only for x86_64 / CPython 3.11). Built on the board into `libet1_hdep_shim.so`. |
| `full_state_probe.cpp` | Read-only dump of all joint motors, IMU, gamepad and robot state. Built on the board into `probe/full_state_probe`. |
| `operator.sh` | Wrapper that runs `operator_keyboard.py` on the board (`ssh -t`). |
| `operator_keyboard.py` | Keyboard operator for runs without a handset (`a`, `s`, space/`x`). |
| `press_key.py` | Sends one key press (`a`, `s`, `space`) for scripted/remote runs. |
| `prepare_galbot_track_policy.py` | Converts an et1-dep tracking checkpoint (policy protocol 3.2) into a Tracker policy + qpos motion. |
| `slow_motion_tail.py` | Slows the end of a qpos motion and appends a still end (builds `11_slow_close`). |

Related files elsewhere:

- `sim2real/config/robots/et1.py` — joint order, limits, armature, gains, stand pose, MJCF path.
- `sim2real/rl_policy/robot_io/zmq.py` — Tracker side of the ZMQ contract (`SIM2REAL_IO_TIMEOUT_S`).
- `sim2real/rl_policy/robot_io/bridge_safety.py` — command arbitration shared with the G1 bridge.
- `sim2real/rl_policy/controllers/bridge_joystick.py` — reads the handset on `PORTS["gamepad"]` (5594).
- `sim2real/utils/operator_keys.py` — operator key messages on `PORTS["operator"]` (5595).
- `sim2real/rl_policy/observations/galbot_track.py` — 111-dim observation of Galbot tracking actors.
- `scripts/g1/prepare_proteus_deployment.py`, `scripts/g1/validate_proteus_deployment.py`, `scripts/export_proteus_onnx.py`, `scripts/prepare_proteus_test_motions.py` — shared Proteus tooling used with `--robot galbot_et1`.
- Tests: `tests/test_et1_bridge.py`, `tests/test_et1_config.py`, `tests/test_galbot_track_obs.py`.
- Docs: `docs/reference/proteus-et1-onboard.md` and its Chinese twin.

## Setup

One-time, on the PC:

1. Get Galbot's et1-dep delivery and point `ET1_DEP_ROOT` at it (default `~/et1-dep`). The MJCF
   `storage/humanoid_description/galbot_et1/galbot_et1_26dof.xml` is not vendored; `et1.py` reads it from there.
   `deploy_onboard.sh` also copies the aarch64 HDEP headers/libraries, CycloneDDS libraries and the
   `galbot_et1` HDEP bridge library from it.
2. Runtime artifacts live under `checkpoints/` (git-ignored; shared Drive folder per `AGENTS.md`).
   The deploy script ships `checkpoints/proteus/proteus_60/galbot_et1`,
   `checkpoints/proteus/proteus_wamseq_62/galbot_et1` and, if present, `checkpoints/galbot_track/galbot_et1`.
3. The HF caches `models--elijahgalahad--h2_model` and `models--elijahgalahad--g1_xmls` must exist in
   `~/.cache/huggingface/hub` on the PC; they are rsynced to the board because modules import them at load.
4. `uv` on the PC (override with `UV=`); `ssh`/`rsync` access to the board (`~/.ssh/config` supplies user/port).

On the board (done by `deploy_onboard.sh`):

- System `python3` (CPython 3.10, aarch64) with packages from `<target>/site-packages`
  (`uv export --extra inference-cpu`, `--python-platform aarch64-manylinux_2_35`, CPU PyTorch index).
  The cache in `outputs/et1-onboard-env` is reused while `uv.lock` is unchanged (`ET1_ENV_CACHE` overrides).
- `g++` (C++17) builds `probe/libet1_hdep_shim.so` and `probe/full_state_probe` against `probe/include` and `probe/lib`.
- Board layout under the target: `repo/ site-packages/ et1-dep/storage/ hf/ probe/{lib,bridge,...} logs/`.
- `start_proteus_onboard.sh` exports `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=<target>/hf`.
  Keep both HF variables in prepared commands as well.

## Adjustments

| What | Current value | Set in |
| --- | --- | --- |
| Armature groups (large / ankle pitch / ankle roll / small) | 0.01890 / 0.01492 / 0.00619 / 0.00398 | `sim2real/config/robots/et1.py` |
| Safe PD kp | 74.6 / 58.9 / 24.4 / 15.7 | `et1.py` (`_KP`) |
| Safe PD kd (also fault damping) | 4.75 / 3.75 / 1.55 / 1.00 | `et1.py` (`_KD`) |
| Joint ranges | from et1-dep MJCF; waist yaw ±0.7854 (not the ±π/2 of the Proteus spec) | `et1.py` (`_LIMITS`) |
| Joint friction loss | 0.1 (et1-dep MJCF value) | `et1.py` |
| Stand pose / root height | Galbot walker pose, root z 0.74 | `et1.py` (`_STAND_JOINT_POS`, `default_qpos`) |
| Head lock | head pitch/yaw soft limit `[0, 0]` (actor tracks no head; otherwise yaws ~1 rad) | `scripts/g1/prepare_proteus_deployment.py` (`DEFAULT_LOCKED_JOINTS`, `--locked-joints`) |
| Overlay | 5% inset soft limits, 15 rad/s target slew, 2 s reference entry | `prepare_proteus_deployment.py`, `prepare_galbot_track_policy.py` |
| CPU pinning | bridge `PROTEUS_BRIDGE_CPU=7`, Tracker `PROTEUS_TRACKER_CPU=5,6`; 1 ORT/BLAS thread | `start_proteus_onboard.sh` |
| I/O timeout | `PROTEUS_IO_TIMEOUT_S=0.2` → `SIM2REAL_IO_TIMEOUT_S` and bridge `--timeout-s` | `start_proteus_onboard.sh` |
| Operator key freshness | 0.3 s (`OPERATOR_FRESH_S`) | `hdep_bridge.py` |
| Takeover resume wait | 0.5 s (`TAKEOVER_RESUME_S`) | `hdep_bridge.py` |
| HDEP interface / rate | `--interface-prefix 192.168.123`, `--rate 200` | `hdep_bridge.py` |
| Board host / target | `ET1_HOST=192.168.121.42`, `ET1_TARGET=/root/proteus_et1_claude` | `deploy_onboard.sh` |

Policy and motion selection (board launcher):

- `PROTEUS_POLICY_DIR` — `checkpoints/proteus/proteus_60/galbot_et1` (default),
  `checkpoints/proteus/proteus_wamseq_62/galbot_et1`, or `checkpoints/galbot_track/galbot_et1/<name>`.
  The launcher asserts the YAML has a `galbot_track` section or `proteus.reference_forcing: true`.
- `PROTEUS_MOTION` — motion dataset directory (needs `manifest.json`).
- `PROTEUS_PLAYBACK_LIMIT_S` — stops the reference clock there and holds (`--npz-playback-limit-s`).
- `PROTEUS_READ_ONLY=1` (+ optional `PROTEUS_MAX_RUNTIME_S`) — dry run, never enters debug mode or writes motors.
- `PROTEUS_RUN_DIR`, `PROTEUS_ROOT` — override the log directory / install root.

Rebuilding artifacts (PC):

```bash
python scripts/g1/prepare_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml          # -> policy-pc.yaml
python scripts/prepare_proteus_test_motions.py --robot galbot_et1                # Proteus test clips
python scripts/et1/slow_motion_tail.py \
  --source checkpoints/proteus/proteus_60/galbot_et1/motions/11_stand_walk_straight \
  --output checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close       # 2x slower after 10.4 s, 1 s still end
```

`slow_motion_tail.py` options: `--after-s 10.4`, `--factor 2.0`, `--still-s 1.0`; it refuses to overwrite an existing output.

et1-dep tracking policies: `prepare_galbot_track_policy.py` needs a checkpoint folder with `policy.toml`
(protocol 3.2, 50 Hz), `inference.toml`, `policy.onnx` and `motion.npz`, and a system `python3 >= 3.11`
to read TOML. It writes `policy.onnx`, `policy.yaml`, `policy-pc.yaml` (fixed joints locked, 5% inset,
`--max-target-velocity-rad-s 15`, `--reference-entry-s 2`) and `motions/<name>`:

```bash
python scripts/et1/prepare_galbot_track_policy.py \
  --checkpoint checkpoints/et1-dep-addon/26-1111/storage/ckpt-shanshenbu \
  --output checkpoints/galbot_track/galbot_et1/shanshenbu
```

## Usage

1. Rehearse in MuJoCo (PC) with the same Tracking, BridgeSafety and A → A → Start flow:

   ```bash
   python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 \
     --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
     --motion-path checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close --walking-s 20
   ```

   A clip shorter than the schedule ends in Hold early, so `passed` can be false on the mode sequence
   alone; read the stability fields. For a Galbot tracking policy, see the `--scenario walk-hold` example in the docs.

2. Deploy from the PC (touches only `ET1_TARGET` on the board):

   ```bash
   ET1_HOST=192.168.121.42 ET1_TARGET=/root/proteus_et1_claude scripts/et1/deploy_onboard.sh
   ```

3. Stop Galbot's Loco, its gamepad programs and any other tracker on the board. Start the stack in one SSH terminal:

   ```bash
   ssh -t 192.168.121.42 'cd /root/proteus_et1_claude/repo && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
     PROTEUS_MOTION=checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close \
     bash scripts/et1/start_proteus_onboard.sh'
   ```

   Galbot tracking policy, opening stand only:
   `PROTEUS_POLICY_DIR=checkpoints/galbot_track/galbot_et1/shanshenbu PROTEUS_MOTION=checkpoints/galbot_track/galbot_et1/shanshenbu/motions/shanshenbu PROTEUS_PLAYBACK_LIMIT_S=2.5`.

4. Without a handset, open the keyboard operator in a second terminal:

   ```bash
   ssh -t 192.168.121.42 /root/proteus_et1_claude/repo/scripts/et1/operator.sh
   ```

   For scripted presses on the board: `python scripts/et1/press_key.py {a,s,space}` (with the same `PYTHONPATH` as `operator.sh`).

5. Operator sequence (robot on the gantry):

   | Step | Handset | Keyboard | Effect |
   | --- | --- | --- | --- |
   | Wait | — | — | `bridge.log`: `Tracker and LowState ready` |
   | 1 | A | `a` | Bridge enters debug mode (takeover); Shaping to the stand pose over 10 s |
   | 2 | A | `a` | 2 s transition to the reference start, then `Hold ready` in `tracker.log` |
   | — | — | — | Lower the gantry, keep it in reach |
   | 3 | Start | `s` | Play the motion; `11_slow_close` ends standing |
   | Stop | Select, or L1 + Down | space or `x` | Latch the stop; active support is dropped |

   The launcher prints the run directory; follow it with `tail -f <run>/bridge.log <run>/tracker.log`.

6. Shutdown: hold the robot on the gantry, then Ctrl-C in the launcher terminal. It stops both
   processes; the bridge latches and sends damping for about 0.2 s before exiting. Logs, `bridge-state.json`
   and `tracking.npz` are in `<target>/logs/run-<UTC>/`.

Board-side state check without control: `probe/full_state_probe 192.168.123 probe/bridge/libhdep_bridge_galbot_et1.so.2.2.0 [SAMPLES] [INTERVAL_MS]`
(run from `<target>` with `LD_LIBRARY_PATH=probe/lib`).

## Safety notes & known issues

- **One motor writer.** Galbot Loco, its gamepad programs (`gamepad-recover*`, `motion-sender`) and other
  trackers must not run concurrently. The launcher refuses to start if it finds them, and `logs/active.lock`
  prevents a second supervisor.
- **The keyboard cannot stop the robot by itself.** A dropped SSH session only releases keys (they expire
  after 0.3 s). Keep the gantry and the hardware stop ready on every run.
- **Takeover blocks the bridge.** `enter_debug_mode()` took 0.11–0.86 s on hardware. The bridge drops the
  commands queued during the block and waits up to 0.5 s for a fresh Tracker command; a Tracker that stays
  silent fails the takeover. It is only requested after a ready Tracker and a fresh A press.
- **GC freeze.** A full garbage collection over ~350 000 setup objects took 442 ms on the board and latched
  damping on the 0.2 s timeout; the Tracker now calls `gc.freeze()` before its loop (`base_policy.py`).
- **Shaping past a limit.** `BridgeSafety` accepts Shaping (`init`) targets between the measured position
  and the limit, so a robot resting past a model limit (seen: waist yaw 1.32 rad) is brought back, never pushed further.
- **Head is locked** in the default overlays; `policy-pc-head-free.yaml` exists for proteus_60 but is not the default.
- **`shanshenbu`** was trained without command delay; on hardware it over-rotates and tilts after ~8 s.
  Only the opening stand (first 2.5 s) is safe to cut and hold.
- The bridge exits without HDEP teardown (`os._exit`), since destroying the HDEP bridge can race a CycloneDDS callback.

## Further reading

- [docs/reference/proteus-et1-onboard.md](../../docs/reference/proteus-et1-onboard.md)
- Chinese: [proteus-et1-onboard.md](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-et1-onboard.md)
- Generic workflow: [../README.md](../README.md)
