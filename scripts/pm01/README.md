# EngineAI PM01 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

Proteus WAMSeq (wsq61 actor) on the 24-joint EngineAI PM01 / `pm_v2` (`engineai_pm01` contract; not the
25-joint T800 and not `engineai_pm01_edu_official`). The PC runs the Tracker (`sim2real/rl_policy/tracking.py`,
CPU ONNX at 50 Hz) and drives the robot over Ethernet through EngineAI's `joint_bridge` ROS 2 interface; a
separate 500 Hz sender republishes the latest command. The transport is the T800 one with two PM01 changes:
the **classic ankle parser** (`parallel_parser_type=0`) and acceptance of the zero header stamps of firmware
`76445d09`. This folder also holds the read-only checks and the bounded relay/bench tools used during the
2026-11 hardware investigation.

## Status

| Item | Value |
| --- | --- |
| Hardware | First gantry-protected walk on 2026-11-05: all 398 playback frames ran, no reset. **Weight-bearing exit unvalidated**; ankle pitch torque peaked at 106 Nm vs the 61 Nm training limit. Real PICO on hardware untested |
| `--robot` | `pm01` (aliases `pm_v2`, `engineai_pm01`) — `sim2real/config/robots/__init__.py` |
| Joints | 24, SDK order `J00`–`J23` (`J12_WAIST_YAW`, `J23_HEAD_YAW` is the only head joint), mapped by name |
| Robot I/O | `--robot-io inline` → `PM01RobotIO` (`interface_protocol` built from the robot's installed schema). Topics `/hardware/joint_state`, `/hardware/joint_command`, `/hardware/imu_info`, `/motion/motion_state`; Cyclone DDS, domain 69, NIC `eno1` |
| Tracker runs on | The PC (no inference install on the robot). Motion computer `user@192.168.0.163`, Orin `ubuntu@192.168.0.162` |
| Default policy | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml` (hardware) and `policy-sim.yaml` (sim) |
| Default motion | `checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz`; PICO hold `checkpoints/engineai_pm01/pico_hold` |

## Files

`scripts/pm01/` — launchers and setup:

| File | Purpose |
| --- | --- |
| `prepare.py` | Stages/verifies `pm_v2.xml` from `GMR-engineai` (axis tolerance 1e-4), exports wsq61 ONNX + YAMLs, writes `xrobot_to_pm01.json` / `bvh_lafan1_to_pm01.json`, builds the PICO hold, retargets four LAFAN1 clips. Flags `--sources-root`, `--proteus-root`, `--release` (default `wsq61`), `--skip-motions`. |
| `prepare_ros.sh` | Copies the **installed** `interface_protocol` msg/srv/action definitions from the robot (rsync, read-only) into `third_party/engineai_pm01/ros2` and builds them with colcon. Arg: `user@host` (default `user@192.168.0.163`). |
| `env.sh` | Source in the PC shell: ROS Humble + the built messages, `ROS_DOMAIN_ID`, Cyclone DDS bound to the NIC, HF offline flags. |
| `check_robot.py` | Read-only readiness check (subscribes only). Exit 0 when 24-joint feedback, IMU and `joint_bridge` are fresh, a compatible BEST_EFFORT command subscriber exists and no other command publisher does; else exit 2. `--duration` (default 5 s). |
| `start_proteus.sh` | Hardware file playback: sources `env.sh`, runs `check_robot` (skipped with `--help`), then `tracking --robot pm01 --robot-io inline --controller keyboard`. Optional first argument is the motion NPZ. |
| `start_pico.sh` | PICO teleop: `publisher`, `sim` (`--sim-dt 0.000625 --decimation 32`), `policy-sim [ip]`, `robot [ip]` (sources `env.sh` and runs `check_robot` first). |

`scripts/pm01/` — investigation tools (experimental; not a deployment path):

| File | Purpose |
| --- | --- |
| `shadow_policy.py` | Runs real PM01 feedback through the full Tracker/ONNX path for 775 frames without any command publisher. |
| `trial_policy.py` | Bounded PC client: publishes candidate commands only to `/sim2real/pm01/request`; needs the onboard relay. `--walk-once` requires new `--policy-signal` / `--play-signal` files. |
| `policy_bridge.py` | Onboard relay (motion computer): the only hardware writer during trials, with an independent supervisor that latches classic damping. Sends **RL_PARSER** for active commands. |
| `bench_damping.py` | Onboard ankle-damping bench; read-only by default, `--execute` allows ≤1 s of low ankle damping. Requires root and an exact motor-driver binary hash. Uses RL_PARSER for the trial. |
| `rehearse_trial.py` | No ROS: replays the relay gate in MuJoCo from a recorded hanging pose with a mocap-weld gantry. |

Related files elsewhere:

- `sim2real/config/robots/pm01.py` + `sim2real/config/robots/specs/engineai_pm01.yaml` (via `proteus_spec.py`).
- `sim2real/rl_policy/robot_io/pm01.py` — `PM01_SDK_JOINT_NAMES`, `PM01Codec` (classic parser, zero-stamp handling).
- `sim2real/rl_policy/robot_io/engineai.py` (shared with T800) and `sim2real/rl_policy/robot_io/ros2_named.py`.
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_pm01.json`.
- `scripts/proteus/` (`prepare_assets.py`, `retarget_lafan.py`, `evaluate.py`), `scripts/build_xrobot_profile.py`, `scripts/export_proteus_onnx.py`, `scripts/make_hold_motion.py`.
- Tests: `tests/test_pm01.py`, `tests/test_pm01_launch.py` (opt-in), `tests/test_pm01_policy_bridge.py`, `tests/test_pm01_bench.py`, `tests/test_proteus_transport.py`, `tests/test_pico_retarget_profiles.py`.
- Records: `docs/experiments/proteus-pm01.json`, `docs/experiments/proteus-pm01-recovery.json`.

## Setup

1. Environments: root project (`uv`) for inference/sim, `venv/pico` for the publisher and offline IK (`--no-sync`).
2. Sources under one parent: `Proteus-wamseq`, `GMR-engineai` (with `assets/engineai_pm01` and the PM01 IK
   profiles), `ubisoft-laforge-animation-dataset`. Then build artifacts (not in git):
   ```bash
   uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects
   ```
   Outputs: `third_party/engineai_pm01/` (model + `validation.json`), `checkpoints/proteus/wsq61/engineai_pm01/`,
   `checkpoints/engineai_pm01/{motions,pico_hold}`.
3. ROS messages from **this robot** (old firmware has `current_motion_task_name`; DDS needs matching definitions):
   ```bash
   bash scripts/pm01/prepare_ros.sh user@192.168.0.163
   source scripts/pm01/env.sh
   uv run --no-sync python -m scripts.pm01.check_robot
   ```
4. `env.sh` and the launchers export `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1` and `SIM2REAL_ORT_NUM_THREADS=1`.

## Adjustments

| Tunable | Current value | Where |
| --- | --- | --- |
| Ankle parser | `parallel_parser_type = 0` (classic, PD in the motor driver) for `start_proteus.sh` / `start_pico.sh robot` | `PM01Codec` in `robot_io/pm01.py` (T800 keeps `1`) |
| Ankle gains | kp 37.701, kd 2.4916 (unchanged training values). Under classic they act motor-side: joint-side pitch ≈4.09×, roll ≈1.11× | `specs/engineai_pm01.yaml` / `policy.yaml` |
| Zero header stamps | Accepted; freshness uses local monotonic receive time. A zero after a nonzero stamp is rejected | `PM01Codec._new_stamp` |
| IMU topic / extrinsic | `/hardware/imu_info`, `imu_to_pelvis_wxyz: [1, 0, 0, 0]` (IMU site aligned with pelvis) | `robot_io_options` in `policy.yaml` |
| Publish rate / timeout | `publish_hz: 500`, `timeout_s: 0.25`; mode heartbeat 1 s | `robot_io_options`; `EngineAICodec._fresh` |
| Stop damping / step bound | damping 2.0, 0.2 rad per publish (100 rad/s at 500 Hz) | `NamedJointCodec` defaults in `ros2_named.py` |
| Joint limits | Training limits; no vendor-limit overrides needed | `prepare.py` → `validation.json` |
| Model axis tolerance | `axis_atol=1e-4` (spec axes rounded to 4 decimals; measured max 8.76e-5) | `prepare.py` |
| Sim timestep | **0.000625 s**, decimation 32 | `start_pico.sh sim`; `scripts/proteus/evaluate.py` default for PM01 |
| DDS / network | `PM01_NETWORK_INTERFACE` (`eno1`), `PM01_ROS_DOMAIN_ID` (69), `PM01_ROS_SETUP`; existing `CYCLONEDDS_URI` wins | `env.sh` |
| Remote message path | `PM01_REMOTE_PROTOCOL` (default `/apps/engineai_robotics/_install/share/interface_protocol`) | `prepare_ros.sh` |
| Policy config | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | launchers |

## Usage

Offline evaluation and tests:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
source scripts/pm01/env.sh   # ROS loopback tests isolate themselves on localhost, domain 171
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py -q
PM01_RUN_LAUNCH_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
  python -m pytest tests/test_pm01_launch.py -q -s
```

Robot bring-up (gantry on, E-stop in reach; the launchers never switch modes):

1. Computer power, then the white power button for motor power (two short beeps, white rear/hip light).
   Do not use the red neck E-stop to power on.
2. On the motion computer, if inactive: `sudo systemctl start robotics.service` (it may wait on the
   motor check in `monitor.service` until motor power is on).
3. Controller on firmware `76445d09`: **LB + Start** (`idle`, motor-enable handshake) → **LB + RB** (`passive`)
   → **LB + A** (`pd_stand`, rejects poses >1.2 rad from standing) → **LB + D-pad left** (`joint_bridge`).
   BACK + A from newer vendor docs does not work on this release. Check joint readings, not appearance.

Hardware run (PC):

```bash
# Terminal 1
bash scripts/pm01/start_proteus.sh checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

1. Feet free, press nothing for 10 s: the robot should stay as quiet as `passive`.
2. **i** shapes to walk frame 0 over 10 s (arms rise to near horizontal).
3. Lower the gantry until the feet lightly touch, then **]** for policy Hold.
4. Transfer load gradually, then **Space** to play. **o** damps; **Ctrl+C** exits.

PICO teleop (publisher on the same PC):

```bash
# Terminal 1: XRoboToolkit PC Service + calibrated body/leg trackers
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: robot policy (or `sim` + `policy-sim 127.0.0.1` for MuJoCo)
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

Buttons: **A** shapes to neutral; release, then **A+B** enables Hold; **X** pauses/resumes; **B** alone damps.
Ports 28701 (motion) / 5592 (buttons); one robot per port set.

Shutdown: **o**, keep the gantry loaded, Ctrl+C (damping repeats on close), then **LB + RB** to `passive`.

Investigation tools (exact tested gains, firmware hash and hanging setup required; read the tutorial first):

```bash
uv run --no-sync python -m scripts.pm01.shadow_policy --policy-config <policy.yaml> --output <new.npz>
uv run --no-sync python -m scripts.pm01.rehearse_trial --policy-config <yaml> --pose-record <rec.npz> \
  --bridge-config <bridge.json> --output <out>
# On the motion computer, as root, with bench_damping.py copied beside it:
python3 policy_bridge.py --execute --config <bridge.json> --output <new-log.jsonl>
# Then on the PC (after `source scripts/pm01/env.sh`):
uv run --no-sync python -m scripts.pm01.trial_policy --policy-config <yaml> --motion <static.npz> --output <new.npz>
```

`policy_bridge.py` options: `--shaping-only --shaping-seconds {1,3,5,10}`, `--hold-seconds {1,3}`,
`--walk-shaping`, `--walk-once`. `bench_damping.py`: `--execute`, `--ankle-kd {0.05,0.1,0.2,0.3}`,
`--ankle-kp {0,…,37.701}`, `--duration`, `--output`.

## Safety notes & known issues

- RL_PARSER (`1`) oscillated the free ankles and reset the whole robot on 2026-11-04 and 2026-11-05 (estimated
  8–9 ms loop delay, real ankle-roll inertia 0.011–0.016 kg m² vs 0.04 in sim). Classic `0` is a deliberate
  deviation from vendor guidance; only `kp=0` classic commands were checked by motor readback.
- The relay/bench tools still send RL_PARSER for active commands; do not treat them as the deployment path.
- Damping alone does not hold a standing robot (`walk-deploy-005`): a feedback timeout during load transfer
  removed stiffness and the torso tilted. The gantry must be able to catch the robot at every step.
- First walk: ankle pitch torque 106 Nm (training limit 61 Nm; sim clips at 61), ankle-roll motion >5 Hz about
  2.5× sim, and ~33 s of rocking in the final Hold. Root position is not measured.
- Zero-stamp feedback cannot distinguish a republished old value from a new one. `check_robot` `ready: true` does
  not prove motors are enabled (24 disable flags were once set while `joint_bridge` was reported).
- Process death or power loss needs firmware-side protection; the bridge runner's command-age timeout is unknown.

## Detailed docs

- English: [`docs/tutorials/proteus-pm01.md`](../../docs/tutorials/proteus-pm01.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md)
- Vendor: [engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace), [PM01 parallel-ankle notes](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
