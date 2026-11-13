# EngineAI PM01 launchers

[中文](README_zh.md) · Generic workflow (Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist): [../README.md](../README.md)

Proteus WAMSeq on the 24-joint EngineAI PM01 / `pm_v2` (`engineai_pm01` contract; not the 25-joint T800 and not
`engineai_pm01_edu_official`). The Tracker (`sim2real/rl_policy/tracking.py`, CPU ONNX at 50 Hz) runs either on
the PC, driving the robot over Ethernet, or **onboard** on the PM01 motion computer (Nezha). Both use EngineAI's
`joint_bridge` ROS 2 interface with a separate sender that republishes the latest command. The transport is the
T800 one with PM01 changes: the **classic ankle parser** (`parallel_parser_type=0`), acceptance of the zero header
stamps of firmware `76445d09`, and optional scales on the commanded ankle and arm gains (set by the onboard
launcher). This folder also holds the read-only checks and the bounded relay/bench tools used during the 2026-11
hardware investigation.

## Status

| Item | Value |
| --- | --- |
| Hardware (PC) | First gantry-protected walk on 2026-11-05 (`wsq61`, PC route): all 398 playback frames ran, no reset. **Weight-bearing exit unvalidated**; ankle pitch torque peaked at 106 Nm vs the 61 Nm training limit. Real PICO on hardware untested |
| Hardware (onboard) | First onboard gantry run on 2026-11-13 (`proteus_wamseq_62_full` on the Nezha, `joint_bridge`). It established that the onboard stack runs on the robot's own ROS, and found two problems: the unpaced ROS callback thread stretched a policy step to 22–28 ms (fixed by pacing it at 200 Hz on CPU 0: inference ≈12 ms, step ≈17.5 ms on average, ~10% of steps slightly over 20 ms), and Hold and the wave rocked the pelvis at 4.3–4.7 Hz with a taut rope carrying most of the weight (answered by ankle kd ×1.5, **checked in simulation only**). The onboard walk is not recorded as run; weight-bearing behaviour remains unvalidated |
| `--robot` | `pm01` (aliases `pm_v2`, `engineai_pm01`) — `sim2real/config/robots/__init__.py` |
| Joints | 24, SDK order `J00`–`J23` (`J12_WAIST_YAW`, `J23_HEAD_YAW` is the only head joint), mapped by name |
| Robot I/O | `--robot-io inline` → `PM01RobotIO` (`interface_protocol` built from the robot's installed schema). Topics `/hardware/joint_state`, `/hardware/joint_command`, `/hardware/imu_info`, `/motion/motion_state`; Cyclone DDS, domain 69 (PC NIC `eno1`) |
| Tracker runs on | The PC (`start_proteus.sh`, `start_pico.sh`), or the motion computer `user@192.168.0.163` (`start_proteus_onboard.sh`, deployed to `~/proteus_pm01`). Orin `ubuntu@192.168.0.162` is not used by these launchers |
| Default policy | PC launchers (`start_proteus.sh`, `start_pico.sh robot`): `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml`; `start_pico.sh policy-sim`: `policy-sim.yaml` of the same release. Onboard (`start_proteus_onboard.sh`, `deploy_onboard.sh`): `checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml`. `PROTEUS_POLICY_CONFIG` overrides any of them |
| Default motion | PC: `checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz`; PICO hold `checkpoints/engineai_pm01/pico_hold`. Onboard: `wave` → `checkpoints/engineai_pm01/wave/motions/wave_right.npz` (default), `walk` → the same walk clip |

## Files

`scripts/pm01/` — launchers and setup (PC):

| File | Purpose |
| --- | --- |
| `prepare.py` | Stages/verifies `pm_v2.xml` from `GMR-engineai` (axis tolerance 1e-4), exports ONNX + YAMLs to `checkpoints/proteus/<release>/engineai_pm01`, writes `xrobot_to_pm01.json` / `bvh_lafan1_to_pm01.json`, builds the PICO hold, retargets four LAFAN1 clips. Flags `--sources-root`, `--proteus-root`, `--release` (default `wsq61`), `--skip-motions`. |
| `prepare_ros.sh` | Copies the **installed** `interface_protocol` msg/srv/action definitions from the robot (rsync, read-only) into `third_party/engineai_pm01/ros2` and builds them with colcon. Arg: `user@host` (default `user@192.168.0.163`). |
| `env.sh` | Source in the PC shell: ROS Humble + the built messages, `ROS_DOMAIN_ID`, Cyclone DDS bound to the NIC, HF offline flags. |
| `check_robot.py` | Read-only readiness check (subscribes only). Exit 0 when 24-joint feedback, IMU and `joint_bridge` are fresh, a compatible BEST_EFFORT command subscriber exists and no other command publisher does; else exit 2. `--duration` (default 5 s). |
| `start_proteus.sh` | Hardware file playback: sources `env.sh`, runs `check_robot` (skipped with `--help`), then `tracking --robot pm01 --robot-io inline --controller keyboard --keyboard-layout et1`. Optional first argument is the motion NPZ. |
| `start_pico.sh` | PICO teleop: `publisher`, `sim` (`--sim-dt 0.000625 --decimation 32`), `policy-sim [ip]`, `robot [ip]` (sources `env.sh` and runs `check_robot` first). |
| `make_wave_motion.py` | Writes the synthetic arm-wave clip (stand, raise one arm, wave with the elbow, lower, stand) and checks soft joint limits and arm-to-body clearance. Flags `--output` (default `checkpoints/engineai_pm01/wave`), `--arm {right,left}`, `--duration` (10 s), `--freq` (1 Hz), `--ramp` (1.5 s), `--hold` (1 s), `--min-clearance` (0.05 m). |

`scripts/pm01/` — onboard deployment (motion computer, Nezha: Ubuntu 22.04, x86_64, CPython 3.10, ROS Humble):

| File | Purpose |
| --- | --- |
| `deploy_onboard.sh` | Run on the PC. Resolves offline x86_64 / CPython 3.10 packages from `uv.lock` (CPU-only torch, cached in `outputs/pm01-onboard-env`), rsyncs code, the policy export, the walk and wave motions, the prepared model and two Hugging Face caches to `<target>/{repo,site-packages,hf,logs}` on the robot, then runs an import check there. Never starts, stops or reconfigures robot software. `--build-only` stops before contacting the robot. |
| `env_onboard.sh` | Source on the robot: the robot's own ROS Humble and installed `interface_protocol`, domain 69, Cyclone DDS, the shipped packages on `PYTHONPATH`, offline HF (`HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`), one ONNX Runtime / BLAS thread. |
| `start_proteus_onboard.sh` | Onboard counterpart of `start_proteus.sh`, run in an interactive SSH terminal. First argument `wave` (default), `walk` or a motion `.npz`; further arguments go to `tracking`. Requires `reference_forcing` in the policy, takes a lock under `logs/`, refuses to start beside another PM01 controller, runs `check_robot`, sets the gain scales and callback pacing, pins Tracking with `taskset`, and records to `logs/run-<time>/` (`check_robot.json`, `tracking.npz`). |
| `watch_sway.py` | Read-only sway monitor (subscribes to IMU, joint state and motion state; publishes nothing). Every 0.5 s prints pelvis pitch/roll, the 1-s oscillating pitch/roll rate (deg/s rms), the fastest joint, both ankle-pitch speeds and the firmware mode. `--seconds` (default 120). |

`scripts/pm01/` — investigation tools (experimental; not a deployment path):

| File | Purpose |
| --- | --- |
| `shadow_policy.py` | Runs real PM01 feedback through the full Tracker/ONNX path for 775 frames (15.5 s) without any command publisher, and reports step timing against the 20 ms period (`policy_step_ms_p50`, `policy_step_ms_p99`, `policy_step_ms_max`, `steps_over_20ms`). Flags `--policy-config`, `--motion`, `--output`. |
| `trial_policy.py` | Bounded PC client: publishes candidate commands only to `/sim2real/pm01/request`; needs the onboard relay. `--walk-once` requires new `--policy-signal` / `--play-signal` files. |
| `policy_bridge.py` | Onboard relay (motion computer): the only hardware writer during trials, with an independent supervisor that latches classic damping. Sends **RL_PARSER** for active commands. |
| `bench_damping.py` | Onboard ankle-damping bench; read-only by default, `--execute` allows ≤1 s of low ankle damping. Requires root and an exact motor-driver binary hash. Uses RL_PARSER for the trial. |
| `rehearse_trial.py` | No ROS: replays the relay gate in MuJoCo from a recorded hanging pose with a mocap-weld gantry. |

Related files elsewhere:

- `sim2real/config/robots/pm01.py` + `sim2real/config/robots/specs/engineai_pm01.yaml` (via `proteus_spec.py`).
- `sim2real/rl_policy/robot_io/pm01.py` — `PM01_SDK_JOINT_NAMES`, `PM01Codec` (classic parser, zero-stamp handling, ankle/arm gain scales), `PM01RobotIO` (`PM01_HAL_HZ` / `PM01_HAL_CPU` → `spin_hz` / `spin_cpu`).
- `sim2real/rl_policy/robot_io/engineai.py` (shared with T800) and `sim2real/rl_policy/robot_io/ros2_named.py` (callback-thread pacing).
- `sim2real/rl_policy/control_mode.py` + `sim2real/rl_policy/controllers/keyboard.py` — the `et1` keyboard layout.
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_pm01.json`.
- `scripts/proteus/` (`prepare_assets.py`, `retarget_lafan.py`, `evaluate.py`), `scripts/build_xrobot_profile.py`, `scripts/export_proteus_onnx.py`, `scripts/make_hold_motion.py`.
- Tests: `tests/test_pm01.py`, `tests/test_pm01_wave.py`, `tests/test_pm01_launch.py` (opt-in), `tests/test_pm01_policy_bridge.py`, `tests/test_pm01_bench.py`, `tests/test_keyboard_layout.py`, `tests/test_proteus_transport.py`, `tests/test_pico_retarget_profiles.py`.
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
5. Onboard assets (on the PC, before `deploy_onboard.sh`; the release `wsq61` was not available for this setup):
   ```bash
   python -m scripts.pm01.prepare --sources-root .. --release proteus_wamseq_62_full --skip-motions
   uv run --project venv/pico --no-sync python -m scripts.proteus.retarget_lafan --robot pm01 --duration 8 \
     --output checkpoints/engineai_pm01/motions --bvh /path/to/lafan1/walk1_subject1.bvh
   python -m scripts.pm01.make_wave_motion --output checkpoints/engineai_pm01/wave
   ```
   `deploy_onboard.sh` stops unless `checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.{yaml,onnx}`,
   `third_party/engineai_pm01/model/pm_v2_tracking.xml` and both `manifest.json` files (motions, wave) exist. The
   robot needs no ROS build from this repo: `env_onboard.sh` uses the firmware's installed messages.

## Adjustments

| Tunable | Current value | Where |
| --- | --- | --- |
| Ankle parser | `parallel_parser_type = 0` (classic, PD in the motor driver) for `start_proteus.sh`, `start_pico.sh robot` and `start_proteus_onboard.sh` | `PM01Codec` in `robot_io/pm01.py` (T800 keeps `1`) |
| Ankle gains | kp 37.701, kd 2.4916 (training values in the policy). Under classic they act motor-side: joint-side pitch ≈4.09×, roll ≈1.11× | `specs/engineai_pm01.yaml` / `policy.yaml` |
| Ankle-pitch gain scale | `PM01_ANKLE_PITCH_GAIN_SCALE`: multiplies commanded ankle-pitch kp **and** kd. Onboard default **0.306** (= 1.25 / 4.09, keeps joint-side stiffness within 0.8–2× of training even if 4.09 is off); allowed 0.1–1 | `start_proteus_onboard.sh` → `PM01Codec` |
| Arm kp scale | `PM01_ARM_KP_SCALE`: multiplies shoulder/elbow kp (`J13`–`J22`). Onboard default **0.7** (sim: moves the delay-induced whole-body oscillation from ≈20 ms to ≈30 ms observation-to-target delay; onboard delay ≈15–18 ms); allowed 0.5–1 | `start_proteus_onboard.sh` → `PM01Codec` |
| Ankle kd scale | `PM01_ANKLE_KD_SCALE`: multiplies ankle pitch and roll kd, after the pitch scale. Onboard default **1.5** (against rocking with lightly loaded feet; sim: wave rocking 8.8 → 0.4 deg/s at 80% support); allowed 1–3 | `start_proteus_onboard.sh` → `PM01Codec` |
| Gain-scale scope | Policy commands only; damping is untouched. The PC launchers set none of them, so the PC route sends the training gains (unset = 1) | `PM01Codec.write_command` |
| ROS callback pacing | `PM01_HAL_HZ` → `spin_hz`, `PM01_HAL_CPU` → `spin_cpu`. Onboard default **200 Hz on CPU 0**; empty disables either. Unset on the PC (unpaced). Each paced cycle runs at most one callback per subscription and timer, so the 500 Hz publish timer also runs at most once per cycle | `start_proteus_onboard.sh` → `PM01RobotIO` → `ros2_named.py` |
| Tracker CPU | `PM01_TRACKER_CPU` (default `3`; the Nezha boots with `isolcpus=1-3` and vendor real-time threads on CPUs 1–2; empty disables `taskset`) | `start_proteus_onboard.sh` |
| Keyboard layout | `--keyboard-layout et1` (both `start_proteus.sh` and `start_proteus_onboard.sh`): **a** shape, **a** again Hold, **s** play/pause, **space** or **x** stop. `default` (i, ], space, o) stays for other robots | `BasePolicyArgs.keyboard_layout` in `base_policy.py`; `control_mode.py` |
| Zero header stamps | Accepted; freshness uses local monotonic receive time. A zero after a nonzero stamp is rejected | `PM01Codec._new_stamp` |
| IMU topic / extrinsic | `/hardware/imu_info`, `imu_to_pelvis_wxyz: [1, 0, 0, 0]` (IMU site aligned with pelvis) | `robot_io_options` in `policy.yaml` |
| Publish rate / timeout | `publish_hz: 500`, `timeout_s: 0.25`; mode heartbeat 1 s | `robot_io_options`; `EngineAICodec._fresh` |
| Stop damping / step bound | damping 2.0, 0.2 rad per publish | `NamedJointCodec` defaults in `ros2_named.py` |
| Joint limits | Training limits; no vendor-limit overrides needed | `prepare.py` → `validation.json` |
| Model axis tolerance | `axis_atol=1e-4` (spec axes rounded to 4 decimals; measured max 8.76e-5) | `prepare.py` |
| Sim timestep | **0.000625 s**, decimation 32 | `start_pico.sh sim`; `scripts/proteus/evaluate.py` default for PM01 |
| DDS / network (PC) | `PM01_NETWORK_INTERFACE` (`eno1`), `PM01_ROS_DOMAIN_ID` (69), `PM01_ROS_SETUP`; existing `CYCLONEDDS_URI` wins | `env.sh` |
| DDS / paths (onboard) | `PM01_ROS_OVERLAY` (default `/apps/engineai_robotics/_install/share/interface_protocol/local_setup.bash`), `PM01_ROS_DOMAIN_ID` (69), `PM01_CYCLONEDDS_URI` (unset keeps the vendor default), `PROTEUS_ROOT` | `env_onboard.sh` |
| Deploy target | `PM01_HOST` (`user@192.168.0.163`), `PM01_TARGET` (`proteus_pm01`, relative to the remote home), `PM01_RELEASE` (`proteus_wamseq_62_full`), `UV`, `PM01_ENV_CACHE` | `deploy_onboard.sh` |
| Remote message path | `PM01_REMOTE_PROTOCOL` (default `/apps/engineai_robotics/_install/share/interface_protocol`) | `prepare_ros.sh` |
| Policy config | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | launchers |

## Usage

Offline evaluation and tests:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
source scripts/pm01/env.sh   # ROS loopback tests isolate themselves on localhost, domain 171
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py tests/test_pm01_wave.py -q
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

### Hardware run on the PC

```bash
# Terminal 1
bash scripts/pm01/start_proteus.sh checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

1. Feet free, press nothing for 10 s: the robot should stay as quiet as `passive`.
2. **a** shapes to walk frame 0 over 10 s (arms rise to near horizontal).
3. Lower the gantry until the feet carry the weight (rope slack, still able to catch a fall), then **a** again for
   policy Hold.
4. **s** plays. **space** or **x** damps; **Ctrl+C** exits.

### Onboard run on the motion computer

```bash
# On the PC, once: install an SSH key, then deploy (or --build-only to resolve packages without the robot)
ssh-copy-id user@192.168.0.163
bash scripts/pm01/deploy_onboard.sh
# Dry run in passive or pd_stand (read-only, no command publisher); check policy_step_ms_p99 and steps_over_20ms
ssh -t user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && \
  python3 -m scripts.pm01.shadow_policy --policy-config checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml \
  --motion checkpoints/engineai_pm01/wave/motions/wave_right.npz --output ../logs/shadow-wave.npz'
# In joint_bridge, feet free on the gantry: the wave (or `walk`, or a motion .npz)
ssh -t user@192.168.0.163 proteus_pm01/repo/scripts/pm01/start_proteus_onboard.sh wave
# Second terminal, read-only pelvis monitor
ssh user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && python3 -m scripts.pm01.watch_sway'
```

1. Continue past the dry run only with `policy_step_ms_p99` well under 20 and `steps_over_20ms` near 0; otherwise
   move Tracking with `PM01_TRACKER_CPU`. The launcher prints the gain scales and callback pacing it uses, and stops
   unless `check_robot` passes.
2. Press nothing for 10 s (quiet as `passive`), then **a** shapes to the first frame over 10 s.
3. Lower the gantry until both feet are flat and carry the weight (rope slack, still able to catch a fall), then
   **a** again for policy Hold. Hold about 10 s and watch `watch_sway`: if the pelvis rocks, press **space**.
4. **s** plays or pauses. **space** or **x** stops (damping); **Ctrl+C** exits. The walk (`walk`) moves about 3 m
   forward: the gantry must follow and the path must be clear.
5. Logs: `~/proteus_pm01/logs/run-*/` (`check_robot.json`, `tracking.npz`).

### PICO teleop (publisher on the same PC)

```bash
# Terminal 1: XRoboToolkit PC Service + calibrated body/leg trackers
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: robot policy (or `sim` + `policy-sim 127.0.0.1` for MuJoCo)
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

Buttons: **A** shapes to neutral; release, then **A+B** enables Hold; **X** pauses/resumes; **B** alone damps.
Ports 28701 (motion) / 5592 (buttons); one robot per port set.

Shutdown (PC or onboard): **space** or **x** (PICO: **B**), keep the gantry loaded, Ctrl+C (damping repeats on
close), then **LB + RB** to `passive`.

### Investigation tools

Exact tested gains, firmware hash and hanging setup required; read the tutorial first:

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
- 2026-11-13 onboard: unpaced, the ROS callback thread decoded the firmware's joint/IMU streams nonstop on the policy
  CPU and stretched a step to 22–28 ms in `joint_bridge`. A dry run outside `joint_bridge` misses this (nothing is
  published there). Keep `PM01_HAL_HZ` / `PM01_HAL_CPU` set; even paced, ~10% of steps run slightly over 20 ms
  (GIL contention).
- 2026-11-13 onboard: with a taut rope carrying most of the weight (knee torque only 5–7 Nm), Hold and the wave
  rocked the pelvis at 4.3–4.7 Hz (pitch rate 15–40 deg/s rms in bursts). Let the feet carry the weight before
  Hold, and never enter Hold with the feet hanging free (the policy reads it as a fall). The ankle kd ×1.5 fix and
  all three gain scales are simulation-derived and the 4.09 classic-parser factor is an estimate.
- Zero-stamp feedback cannot distinguish a republished old value from a new one. `check_robot` `ready: true` does
  not prove motors are enabled (24 disable flags were once set while `joint_bridge` was reported).
- Process death or power loss needs firmware-side protection (PC and onboard); the bridge runner's command-age
  timeout is unknown.

## Detailed docs

- English: [`docs/tutorials/proteus-pm01.md`](../../docs/tutorials/proteus-pm01.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md)
- Vendor: [engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace), [PM01 parallel-ankle notes](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
