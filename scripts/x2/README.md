# AgiBot Lingxi X2 launchers

`scripts/x2/` runs the Proteus WAMSeq Tracker on the AgiBot Lingxi X2 Ultra (Proteus
contract `lingxi_x2`, public v1.3 model) through the AimDK ROS 2 HAL. It also stages the
model, exports the policy, converts / retargets references, records sim2sim videos, and
provides read-only feedback and system-mode preflight tools. The generic workflow
(Shaping → Hold → Go → Final Hold, E-stop, new-robot checklist) is in
[`../README.md`](../README.md); this page covers only what is specific to X2.
[中文](README_zh.md)

## Status

| Item | Value |
| --- | --- |
| Hardware | **Control untested.** Wired read-only feedback and a read-only system-state query worked on one Ultra (2026-11-04); its `Develop_MC` migration was rejected as not configured, so no takeover was done |
| `--robot` aliases | `x2`, `lingxi_x2`, `agibot_x2` (`sim2real/config/robots/__init__.py`) |
| Joints | 31 (12 leg, 3 waist, 14 arm, 2 head) |
| Robot I/O | `--robot-io inline` → `X2RobotIO`: ROS 2 Humble + AimDK messages, `/aima/hal/joint/{leg,waist,arm,head}/{state,command}` and `/aima/hal/imu/torso/state` |
| DDS | HAL launcher: CycloneDDS bound to `10.0.1.2`. System-state helper: Fast DDS via `fastdds_host.xml` |
| Where the Tracker runs | Host PC on the robot's wired network (ROS 2 Humble + AimDK installed) |
| Default checkpoint | `checkpoints/proteus/wsq61/lingxi_x2/` (all launchers, `check_feedback.py`, `render_sim2sim.py`; `export_policy.py --release` defaults to `wsq61`) |
| Policy YAMLs | `policy.yaml`: hardware, `reference_forcing: true`, with `robot_io_options` and `soft_joint_pos_limits`. `policy-sim.yaml`: measured-root simulation, `reference_forcing: false`, no soft limits |
| Earlier results | Sim2sim tables and the 2026-11-03 runtime audit used `proteus_wamseq_62`; they do not validate wsq61 |

## Files

| File | Purpose |
| --- | --- |
| `start_proteus.sh` | Hardware run of one any4hdmi dataset with the keyboard controller (inline AimDK I/O) |
| `start_first_hold.sh` | First hardware Hold: requires `Develop_MC`, runs `check_feedback.py` with ownership / limit checks, then tracks the static `outputs/x2/pico_hold` with recording |
| `start_pico.sh` | PICO teleop: `publisher`, `sim`, `policy-sim` or `robot` mode |
| `check_feedback.py` | Subscribe-only HAL feedback check (rates, names, freshness, IMU, positions); optional `--require-exclusive-control`, `--require-joint-limits` |
| `system_state.sh` / `system_state.py` | Read-only system-state query; `--target Develop_MC` / `--target Ready` migrate; `--require-develop-mc` read-only gate |
| `fastdds_host.xml` | Fast DDS profile whitelisting host interface `10.0.1.2` |
| `prepare_assets.py` | Stages vendor `X2_URDF-v1.3.0/x2_ultra.xml` as `third_party/lingxi_x2/model/scene_tracking.xml` with training limits / armature, writes `validation.json` |
| `export_policy.py` | Exports the actor plus `policy-sim.yaml` (`--release`, default `wsq61`) |
| `convert_reference.py` | sonic-x2 PKL (`relaxed_walk`, `gangam_dance`, `idle_stand`) → any4hdmi |
| `retarget_lafan.py` | LAFAN BVH → X2 with the bundled calibration and `X2Retargeter` |
| `render_sim2sim.py` | X2 wrapper around `scripts/oli/render_sim2sim.py` (defaults to wsq61 `policy-sim.yaml`) |
| `summarize_sim2sim.py` | Metrics from a rendered video's `.trajectory.npz` / `.root.npz` |
| `build_pico_profile.py` | Regenerates `sim2real/teleop/configs/xrobot_to_x2.json` from TaiE64/AgibotX2-Retargeter |

Related files elsewhere:

- Robot config: `sim2real/config/robots/x2.py` (`X2_CFG`, `X2_AIMDK_GROUPS`, `X2_VENDOR_LIMITS`, `x2_deploy_joint_limits`).
- Robot I/O: `sim2real/rl_policy/robot_io/x2.py` (`X2StateCommand` codec, `X2RobotIO`); dispatch in `robot_io/factory.py`.
- Policy YAML generation: `scripts/export_proteus_onnx.py` (`robot_io_options`, `soft_joint_pos_limits` for X2).
- Simulation model tweaks: `sim2real/sim_env/utils/mjcf.py`.
- Teleop: `sim2real/teleop/x2_retarget.py`, `sim2real/teleop/pico_retarget_pub.py`,
  `sim2real/teleop/configs/{xrobot_to_x2.json,bvh_lafan1_to_x2.json,X2-RETARGET-LICENSE}`.
- Shared helpers: `scripts/make_hold_motion.py`, `scripts/lafan_tracking_suite.py`, `scripts/summarize_proteus_sim2sim.py`.
- Tests: `tests/test_x2.py`, `tests/test_pico_retarget_profiles.py`.
- Records: `docs/experiments/proteus-x2-runtime-audit.json`, `docs/experiments/proteus-wsq61.json`.

## Setup

Run from the repository root. Tracker, simulator and tools use the root `uv` project
(`uv run --no-sync`); the PICO publisher uses `venv/pico`.

1. External checkouts: Proteus `wamseq` worktree (`../Proteus-wamseq`),
   `AgibotTech/agibot_x2_urdf` (`../agibot_x2_urdf`); for references `meetsitaram/sonic-x2`
   (`../sonic-x2-reference`); for LAFAN `limxdynamics/GMR` (`../GMR-limx`).
2. Stage the model and export the default (wsq61) policy; both are Git-ignored:

   ```bash
   uv run --no-sync python scripts/x2/prepare_assets.py \
     --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/x2/export_policy.py --proteus-root ../Proteus-wamseq
   ```

   `--release proteus_wamseq_62` writes `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/` instead.
3. Host: ROS 2 Humble and the AimDK message package that matches the robot firmware. The
   launchers that source it themselves default to `third_party/aimdk_x2_ros_v082/install/setup.bash`
   (override with `X2_AIMDK_SETUP`); `start_proteus.sh` and `start_pico.sh robot` expect
   you to source ROS 2 + AimDK in the shell first.
4. Network (observed on one Ultra, not a general requirement): tagged VLAN 200, host
   `10.0.1.2/24` on `eno1.200`, robot `10.0.1.40`. The host address is hard-coded in
   `fastdds_host.xml` and the `CYCLONEDDS_URI` of `start_first_hold.sh`; edit both for a
   different address. `ROS_DOMAIN_ID=0`.
5. Copy `checkpoints/proteus/wsq61/lingxi_x2/` (ONNX, YAMLs, contract JSON),
   `third_party/lingxi_x2/` and motion datasets to the host.
6. All launchers export `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`; set them in
   your shell for manual commands.
7. PICO / first Hold: create the neutral reference on the policy computer:
   `uv run --no-sync python scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold`.
   PICO publisher setup follows `docs/getting-started/teleop-x86-64.md`.

## Adjustments

| Tunable | Current value | Set in |
| --- | --- | --- |
| Training joint limits, kp/kd, armature, default pose | Proteus `lingxi_x2` spec, root height `0.6735` m | `sim2real/config/robots/x2.py` `_JOINTS` |
| Vendor (physical v1.3) limits | Narrower than training on `waist_yaw` (upper 2.2078), wrist pitch (±0.5236), wrist roll outer limit (1.5097 vs 1.571), `head_yaw` (±0.349) | `X2_VENDOR_LIMITS` |
| Hardware target clip | Intersection of training and vendor limits; never widened | `x2_deploy_joint_limits`, applied in `X2StateCommand.write_command` |
| Soft limits (Shaping clip) | `soft_joint_pos_limits` = same intersection, in `policy.yaml` only | `scripts/export_proteus_onnx.py` |
| `robot_io_options` | `publish_hz: 500`, `timeout_s: 0.25`, `head_pitch_enabled: false`, `imu_topic: /aima/hal/imu/torso/state`, `imu_to_pelvis_wxyz: [1,0,0,0]` | `policy.yaml` (generated by `scripts/export_proteus_onnx.py`) |
| IMU extrinsics | Identity matches the model; hardware must use the factory calibration | `imu_to_pelvis_wxyz` |
| Head pitch | Disabled: all command fields for `head_pitch_joint` sent as zero | `head_pitch_enabled` |
| Shaping target | `motion_first_frame` | `policy.yaml` / `policy-sim.yaml` |
| Preflight limit margin | 0.01 rad beyond the deploy limits | `check_feedback.py --require-joint-limits` |
| Simulation | Floor contacts only, zero passive damping, `sim_dt` 0.005 s, policy 50 Hz | `sim2real/sim_env/utils/mjcf.py`, `IntegratedSim2SimArgs` |
| Teleop joint speed bound | 12 rad/s between consecutive targets | `X2Retargeter.max_joint_speed` |

## Usage

### Sim2sim rehearsal

```bash
uv run --no-sync python scripts/x2/convert_reference.py --clip gangam_dance --output outputs/x2/gangam_dance
uv run --no-sync python scripts/x2/render_sim2sim.py --motion-path outputs/x2/gangam_dance \
  --duration 11.2 --output outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/lafan_tracking_suite.py --robot x2 \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
```

### Hardware preflight and takeover

```bash
# ROS env as in start_first_hold.sh (CycloneDDS on 10.0.1.2), then:
uv run --no-sync python scripts/x2/check_feedback.py --duration 10 --output outputs/x2/feedback.json
bash scripts/x2/system_state.sh                       # read-only query
bash scripts/x2/system_state.sh --target Develop_MC  # only if the robot configures this mode
bash scripts/x2/start_first_hold.sh                  # static Hold, records outputs/x2/first_hold_<time>.npz
bash scripts/x2/start_proteus.sh /path/to/x2/any4hdmi_dataset [policy.yaml]
```

Keyboard: `i` Shaping to frame 0 over 10 s, `]` policy Hold, Space play / pause,
`o` damping, Ctrl+C exit. Tracking sends damping from start-up, before any key.

### PICO teleop

```bash
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70   # venv/pico
bash scripts/x2/start_pico.sh sim
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
bash scripts/x2/start_pico.sh robot PUBLISHER_IP                      # hardware, replaces the last two
```

Buttons: **A** Shaping to neutral (10 s), **A+B** policy Hold, **X** start / pause live
reference, **B** alone damping. Open TCP 28701 and 5592; OLI and X2 PICO defaults use the
same ports, so run only one at a time per host.

### Shutdown

Stop Tracking (`o`, then Ctrl+C); the backend publishes damping five times on orderly
exit. Keep support, then restore native control with
`bash scripts/x2/system_state.sh --target Ready` (may continue to Business).

## Safety notes & known issues

- **Hardware control is untested.** Matching v0.8.2 feedback does not confirm firmware
  version or provide motor-fault monitoring (no `error_code` field in that layout).
- On the tested robot `Develop_MC` is not configured; do not guess mode names or start
  Tracking while native MC owns the HAL.
- The preflight checks are start-up checks, not an ownership lock.
- The HAL has no command watchdog: process kill or power loss leaves the last command.
  Keep gantry support and an independent stop.
- Verify IMU axes against physical pelvis tilt and encoder zeros; do not widen limits to
  bypass refusal. Other X2 revisions (e.g. v1.4 model) are not covered.
- `policy-sim.yaml` needs world localization; do not use it on hardware.
- PICO body-stream loss holds the last reference; it does not select damping.

## Detailed docs

- English: [`docs/tutorials/proteus-x2.md`](../../docs/tutorials/proteus-x2.md)
- 中文: [`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md)
- Generic workflow: [`../README.md`](../README.md)
