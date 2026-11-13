# Primebot Q1 launchers

Launchers and tools for running the Proteus WAMSeq tracking policy on the
Primebot Q1 (`primebot_q1` contract, 22 policy joints). The Tracker talks to the
Q1 HAL over ROS 2 (AimDK messages) through the inline robot I/O; the same policy
also runs in MuJoCo sim2sim and with PICO teleop. This page covers what is
specific to Q1. For the shared workflow (Shaping → Hold → Go → Final Hold,
E-stop, new-robot checklist) see [`../README.md`](../README.md).

## Status

| Item | Q1 |
| --- | --- |
| Hardware status | Software, MuJoCo and read-only physical feedback tested. The first hardware Shaping trial crossed the legs; the command-order fix has only offline coverage. Policy takeover and live headset use are **not validated**. |
| `--robot` aliases | `q1`, `primebot_q1` (`sim2real/config/robots/__init__.py`) |
| Joints | 22 policy joints, single wire group `body` |
| Robot I/O | `--robot-io inline` → `Q1RobotIO` (`robot_io/q1.py`) on `NamedRosRobotIO` (`robot_io/ros2_named.py`); `aimdk_msgs` `JointStateArray` / `JointCommandArray` on `/aima/hal/joint/state` / `/aima/hal/joint/command`, `sensor_msgs/Imu` on `/aima/hal/imu/state` |
| Where the Tracker runs | A host PC on the robot's ROS 2 network (the tutorial's "inspected PC", wired DDS). No onboard deployment is documented. |
| Default checkpoint | `wsq61` (`wsq61_wam_61robots_30000.pt`): `checkpoints/proteus/wsq61/primebot_q1/` |
| Policy YAMLs | `policy-sim.yaml` (sim2sim, PICO sim), `policy.yaml` (hardware base), `policy-pc.yaml` (hardware overlay from `prepare_deployment.py`, used by `start_proteus.sh` and `start_pico.sh robot`) |

## Files

| File | Purpose |
| --- | --- |
| `prepare.py` | Builds the vendor model, exports the WAMSeq ONNX (`--release`, default `wsq61`), writes PICO retarget profiles, the `pico_hold` dataset and three LAFAN1 motions. |
| `prepare_deployment.py` | Writes the hardware overlay (`policy-pc.yaml` by default) next to `policy.yaml`: IMU topic/extrinsic, joint limits, limit mode, entry gates. |
| `start_proteus.sh` | Hardware file playback: inline I/O, keyboard controller, NPZ motion, `policy-pc.yaml`. |
| `start_pico.sh` | PICO teleop: `publisher`, `sim`, `policy-sim`, `robot`. |
| `validate_deployment.py` | Offline rehearsal (no ROS): real Tracker, codec and serializer against MuJoCo with a gantry weld. |
| `commission_joints.py` | Disarmed-by-default measured-pose hold and one-joint ±0.03 rad probe via the production transport; no policy. |
| `inspect_imu.py` | Read-only IMU subscriber; prints assumed pelvis RPY, gyro and gravity mismatch. |

Related files elsewhere:

- Robot config: `sim2real/config/robots/q1.py` (from `sim2real/config/robots/specs/primebot_q1.yaml`).
- Robot I/O: `sim2real/rl_policy/robot_io/q1.py`, `sim2real/rl_policy/robot_io/ros2_named.py`, dispatch in `sim2real/rl_policy/robot_io/factory.py`.
- Target post-processing: `sim2real/rl_policy/utils/joint_targets.py`.
- Policy YAML export (Q1 `robot_io_options`): `scripts/export_proteus_onnx.py`.
- Teleop profiles: `sim2real/teleop/configs/xrobot_to_q1.json`, `bvh_lafan1_to_q1.json`.
- Shared Proteus tools: `scripts/proteus/` (`evaluate.py`, `prepare_assets.py`, `build_retarget_profiles.py`, `retarget_lafan.py`), `scripts/make_hold_motion.py`.
- Tests: `tests/test_q1_commission.py`, Q1 cases in `tests/test_proteus_transport.py`, `tests/test_proteus_sim_resolution.py`, `tests/test_tracker_action_postprocess.py`, `tests/test_pico_retarget_profiles.py`.
- Experiment record: `docs/experiments/proteus-q1.json`.

## Setup

- **ROS 2 + vendor messages.** Source ROS 2 Humble and a vendor workspace that
  provides `aimdk_msgs` ([primebot-q1-sdk](https://github.com/PrimebotTech/primebot-q1-sdk)).
  The Python environment does not install robot SDKs.
  ```bash
  source /opt/ros/humble/setup.bash
  source /path/to/vendor_workspace/install/setup.bash
  ```
- **Network.** The host must see the Q1 HAL topics over DDS. On the inspected PC
  the wired DDS settings live in a local `outputs/q1/source_env.sh` (not in git).
- **Vendor control mode.** Put the vendor controller in Advanced / low-level
  development mode and stop other command publishers.
- **Environments.** Policy, sim and tools use the root `uv` project; PICO IK and
  the offline retargeting use `venv/pico`. The launchers pick the right one.
- **HF offline.** Export `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`. Both
  launchers already set them.
- **Runtime artifacts (not in git).** ONNX, vendor assets, motions and the hold
  dataset are rebuilt locally. Source checkouts expected under `--sources-root`:
  `Proteus-wamseq` (Proteus `wamseq` worktree), `primebot-q1-urdf`, `GMR`,
  `ubisoft-laforge-animation-dataset`. Then:
  ```bash
  uv run --no-sync python -m scripts.q1.prepare --sources-root /path/to/projects
  # --release <name>, --proteus-root <dir>, --skip-motions are optional
  ```
  Outputs: `checkpoints/proteus/<release>/primebot_q1/`,
  `checkpoints/primebot_q1/motions/`, `checkpoints/primebot_q1/pico_hold/`.

## Adjustments

Hardware tunables live in `policy-pc.yaml` (written by `prepare_deployment.py`)
and in `Q1RobotIO` / `NamedJointCodec` defaults.

| Setting | Current value | Where |
| --- | --- | --- |
| `policy_limit_mode` | `clip` (default); `pass`, `feedforward` experimental | `prepare_deployment.py --policy-limit-mode` → `robot_io_options` |
| `limit_feedforward_band_rad` | `0.1` (feedforward fades to zero over the last 0.1 rad) | `NamedJointCodec` default |
| Joint limits | `clip` → `soft_joint_pos_limits` (Tracker clips policy and Shaping targets); `pass`/`feedforward` → `shaping_joint_pos_limits` (Shaping only) | `prepare_deployment.py` |
| `max_target_velocity_rad_s` | `15.0` | `prepare_deployment.py` |
| Entry gates | guarded: `policy_entry_max_joint_error_rad 0.10`, `policy_entry_max_joint_velocity_rad_s 0.5`; `--manual-policy-entry` removes them | `prepare_deployment.py` |
| `shaping_target` / `reference_entry_s` | `motion_first_frame` / `0.0` | `prepare_deployment.py` |
| Shaping gravity feedforward | off; `--enable-shaping-gravity` sets 10% of effort limits, 2 s ramp | `prepare_deployment.py` |
| `allow_init_from_measured` | `True`: Shaping returns gradually from a measured pose outside the model range | `Q1RobotIO` |
| `start_disarmed` | `True`: no motor command (even on exit) before the first `i` / `]` | `Q1RobotIO` |
| `preserve_feedback_order` | `True`: every command field is sent in the HAL feedback slot order | `Q1RobotIO` |
| IMU | topic `/aima/hal/imu/state`, `imu_to_pelvis_wxyz: [1, 0, 0, 0]` | overlay args; `SIM2REAL_Q1_IMU_TOPIC` is the fallback when `imu_topic` is unset |
| `publish_hz` | `1000` (policy at 50 Hz) | `Q1RobotIO`, `export_proteus_onnx.py` |
| `timeout_s` | `0.25` (stale feedback/command → damping) | `export_proteus_onnx.py` |
| Damping | kd `5`, kp `0` in zero mode, timeout or error | `Q1RobotIO(damping=5.)` |
| `max_position_step` | `0.2` rad per publish | `Q1RobotIO` |
| Sim timestep | `0.000625` s (`evaluate.py` default for Q1; `start_pico.sh sim` uses `--decimation 32`) | `scripts/proteus/evaluate.py`, `start_pico.sh` |

Notes:

- **Limit modes.** Shaping targets are always clipped. `pass` sends policy targets
  unchanged; on Q1 use it only with motions that start with the arms down (a
  walk1 T-pose start swung the right arm to -2.28 rad on 2026-11-06; shoulder
  roll has no hard stop at the model limit). `feedforward` adds effort
  kp·(target − bound); earlier evidence suggests the Q1 hardware path ignores
  `effort`, in which case it behaves like `clip`.
- **IMU.** The vendor V3 MJCF places `imu_0` on `pelvis` with no rotation, hence
  identity. Whether the live ROS IMU frame matches the model frame is still to be
  verified on hardware. Do not cancel the robot's current lean to fake an
  upright reading.

## Usage

**Sim2sim evaluation** (MuJoCo, 0.000625 s default):

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot q1 \
  --policy-config checkpoints/proteus/wsq61/primebot_q1/policy-sim.yaml \
  --dataset checkpoints/primebot_q1/motions --output outputs/q1/eval-new
```

**Prepare the hardware overlay** (writes `policy-pc.yaml`; add
`--policy-limit-mode pass --output-name policy-pc-pass.yaml` for a variant):

```bash
uv run --no-sync python -m scripts.q1.prepare_deployment \
  --imu-topic /aima/hal/imu/state --imu-to-pelvis-wxyz 1 0 0 0 --manual-policy-entry
```

**Offline rehearsal** (no ROS; exits non-zero on failure; writes `outputs/q1/deployment-rehearsal.{json,npz}`):

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.q1.validate_deployment \
  --policy-config checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml
# Optional: --initial-feedback <ros-readonly.json> | --initial-tracking-record <record.npz>
#           --no-use-recorded-orientation
```

**Joint commissioning** (gantry carries the weight, legs uncrossed, vendor stop at hand):

```bash
uv run --no-sync python -m scripts.q1.commission_joints --joint left_elbow_joint
uv run --no-sync python -m scripts.q1.commission_joints --read-only --duration-s 15
```

Starts disarmed. `h` captures and holds the measured pose (kp ramps 0→5 over
2 s, kd 2); `j` runs one `+0.03 → 0 → -0.03 → 0` rad cycle on the selected joint
over 8 s; `o` / Space / `q` / Esc / Ctrl+C damp and exit. Stops on drift
> 0.10 rad, speed > 0.5 rad/s, stale feedback, a second command publisher or a
changed wire order; armed deadline 25 s. Logs go to `outputs/q1/commission/`. No
automatic PASS: inspect physical identity and direction before the next joint.

**IMU inspection** (subscriber only, no commands):

```bash
uv run --no-sync python -m scripts.q1.inspect_imu --duration-s 30
# --topic, --imu-to-pelvis-wxyz, --output <file.json>
```

**Hardware run** (file playback, keyboard):

```bash
bash scripts/q1/start_proteus.sh checkpoints/primebot_q1/motions/motions/walk1_subject1.npz
# PROTEUS_POLICY_CONFIG=<yaml> overrides policy-pc.yaml
```

Keys: `i` Shaping (10 s, optional with manual entry); `]` enters policy Hold;
Space plays/pauses the reference (not a stop); `o` damping; Ctrl+C exits.
Hold is active policy control: limbs can move while playback is paused.

**PICO teleop:**

```bash
bash scripts/q1/start_pico.sh publisher --actual-human-height 1.8   # terminal 1
bash scripts/q1/start_pico.sh sim                                   # terminal 2
bash scripts/q1/start_pico.sh policy-sim 127.0.0.1                  # terminal 3
# Hardware instead of terminal 3 (uses policy-pc.yaml; PROTEUS_POLICY_CONFIG=<yaml> overrides):
bash scripts/q1/start_pico.sh robot PUBLISHER_IP
```

Motion on TCP 28701, buttons on TCP 5592. **A** Shaping to neutral; **A+B**
policy Hold; **X** pauses/resumes the live reference; **B** alone damps. Stream
loss holds the last reference; it does not damp. The `robot` mode uses the hardware
overlay `policy-pc.yaml` (it carries the IMU topic and extrinsic) and exits with a
hint if that file has not been prepared.

**Shutdown.** `o` (or PICO **B**) damps; Ctrl+C exits. Once armed, a normal exit
sends damping several times. These are process-level checks, not firmware
protection after a kill or power loss; keep the vendor stop accessible.

## Safety notes and known issues

- Software, MuJoCo and read-only physical feedback are tested. A read-only 501-tick
  wsq61 probe on 2026-11-04 saw all 22 joint names and no stale feedback.
- **Crossed legs (2026-11-04).** The first `i`-only Shaping trial crossed the
  legs: the adapter sent legs-first order while the HAL feedback is arms-first
  and appears to be consumed by slot. `preserve_feedback_order` fixes this in
  software and has offline coverage, but has **not** passed another hardware
  trial. Next step is `commission_joints.py`, not full-body Shaping.
- Policy takeover and live headset operation remain unvalidated.
- Sustained shoulder sag was seen at low kp; the gravity feedforward did not
  appear in reported efforts.
- `validate_deployment.py` idealizes the gantry as a weld; passing it does not
  certify hardware release.
- Developer mode and an empty publisher list do not prove the vendor controller
  has released ownership.

## Detailed docs

- [Proteus WAMSeq on Primebot Q1](../../docs/tutorials/proteus-q1.md)
  ([中文](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-q1.md))
- Generic workflow: [`../README.md`](../README.md)
