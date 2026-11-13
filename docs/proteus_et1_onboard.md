# Proteus on the Galbot ET1 board

The ET1 runs Proteus actors (the 60-robot actor, or WAM-62) and Galbot's own et1-dep tracking policies on its own board (aarch64, CPython 3.10): an HDEP bridge at 200 Hz on CPU 7 and CPU ONNX Tracking at 50 Hz on CPUs 5–6 (A76 cores; cores 5 and 6 run the WAM actor in 9.7 ms, p99 10.5; core 7 takes 14 ms and core 4 shares interrupts; an A55 core needs 12–22 ms). Galbot's Loco and its gamepad programs must not run at the same time; the launcher refuses to start next to them.

## Model, configuration and motions

`sim2real/config/robots/et1.py` (`galbot_et1`, alias `et1`) follows the et1-dep MJCF: joint order and ranges, including waist yaw ±0.785 rather than the ±π/2 of the Proteus training spec. Armature and gains match Proteus and Galbot's own trackers. The MJCF is not vendored; set `ET1_DEP_ROOT` (default `~/et1-dep`). Its kinematics match the Proteus training MJCF to 1e-7 m on all shared bodies.

```bash
python scripts/export_proteus_onnx.py --contract galbot_et1 --robot galbot_et1   # max |d| = 0 over 64 inputs
python scripts/g1/prepare_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml          # policy-pc.yaml
python scripts/prepare_proteus_test_motions.py --robot galbot_et1                # Proteus ScaleBridge clips
python scripts/et1/slow_motion_tail.py \
  --source checkpoints/proteus/proteus_60/galbot_et1/motions/11_stand_walk_straight \
  --output checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close
```

`policy-pc.yaml` locks the head at 0 with a zero-width soft limit: the actor tracks no head body and otherwise turns the head to about 1 rad yaw. Of the Proteus test walks only `10_walk_b` and `11`–`14` walk forward (`07`–`09` walk backwards relative to the pelvis), and only `11`–`14` start from standing. At source speed the ET1 squats while closing its feet at the end of `11_stand_walk_straight`; `11_slow_close` slows the motion after 10.4 s by 2x and ends still, so the robot stops standing (simulation: 3.66 m, worst upright 0.941, 0.01 m drift while holding).

Rehearse in MuJoCo with the same Tracking, BridgeSafety and A → A → Start flow:

```bash
python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
  --motion-path checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close --walking-s 20
```

The error metric compares bodies against the yaw-aligned reference the policy sees. A clip shorter than the schedule ends in Hold before the scripted pause, so `passed` can be false on the mode sequence alone; read the stability fields.

## Onboard stack

- `scripts/et1/hdep_shim.cpp`: a C ABI over HDEP for ctypes, because `hdep_py` exists only for x86_64 / CPython 3.11. Built on the board.
- `scripts/et1/hdep_bridge.py`: HDEP ⇄ the ZMQ LowState/LowCmd contract of the G1 bridge, arbitrated by `BridgeSafety`. Startup only reads. A ready Tracker and a new A press call `enter_debug_mode()` (0.11–0.86 s on hardware), which blocks the bridge loop. The bridge then drops the commands queued during the block (they are as old as the block and used to latch the takeover as expired), keeps publishing state and waits up to 0.5 s for the Tracker's next command. Select, Galbot's L1 + Down, timeouts and invalid commands latch.
- `sim2real/rl_policy/controllers/bridge_joystick.py`: `--controller bridge_joystick` reads the handset the bridge republishes (Unitree key layout).
- `scripts/et1/operator_keyboard.py`, `operator.sh`, `press_key.py`: keyboard or scripted keys without a handset. Keys are merged per sender and expire after 0.3 s, so a dropped SSH session releases them; it cannot stop the robot, keep the gantry and the hardware stop ready.
- `scripts/et1/full_state_probe.cpp`: read-only dump of all joints, IMU, gamepad and robot state.

The Tracker freezes the objects created during setup (`gc.freeze()`) before its control loop. Motion datasets and FK caches leave about 350 000 of them; a full garbage collection over them took 442 ms on the board, stalled a step past the 0.2 s command timeout and latched damping. After freezing, a full collection takes about 1 ms.

`BridgeSafety` accepts shaping (`init`) targets between the measured position and the limits, so A brings a robot that rests past a model limit back (the ET1 rested at waist yaw 1.32 rad and left wrist roll −1.93 rad); it never commands further out. Policy targets keep the plain limits.

## Deploy and run

From the PC (uses `uv` to resolve aarch64 wheels once, then rsync; touches only the target directory):

```bash
ET1_HOST=192.168.121.42 ET1_TARGET=/root/proteus_et1_claude scripts/et1/deploy_onboard.sh
```

On the board, in one SSH terminal (`PROTEUS_READ_ONLY=1` for a dry run that never enters debug mode or writes motors):

```bash
ssh -t 192.168.121.42 'cd /root/proteus_et1_claude/repo && \
  PROTEUS_MOTION=checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close bash scripts/et1/start_proteus_onboard.sh'
```

and the keyboard in a second one:

```bash
ssh -t 192.168.121.42 /root/proteus_et1_claude/repo/scripts/et1/operator.sh
```

Wait for `Tracker and LowState ready`. On the gantry: `a` shapes to the standing pose over 10 s; `a` again interpolates to the reference start over 2 s and enters Hold; after `Hold ready`, lower the gantry; `s` plays the motion, which ends standing. Space (Select) latches the stop and drops active support. Ctrl-C in the controller terminal stops both processes; the bridge sends damping on exit, so hold the robot on the gantry first.

`PROTEUS_POLICY_DIR` selects the actor: `checkpoints/proteus/proteus_60/galbot_et1` (default), `checkpoints/proteus/proteus_wamseq_62/galbot_et1`, or a Galbot tracking export (below). `PROTEUS_PLAYBACK_LIMIT_S` stops the reference clock early and holds that frame.

## Galbot et1-dep tracking policies

`scripts/et1/prepare_galbot_track_policy.py` turns an et1-dep tracking checkpoint (policy protocol 3.2: `policy.toml`, `inference.toml`, `policy.onnx` with the observation normalizer, `motion.npz`) into a Tracker policy: `policy.yaml`, the deployment overlay `policy-pc.yaml` (5% inset limits, head locked at 0, 15 rad/s slew, 2 s reference entry) and the Motion as a qpos dataset. The 111-dim observation is implemented by `sim2real/rl_policy/observations/galbot_track.py`; its conventions (6D orientation flattening, absolute reference joints, joint positions relative to the policy default) were checked against the checkpoint's normalizer statistics.

```bash
python scripts/et1/prepare_galbot_track_policy.py \
  --checkpoint <addon>/storage/ckpt-shanshenbu --output checkpoints/galbot_track/galbot_et1/shanshenbu
python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 --scenario walk-hold \
  --policy-config checkpoints/galbot_track/galbot_et1/shanshenbu/policy-pc.yaml \
  --motion-path checkpoints/galbot_track/galbot_et1/shanshenbu/motions/shanshenbu \
  --npz-playback-limit-s 49.78 --min-forward-m -100 --max-lateral-error-m 100 --action-delay-ms 10
```

The `shanshenbu` dance (49.8 s) was trained without command delay. In simulation it completes at 0–10 ms delay and falls about 8 s in from 12 ms (from 10 ms with the training model's armature 0.01 and no joint friction); holding any frame after the motion starts walks off or falls, so only the opening stand (the first 2.5 s) can be cut. With no position feedback it drifts about 4 m over the dance in simulation (about 0.5 m in Galbot's training preview).

## Hardware status

- 2026-10-29: proteus_60 `01_idle`, then `11_slow_close`, without a bridge fault. Read-only dry runs measured a 20.07 ms median control period (p99 26 ms, max 28 ms).
- WAM-62 stood (`01_idle`) and started `11_slow_close` on hardware; one run latched on a 241 ms garbage-collection stall, which `gc.freeze()` now prevents.
- 2026-11-11, after the takeover and garbage-collection fixes: every takeover completed (`enter_debug_mode` 0.11–0.14 s), with no stall or timeout. proteus_60 `11_slow_close` played its 14.5 s to the end and held standing (worst upright 0.965 while walking, 0.993 while holding, −6.6° heading change).
- 2026-11-11, `shanshenbu`: the opening stand (2.5 s cut) played and held. Two full runs tracked the first 8 s, including the fast turn, then over-rotated by 10–37° per turn and tilted up to about 30°; the operator stopped them at 12.9 s and 12.4 s. A passive listener on the board measured the software chain at about 8 ms on average (state publish to command 0.8 ms median, plus waiting for the 200 Hz bridge and the 50 Hz Tracker periods), before the HDEP/DDS path to the drives, which is not measurable from software.
