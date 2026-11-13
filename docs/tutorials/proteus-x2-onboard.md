---
title: Proteus on AgiBot Lingxi X2 (onboard)
sidebar_position: 7
---

The Proteus policy runs on the X2 motion-control board (RK3588), operated from a PC over SSH. The flow is the same as
on the Galbot ET1: a bridge process owns the robot I/O and the stop arbitration, the Tracking process runs the policy,
and an operator keyboard in a second SSH session sends a, a, s and space.

**Verified on the robot** on 2026-11-12, on an X2 that reports itself as **X2 Basic**, firmware v0.8.2.16, on a
gantry, with `proteus_wamseq_62` and `policy-pc-damped.yaml`:

- takeover, shaping to the standing pose and the second `a` into Hold; the policy stood steadily for 23 s and 42 s
  (leg and waist joints 0.01–0.03 rad/s, pitch within 0.5–1.4°);
- `21_wave_right` played without arm or waist oscillation (arm speed 0.38 rad/s against 0.32 rad/s in sim2sim, pitch
  −0.1° to +1.3°);
- `15_walk_short` played in full (10 s) and stopped standing.

In all these runs the rope still carried most of the robot's weight (see "Load on the feet" below); walking with the
full weight on the feet has not been tested yet. The takeover and the motions were also rehearsed in MuJoCo with the
same bridge and Tracking processes (see "Rehearse on the PC").

## Network

The PC connects to the robot's rear Ethernet port. The robot's `10.0.1.0/24` network is **tagged VLAN 200**: ARP is
answered untagged, but untagged IP gets no reply. Create a VLAN interface with the PC at `10.0.1.2`:

```bash
nmcli connection add type vlan con-name x2-vlan200 ifname enp5s0.200 dev enp5s0 id 200 \
  ipv4.method manual ipv4.addresses 10.0.1.2/24 ipv6.method disabled
nmcli connection up x2-vlan200
```

If activation fails with "Failed to find a compatible device", set `vlan.parent` to the UUID of the connection active on
`enp5s0` (netplan rewrites the interface name). The motion-control board is `agi@10.0.1.40`. Install a key once with
`ssh-copy-id agi@10.0.1.40`.

## Policies and references

```bash
python scripts/export_proteus_onnx.py --contract lingxi_x2 --robot x2 \
  --checkpoint-dir ../Proteus-wamseq/ScaleTrack/checkpoints/proteus_wamseq_62 \
  --output-root checkpoints/proteus/proteus_wamseq_62
python scripts/g1/prepare_proteus_deployment.py \
  --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.yaml --robot x2
```

`policy-pc.yaml` shapes to the default standing pose, enters the reference over 2 s, keeps targets inside the training
∩ vendor joint range minus 5%, limits targets to 15 rad/s and locks the head. `proteus_60` and `proteus_60_bigdata`
are exported the same way (without `--checkpoint-dir`, or with `.../proteus_60_bigdata --output-root
checkpoints/proteus/proteus_60_bigdata`).

The MuJoCo model comes from `scripts/x2/prepare_assets.py` (vendor model `X2_URDF-v1.3.0`, AgibotTech/agibot_x2_urdf
`575cc6b`). The walking references are LAFAN1 `walk1_subject1.bvh` windows that start from standing, retargeted with
GMR and the calibrated X2 IK table; `--hold-s` prepends a still copy of the first frame:

```bash
python scripts/x2/retarget_lafan.py --gmr-root <GMR-limx> --bvh walk1_subject1.bvh \
  --start-s 0 --duration 10 --output outputs/x2/lafan/11_stand_walk_straight
python scripts/x2/retarget_lafan.py --gmr-root <GMR-limx> --bvh walk1_subject1.bvh \
  --start-s 119.8 --duration 9.7 --hold-s 2 --output outputs/x2/lafan/13_stand_walk_c
```

The wave is synthetic, not motion capture: `scripts/x2/make_wave_motion.py` keeps the legs, pelvis and waist at the
standing pose, raises the right arm to the side with the forearm upright beside the head, and rocks it with shoulder
roll and elbow. It checks every joint against the deployed soft limits (peak joint speed 2.2 rad/s):

```bash
python scripts/x2/make_wave_motion.py --output outputs/x2/wave/21_wave_right
```

For a short forward walk on the gantry, `scripts/x2/make_short_walk.py` cuts clip 11 down to about 1.4 m. It splices
the walk onto the clip's slow-down where the gait phase matches best, ends on a frame with both feet flat and the
centre of mass between them, slows playback to zero over the last 1 s, and holds that frame for 3 s. Clip 11 itself
ends with the rear heel lifting for the next step, so a tracker holding that frame keeps walking.

```bash
python scripts/x2/make_short_walk.py --output outputs/x2/lafan/15_walk_short
```

With `policy-pc-damped.yaml`, `--sim-dt 0.001` and a 40–60 ms command delay, `proteus_wamseq_62` walked
`15_walk_short` without falling in every run (nominal, kp0.8, mass1.1, friction0.5, push60). Under that delay it
overshoots the reference by about 0.5 m (1.8–1.9 m in all) and then stands still. Under kp0.8 the pelvis dipped to
0.44 m at the stop and the robot stepped back. The bridge rehearsal with `--lower-before-hold` also passed.

| Reference | Source window | Motion |
| --- | --- | --- |
| `outputs/x2/pico_hold` | — | Standing (`scripts/make_hold_motion.py --robot x2`) |
| `11_stand_walk_straight` | 0.0–10.0 s | Standing 3.6 s, then walking about 4.8 m |
| `12_stand_walk_b` | 10.6–20.0 s, 2 s hold | Walking about 4 m with a slight turn |
| `13_stand_walk_c` | 119.8–129.5 s, 2 s hold | Walking at about 0.6 m/s |
| `14_stand_walk_d` | 237.0–246.5 s, 2 s hold | Walking at about 0.65 m/s, arms raised once |
| `15_walk_short` | 11 spliced at 5.7 → 8.6 s, stop at 9.4 s | Standing 3.5 s, walking 1.4 m forward, stopping, standing 3 s |
| `outputs/x2/wave/21_wave_right` | synthetic | Standing, right arm raised, waved for about 5 s at 1 Hz, lowered |

### Damping on the robot

The launcher loads `policy-pc-damped.yaml` (`PROTEUS_POLICY_FILE`), written by
`scripts/x2/make_damped_config.py`: `policy-pc.yaml` with the arm joints' kd ×5 (1.1 → 5.5), the leg and waist
joints' kd ×2 (knee 7.7 → 15.3), and kp unchanged. The X2 bridge accepts kd up to 25 (10 elsewhere). With
the training gains (arm kp 17.5 / kd 1.1), the robot's arms oscillated at 4.4 Hz with growing amplitude within 3 s of
the policy taking over on 2026-11-12, while the legs and pelvis stayed steady.

sim2sim reproduces it with the commands reaching the motors 40 ms after the state the policy saw
(`scripts/x2/eval_sim2sim.py --sim-dt 0.001 --cmd-delay-ms 40`; on the board the policy alone takes 16 ms). Arm joint
speed rms while standing, first second → last second:

| Extra delay | Training gains | Arm kd ×3 | Arm kd ×5 |
| --- | --- | --- | --- |
| 40 ms | 0.21 → 0.48, 4.3 Hz | 0.03 → 0.00 | — |
| 50 ms | 1.13 → 3.80 | 0.24 → 0.34 | 0.15 → 0.19 |
| 60 ms | 3.49 → 5.37, fell | 0.41 → 0.45 | 0.23 → 0.36 |

On the wave at 50 ms the training gains fall (14.5 cm error) and kd ×5 tracks at 3.1 cm. Without extra delay, kd ×5
leaves standing, wave and walking errors within 0.2 cm of the training gains under kp ×0.8, pushes and one step of
delay. The vendor's arm gains (kp 50 / kd 3) oscillate worse under delay (5.2 rad/s standing at 40 ms).

With arm kd ×5 on the robot, the arms stayed calm (0.11–0.13 rad/s for 5 s), but the hips and pelvis rocked at 3.7 Hz
with a steady amplitude (hip pitch ±0.08 rad). At 50 ms of delay sim2sim shows the same standing leg oscillation
(0.88 → 0.72 rad/s); leg and waist kd ×2 removes it (0.04 → 0.01) and keeps standing calm at 60 ms (0.05), with
walking and wave errors unchanged (3.8 / 2.2 cm). On the robot, leg and waist kd ×2 removed the rocking: in 23 s of
Hold the leg joints moved at 0.01–0.03 rad/s and the waist at 0.01 rad/s.

With `policy-pc-damped.yaml` the robot keeps its feet planted on the wave up to 45 ms of delay (feet within 6 cm).
From 50 ms it takes a corrective step: about 40 cm forward as the arm rises, and 16 cm even when only standing. The
robot's delay is estimated at 40–50 ms, so expect a possible recovery step on the gantry. A second ONNX Runtime thread
does not shorten inference on the board (38 ms on two isolated cores against 10.4 ms on core 7 alone).

## Simulation checks

`scripts/x2/eval_sim2sim.py` runs the runtime's IntegratedSim2Sim with `policy-pc.yaml` (local tracking, no world
localization), nominally and under kp or kd ×0.8 / ×1.2, masses ×1.1, half floor friction, one control step of
action delay, and 60 N lateral pelvis pushes every 3 s. It reports the mean tracked-body error in the pelvis frame.
Results on 2026-11-12 (9 runs per cell):

| Actor | Standing | 11 | 12 | 13 | 14 | Falls |
| --- | --- | --- | --- | --- | --- | --- |
| `proteus_wamseq_62` | 1.8 / 2.0 | 3.7 / 7.5 | 3.9 / 21.7 | 3.2 / 8.4 | 4.2 / 10.1 | 0 of 45 |
| `proteus_60_bigdata` | 2.3 / 13.8 | 3.8 / 15.4 | 4.7 / 20.7 | 3.5 / 26.5 | 4.6 / 11.6 | 2 of 45 |
| `proteus_60` | 1.3 / 18.5 | 4.1 / 10.9 | 4.8 / 16.6 | 3.7 / 7.0 | 7.2 / 15.6 | 0 of 45 |

Each cell is the nominal mean / worst maximum error in cm. `proteus_wamseq_62` is the deployment default.
`12_stand_walk_b` drops the pelvis to 0.45 m under kp ×0.8, so start with the standing reference, then 11, 13 and 14.

On `21_wave_right`, `proteus_wamseq_62` scored 2.1 / 3.1 with no falls in 9 runs, and its right hand swings 12.7 cm
side to side against 15.3 cm in the reference. `proteus_60_bigdata` scored 2.5 / 13.8 (pelvis down to 0.50 m under
kp ×0.8), and `proteus_60` fell under kp ×0.8. An earlier wave with the arm raised forward made `proteus_wamseq_62`
step forward about 1.9 m under kp ×0.8, so the deployed wave raises the arm to the side.

## Rehearse on the PC

`scripts/x2/bridge_sim_test.py` runs the real `aimdk_bridge.py` and `tracking.py` processes against MuJoCo in place
of the AimDK HAL, with a virtual gantry, and presses the operator keys: a, a, lower the gantry, s, space
(`--lower-before-hold`: a, lower to the floor with the rope taut, a, release, s, space; `--gantry-gravity` and
`--gantry-z-offset` set how the robot hangs). It checks
that the bridge sends nothing before the first a, and reports the lowest pelvis height after the gantry is lowered and
the real-time factor. With `proteus_wamseq_62` all six references passed at a real-time factor of 1.00, with the pelvis
never below 0.546 m (0.654 m on the wave).

## On the robot

Deploy from the PC (no robot state is changed). The aarch64 / CPython 3.10 packages are the ET1 set resolved from
the same `uv.lock`:

```bash
bash scripts/x2/deploy_onboard.sh
```

On the board, the scripts load `/agibot/software/entry/cfg/basic_env.sh` and `aimdk_msgs` from
`/agibot/software/hal_ethercat`. The HAL publishes joint state at 1000 Hz and the IMU at 500 Hz. The board has
`isolcpus=5-7`. Core 4 runs `hal_ethercat_x2`, so the bridge runs on core 5 and Tracking on core 7 (ONNX in
10.8 ms there, 13.4 ms in the full loop).

1. Read-only check (with mc still running):

   ```bash
   ssh agi@10.0.1.40 proteus_x2/repo/scripts/x2/probe.sh
   ```

   **Compare the joint readings with the robot.** In passive mode on the gantry, the hips can be turned far from the
   standing pose (hip yaw ±1.7 rad was seen). Bring the limbs close to standing by hand before a takeover.

2. Dry run with mc running (the bridge creates no command publisher):

   ```bash
   ssh -t agi@10.0.1.40 PROTEUS_READ_ONLY=1 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

3. Run with the robot on the gantry. Terminal 1:

   ```bash
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

   The default reference is the wave `outputs/x2/wave/21_wave_right`. Its first frame is the standing pose, so the
   takeover, shaping and Hold are the same as with `pico_hold`; only `s` starts the arm. Choose another reference with
   `PROTEUS_MOTION=outputs/x2/pico_hold` (standing only) or `PROTEUS_MOTION=outputs/x2/lafan/15_walk_short` (short
   forward walk, verified on the robot).

   Terminal 2 (operator):

   ```bash
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/operator.sh
   ```

   Wait for "Tracker and LowState ready" in `logs/run-*/bridge.log`. Then:

   - `a`: takeover and shaping to the standing pose. The takeover runs `scripts/x2/release_mc.sh`, which stops mc
     and waits until `mc_main` has exited (`aima em stop-app mc` can take more than 20 s; allow up to 60 s)
   - lower the gantry until both feet are flat on the floor and the robot stands upright, rope still taut
   - `a` again: 2 s transition to the reference; the policy takes over (Hold)
   - slacken the rope
   - `s`: play
   - `space`: latched stop

   **Never press the second `a` while the robot hangs.** The policy already runs during Hold. On the first run on
   2026-11-12 the robot hung tilted 13° forward and 7° sideways with its feet off the floor when the second `a`
   came. The policy took that for a fall and stepped hard; with nothing under its feet it flailed and spun on the rope
   (yaw rate up to 7 rad/s) until `space`. The PC rehearsal reproduces this with
   `--gantry-gravity 0.22,0.10,-0.97` (leg joint speed rms 3.0 rad/s against 3.5 on the robot) and shows the order
   above keeping Hold calm (0.4 rad/s).

   **Load on the feet.** In Hold, the monitor (`scripts/x2/watch.sh`) shows the knee torques. Standing on its own
   (42 kg), each knee carries about +15 Nm. Slacken the rope until both read about +10 to +20 Nm before `s`, and let
   the gantry follow the robot. On 2026-11-12 the knees read about 0 Nm (mean −1.9 / −1.4 Nm while walking): the rope
   carried the weight, the feet had little traction, and `15_walk_short` covered about 0.7 m by leg odometry against
   1.33 m in the reference (1.26 m in the PC rehearsal), with roll swinging down to −12°.

   **After a board reboot** `mc_main` starts again, and `aima em stop-app mc` then takes about 15 s. The bridge
   loop blocks meanwhile and the Tracking shaping clock runs on, so the bridge latches damping with "takeover took …
   s; mc is released, restart the launcher and press a again". Quit and relaunch: with mc stopped the next takeover
   is immediate. The launcher warns when `mc_main` is running.

   **Joints past their limits.** A limp robot on the gantry can rest with a joint past its limit. Shaping may hold or
   bring back such a joint, never command it further out than it has been since `a`. A rejected command latches with
   the joint, target, limits and measured value in `bridge.log`.

   A stop, a command or state timeout over 0.2 s, or an invalid command latches damping. On exit the bridge sends
   damping. Restart the native controller with `aima em start-app mc`. Logs, bridge state and the Tracking recording
   are in `proteus_x2/logs/run-*/`.

The HAL has no command timeout of its own: if the bridge process is killed or the board loses power, the last command
stays in force. Keep the gantry and the physical stop within reach.
