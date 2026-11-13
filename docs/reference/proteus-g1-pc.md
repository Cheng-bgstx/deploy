# Proteus G1 from an external PC

To run the same stack on the G1's own Orin instead, see [Proteus G1 on the robot's Orin](proteus-g1-onboard.md).

The launchers default to the hardware-tested `checkpoints/proteus/proteus_wamseq_62/unitree_g1`. The per-robot
branches also exported the **wsq61** actor (`wsq61_wam_61robots_30000.pt`, SHA256
`1d1c65ddf5706b940744d87e2f86f7cea453c9240bcccc12d31923211f694bba`, provenance in
`checkpoints/proteus/wsq61/source/export.json`); select it with
`PROTEUS_POLICY_DIR=checkpoints/proteus/wsq61/unitree_g1`. The hardware results below do not cover wsq61.

Run CPU ONNX Tracking at 50 Hz and the Python SDK2 Bridge at 200 Hz on the PC. Tracking reads the cached NPZ reference; the G1 receives joint position/velocity targets, feedforward torque and PD gains over DDS. This route uses the existing Python 3.10 environment, SDK2, CycloneDDS and ZMQ. It does not require the aarch64 inline `unitree_interface` module.

The launcher defaults ONNX, OpenMP and BLAS to one CPU thread. Unbounded ONNX thread pools can consume many cores between 50 Hz inference calls and compete with DDS callbacks. A prior log also recorded a 174 ms command-send delay. The launcher now uses a shared 250 ms I/O timeout instead of the former 100 ms limit, so brief gaps do not latch a stop. Both settings apply when new processes start. Existing processes do not pick up launcher edits. Do not run policy benchmarks or simulation alongside a hardware test.

The exported artifact stays in `checkpoints/proteus/proteus_60/unitree_g1/`. Keep its generated `policy.yaml` unchanged. Generate the separate deployment overlay once:

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD"
export SIM2REAL_PYTHON=/absolute/path/to/existing/.venv/bin/python
"$SIM2REAL_PYTHON" scripts/g1/prepare_proteus_deployment.py
```

Model and motion artifacts remain outside Git. With the exported actor present and the pinned source motion plus G1 MJCF cached, generate the forward clip once:

```bash
"$SIM2REAL_PYTHON" scripts/g1/prepare_proteus_motion.py
```

The script copies source frames 90–330 inclusively at 30 Hz, without resampling or modifying qpos, and refuses to overwrite existing artifacts. The launcher also prepares it offline if the default clip directory is absent. Cache the source motion and MJCF before deployment; an incomplete existing directory must be inspected rather than silently overwritten.

`policy-pc.yaml` retains `proteus.reference_forcing: true`, shapes to the arms-down default standing pose, sets waist roll/pitch gains to kp=100, kd=2.5 (yaw remains 100/2), lets targets point up to 1 rad past the joint limits (the Bridge accepts the same range), and limits target changes to 15 rad/s. Proteus trains without clipping targets, and its ankles push them up to about 0.8 rad past the joint range; the former 5% inset clipped the ankles on most walking steps, causing lateral drift and falls during Hold in simulation. The original actor and observation contract are unchanged. The initial arm targets match the previous deployed default pose. A second A interpolates to the reference starting pose over 2 seconds (including settling), then resets histories and enters Hold. Direct policy takeover from the standing pose and lowering before Hold failed rehearsals, so wait for Hold ready before lowering.

Rehearse the actual Tracking controller without DDS or hardware:

```bash
"$SIM2REAL_PYTHON" scripts/g1/validate_proteus_deployment.py \
  --policy-config checkpoints/proteus/wsq61/unitree_g1/policy-pc.yaml \
  --npz-playback-limit-s 3
```

The rehearsal includes early button rejection, shaping, Hold, ideal gantry lowering, reference playback and a latched Select stop. The root constraint approximates support and does not model cable forces. Select does not maintain standing; stability metrics end before Select. The tests in `tests/test_g1_bridge_safety.py` cover real local ZMQ transport with mocked SDK channels, malformed/out-of-order commands, repeated state ticks, timeouts and stop priority.

Read-only bring-up using real G1 feedback and ONNX inference:

```bash
G1_INTERFACE=enp5s0 PROTEUS_READ_ONLY=1 scripts/g1/start_proteus_pc.sh
```

Then launch the protected hardware session:

```bash
G1_INTERFACE=enp5s0 scripts/g1/start_proteus_pc.sh
```

Wait for `Tracker and LowState ready` in the bridge log. Startup emits no DDS motor command and does not release high-level mode. Only a fresh physical A press with a ready Tracker, followed by its new init command, requests takeover. Keep the robot supported and an operator on the remote and hardware stop.

| Remote | Behavior |
| --- | --- |
| First A | Release high-level mode and shape over 10 seconds to the arms-down default standing posture, with gantry support. |
| Second A, after shaping completes | Smoothly transition to the reference pose over 2 seconds, then reset histories and heading and enter Hold. Wait for Hold ready while still supported. |
| Lower the gantry | Place the feet on the floor while Hold is already active; retain protective support. |
| Start | Play/pause the reference. The launcher defaults to eight seconds of original-speed forward walking, then holds with policy control active. |
| Select | Priority latched software stop. Follow measured joint angles with PD gains retained; active posture support is lost. |

After Select or a fault, restart with A → A → Start. At the configured playback limit, Start cannot extend the segment; a new shaping session resets the reference. `PROTEUS_PLAYBACK_LIMIT_S` changes the limit for a new session, but cannot extend beyond the selected motion's length.

The launcher now defaults to `PROTEUS_CLIP=walk_straight_short` (`checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short/`, playback limit 5.1 s), including when launched through `/home/galbot/Documents/Proteus-g1/start-pc.sh`. It is `walk3_subject1` frames 6670–6823: from a stand, 2.2 m straight forward at 0.52 m/s, to a double-support stand. A scan of the dataset's 12 walk motions found four stand → straight forward walk → stand segments; in WAM simulation with local forcing and 0.15 m/s pushes, this one drifted 0.05 m sideways on average (0.15 m worst) against 0.53 m for `walk_forward_to_stand`, with no falls. In-place dances, each from a stand to a stand, are `PROTEUS_CLIP=dance_in_place` (`dance1_subject3` frames 345–525, 6 s, feet down, the richest arm movement) and `dance_steps` (`dance1_subject3` frames 165–405, 8 s, two steps); they come from scanning the 8 dance motions for windows within 0.6 m, without flight and with arm movement in every second (an earlier `dance_arms` pick stood still for its first six seconds), and none fell in simulation with the measured joint friction and a 30 ms command delay, though Hold after `dance_steps` moved up to 0.55 m. `PROTEUS_CLIP=jump_once` (playback limit 3.16 s) is one vertical jump in place from a stand to a stand (`jumps1_subject5` frames 6490–6585, 0.23 s in the air, about 7 cm lift): with the measured friction and 0.15 m/s pushes it did not fall in simulation up to 40 ms command delay with wsq62 or ht500. The following jumps of the same sequence (0.33 s in the air) already fell at 30 ms, so longer jump clips are not provided. `PROTEUS_CLIP=walk_forward_to_stand` selects the longer walking clip (playback limit 8.63 s), which retains seconds 3–11.63 (frames 90–349) of `walk1_subject1` at original speed, skipping the initial preparation segment and ending after the subject has stopped with both feet flat and side by side, before the turn at 12.2 s. Hold freezes the last reference frame, so the clip must end standing: the former `walk_forward_original` clip (seconds 3–11) ended with the right foot mid-step, and Hold kept it raised about 3 cm with the feet 12 cm apart fore-aft. In simulation with the WAM actor, the new clip holds both feet flat for 10 s after playback (pelvis drift 0.2 cm/s, local forcing), including after 0.15 m/s pushes. `prepare_proteus_motion.py` regenerates it. `PROTEUS_MOTION_PATH` can override the reference for both preflight and Tracking. Local forcing still drifts about 0.5 m sideways over the walk, and the clip has not been hardware tested. Keep gantry protection through playback and Hold. Changes apply at the next launch and do not change a running session.

The launcher defaults `PROTEUS_IO_TIMEOUT_S=0.25` and passes the same window to Bridge (`--timeout-s`) and Tracking (`SIM2REAL_IO_TIMEOUT_S`). Within this window, Tracking keeps using the latest valid feedback for normal inference and reference playback, and Bridge keeps resending the latest valid command at 200 Hz through command gaps. A fresh update continues immediately without freezing targets, latching a stop epoch, or requiring A. Repeated LowState ticks do not reset the freshness clock. This is bounded runtime tolerance; it does not establish physical gait stability with 250 ms delayed observations or commands.

Feedback or command loss longer than this window no longer drops the robot by default: with `PROTEUS_HOLD_THROUGH_GAPS=1` (the launcher default, Bridge `--hold-through-gaps`) Bridge keeps sending the last valid PD targets, Tracking pauses inference and the reference clock until fresh LowState returns, and control then resumes without a new A; late or repeated commands after the gap are dropped. A gap over 1 s, or a LowState tick that goes backwards, still latches damping and needs A → A again: on 2026-11-04 a robot reboot during a running session (117 s of silence) let the Bridge resume the old session on the freshly booted robot for about 3 s before it was stopped. Stop the launcher before rebooting the robot. On 2026-11-04 a 343 ms LowState reception gap 4.6 s into walking (the robot tick kept advancing, so the samples were not delivered to the PC process) latched `kp=0` damping and the robot fell. Holding is not a cure: in simulation, any blind interval over about 0.3 s during walking (holding targets or inferring on stale state) still usually fell, while 0.15 s gaps were survived; standing Hold rides through. `PROTEUS_HOLD_THROUGH_GAPS=0` restores latched pure damping (`kp=0`, robot-configured damping gains). Bridge now decodes LowState in its 200 Hz loop (the DDS listener only stores the newest sample, without the lock), runs on performance cores 0-3 and 6-15, and logs every reception gap over 30 ms with its robot tick jump; `scripts/g1/lowstate_gap_monitor.py` runs beside it (`lowstate-monitor.log`) as an independent read-only receiver, so a gap seen by both points at the robot or network and a gap only Bridge logs points at the Bridge process. Invalid state/commands, mode or DDS write faults, and orderly shutdown retain their existing fault response. Select retains its higher-priority software stop behavior. Standby never emits motor commands before takeover. For another window, set `PROTEUS_IO_TIMEOUT_S` before a new launcher session; the accepted range is greater than zero and at most one second. Standalone Bridge and ZMQ I/O retain the 100 ms default, and manual launches must configure both together. Zero commands following measured angles within 0.05 rad are allowed at mechanical limits so they can acknowledge a stop. Updated Python Bridge and Tracking use optional control-generation / stop-epoch ZMQ metadata; deploy both together.

Local forcing lacks global root localization and permits drift. One 60-second diagnostic remained upright but reached 3.16 m root XY error. The revised two-10-second playback rehearsal reached 1.35 m; the latest bounded three-second rehearsal with waist roll/pitch PD=100/2.5 and the arms-down-to-reference transition reached 0.138 m (relative body error mean 0.042 m). Simulation does not certify hardware safety. A PC watchdog cannot deliver a stop after cable or PC failure: the robot-side communication timeout and independent hardware stop still require on-site validation.

Logs, PIDs and `tracking.npz` are saved under `outputs/proteus-pc/<timestamp>/`. Ctrl+C or SIGTERM stops both processes; the bridge sends a brief damping sequence on orderly shutdown. A killed process, lost network, or Python shutdown is not a hardware emergency stop.

Native SDK channel destruction is bounded during shutdown; recordings and motor fallback finish before any forced process exit caused by a destructor timeout.
