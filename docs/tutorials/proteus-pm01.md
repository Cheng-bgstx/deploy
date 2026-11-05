---
title: Proteus WAMSeq on EngineAI PM01
slug: /tutorials/proteus-pm01
---

# Proteus WAMSeq on EngineAI PM01

## Current baseline: classic ankle parser (2026-11-05)

`start_proteus.sh` and `start_pico.sh robot` now send `parallel_parser_type=0` (`PM01Codec.parallel_parser_type`). `policy.onnx`, `policy.yaml` and every commanded kp/kd are unchanged. T800 keeps `RL_PARSER=1`, and the relay tools (`policy_bridge.py`, `trial_policy.py`, `bench_damping.py`) still use `RL_PARSER=1` for active commands. **A first gantry-protected hardware walk completed on 2026-11-05; the result is below.**

Why the parser changed:

- On 2026-11-05 at 19:35, the first powered run after the startup containment was removed shook the robot and reset it about 3.4 seconds after `tracking` started. The PC link events after the loss matched the 2026-11-04 reset to within a second. That run has no recording, so its joint-level cause is inferred from this timing and the unchanged transport.
- No competing firmware writer was found. `JointCommandCallback` only runs from `JointBridgeRunner::Run()`, and `joint_bridge` has a single runner. Across 18 trials, 38,277 RL-phase motor-driver samples matched the commanded gains, with ankle motor kp and kd at zero.
- On the same topic and in the same mode, classic ankle `kd=3` was stable in all 23 logged trials. In `walk-deploy-006` it reduced ankle roll from 9.39 to 0.42 rad/s within 24 ms.
- `walk-deploy-006` gives about 8-9 ms from the PD law to real acceleration, and an ankle-roll inertia of 0.011-0.016 kg m^2. The simulation model has 0.04, which is why simulation stayed stable. A delayed model with 0.0107 kg m^2 and 8 ms reproduces the incident: +/-40.8 rad/s, +/-0.24 rad and 28 Hz, against +/-39 rad/s, +/-0.24 rad and 25-28 Hz recorded. These are estimates from one oscillation and one recording.
- In that model, the free-foot stable RL range at `kp=37.701` is about `kd` 0.54-1.23. Training `kd=2.4916` is above it and the earlier `kd=0.3` is below it. Classic keeps PD inside the motor driver, outside this loop.

Trade-offs:

- Classic gains act on the motors, so joint-side ankle gains are `J^T K J`: pitch about 4.09 times and roll about 1.11 times the commanded values, using the installed firmware's ankle math at recorded poses.
- Vendor documentation recommends the RL parser for policies trained on serial joints, so this is a deliberate deviation.
- Motor readback confirms classic ankle kd reaches the motors unscaled, but only `kp=0` classic commands have been tested here. `kp>0` relies on vendor documentation.

Simulation with the unchanged ONNX and those joint-side ankle gains passes all four motions at a 2-ms step:

| Motion | Mean error, RL baseline | Mean error, classic | Max frame error, RL baseline | Max frame error, classic |
| --- | --- | --- | --- | --- |
| aiming | 2.59 cm | 3.21 cm | 4.43 cm | 5.56 cm |
| dance | 2.93 cm | 3.84 cm | 6.63 cm | 6.61 cm |
| fight | 4.79 cm | 4.44 cm | 19.37 cm | 10.54 cm |
| walk | 2.29 cm | 4.29 cm | 3.51 cm | 11.60 cm |

Walk tracking is worse. This simulation has neither the measured ankle inertia nor the loop delay, so it does not prove hardware stability. Artifacts: `outputs/pm01/classic-equivalent-2ms/summary.json` and `outputs/pm01/classic-equivalent-config/policy-sim.yaml`.

Hardware run, with the gantry and emergency stop ready throughout:

1. Check joint readings, not just appearance. On 2026-11-05 the robot looked upright with its pelvis turned about 180 degrees relative to its torso (`J12` about -3.4, `J08` about -3.1, `J00` about -3.1, `J01` about +2.3). Enter `joint_bridge` through `pd_stand`, which rejects poses more than 1.2 rad from standing, not straight from `passive`.
2. Hang the robot with its feet free. Run `bash scripts/pm01/start_proteus.sh --record` and press nothing for 10 seconds. It should stay as quiet as `passive`.
3. Press **i** to shape. The arms rise to nearly horizontal: this is walk frame 0.
4. Lower the gantry until the feet lightly touch, then press **]** to enter policy Hold.
5. Transfer load gradually, then press **Space** to play.

Damping alone does not hold a standing robot (see `walk-deploy-005`), so the gantry must be able to catch it at every step.

First hardware walk (2026-11-05 21:01, `policy_tracking_record_policy_20261105_210134.npz`): all 398 playback frames ran with no reset, and motor temperatures stayed at or below about 43 C afterwards. Against simulation with the same ankle gains and reference forcing, knee range was 1.14/1.04 rad versus 1.18/1.01, foot lift 17.1/16.9 cm versus 17.8/18.1, and mean pelvis tilt 5.2 degrees versus 5.9. Differences from that simulation:

- Reported ankle pitch torque peaked at 106 Nm against the 61 Nm training limit. Simulation clips at 61 Nm, so it cannot show this.
- Ankle-roll motion above 5 Hz was about 2.5 times simulation.
- Ankle pitch tracking is about twice as far from the reference as the ideal RL baseline in simulation (0.145 versus 0.074 rad RMS). The classic simulation shows the same loss (0.147).

After playback the robot kept rocking in the final-frame Hold for about 33 seconds, as it also does in simulation, with a peak of 18.27 rad/s and 133.7 Nm near 110 s. Root position is not measured, so foot placement and distance travelled were not compared.

## Superseded baseline: original training outputs with RL_PARSER (2026-11-05)

This baseline used `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml`, regenerated from the original wsq61 contract. The recovery candidates below are historical experiments, not the current deployment configuration.

- Restored the original actor morphology and all training gains, including ankle `kp=37.701, kd=2.4916`; the kd=0.3 and kd=0.5 candidates are not the baseline.
- Removed policy soft-position clipping, body/arm/head target slew, the two-second takeover blend, and limiter-state reseeding. First-frame shaping no longer requires enabling policy clipping. The ten-second pre-policy shaping interpolation remains an entry procedure.
- PM01 uses the standard EngineAI transport: policy q/dq/feed-forward/kp/kd are forwarded in SDK order under the shared position/velocity/effort bounds and the per-publish 0.2-rad target limiter (100 rad/s at the 500 Hz publish rate). No PM01-only clipping, slew or latched stop state remains.
- Training gains are used everywhere: zero mode commands the measured pose with zero stiffness and the base passive damping, and the parser stays `RL_PARSER=1`. Finite-data checks, the feedback/command watchdogs and the joint speed/torque limits at the released actuator effort remain. The client rejects retired limiter/blend profiles instead of silently ignoring them.
- Source-versus-export comparison: 64 inputs, maximum absolute error 0. All four reference motions pass the 2-ms integrated evaluation; original walk mean world-body error is 0.02288 m. These results do not model the physical parallel-motor feedback loop.
- Rehearsal from the recorded `walk-deploy-006` pose, with no experimental command gate, completes the original walk. The existing 2–12-second supported-hold quality check fails on left ankle-roll excursion (0.1604 rad versus its 0.12-rad criterion); do not label this report a full pass. Over seconds 5–12, maximum whole-body excursion is 0.02893 rad and maximum speed RMS is 0.00430 rad/s: the early motion settles rather than persisting throughout the hold.
- The live read-only attempt could not create DDS because `eno1` was down with no IPv4 address. No new motor commands were sent. Restoring training output is not evidence of completed physical deployment.

Artifacts: `outputs/pm01/pure-training-2ms/summary.json`, `outputs/pm01/pure-training-pose006.json`, and `docs/experiments/proteus-pm01-recovery.json`.


:::warning Hardware output re-enabled for supervised trials
An enabled PM01 violently oscillated after policy startup on 2026-11-04, before any
operator command, and the operator reported a whole-robot reboot. The latest
recording contains 128 frames, all in `zero` mode; the transport sent damping
commands with `kp=0`, `kd=2` and `RL_PARSER=1`. The recording does not contain the
actual wire gains or firmware command feedback, so the physical cause is still
unestablished.

The code-level lockout, the latched classic-damping stop and the kd=0.3
qualification gate have been removed: they changed the closed-loop gains and the
ankle control law away from training, which is the opposite of what the deployment
needs. What remains is limited to protections that do not alter the control law:
finite-data checks, command/feedback/IMU freshness watchdogs, `joint_bridge` mode
ownership, released actuator effort/velocity bounds and the relay supervisor.

Treat the first powered run as supervised test time: support the robot with a
gantry or sling, keep the emergency stop in reach, and watch for the startup
oscillation from the incident. The weight-bearing exit also remains unvalidated
after `walk-deploy-005` (a feedback timeout during load transfer removed position
stiffness, followed by knee flexion and a large torso tilt). Earlier loopback
success demonstrated software integration only. The history below is retained as
diagnostic context, not as a validated deployment procedure.
:::

Read-only investigation after reconnection on 2026-11-05 narrowed the failure:

- The 2.539-second incident recording contains only `zero` mode. Reported peak
  ankle-roll velocities were 41.44 rad/s (left) and 39.44 rad/s (right), with
  peak reported torques of 70.51 and 72.83 Nm respectively. These are feedback
  values, not a capture of motor commands.
- Disassembly of the installed firmware confirms that `RL_PARSER=1` computes
  joint-space PD torque, transforms it through the parallel-ankle Jacobian,
  writes it as motor feed-forward torque, and clears ankle motor stiffness and
  damping. Thus sending `kp=0, kd=2` does not mean motor-side damping of 2.
  Startup transmission through this path is confirmed; the physical mechanism
  behind the oscillation (including loop delay and transform behavior) remains
  unverified. Changing the parser or damping alone is not a validated fix.
- Boot records confirm a motion-computer reboot, with no clean shutdown entry.
  Previous-boot journal, pstore and motor logs did not establish its cause.
  The last motor-log entries predate the incident; their powerboard error must
  not be attributed to this event. Robot and PC clocks differed by about
  12 minutes at inspection, so raw wall-clock timestamps cannot be compared
  directly. Power protection, a hardware reset and a software failure remain
  unresolved alternatives.

Further offline checks and containment:

- An earlier containment made the PM01 codec silent during initial `zero` mode,
  model loading and close-before-start, and latched classic native damping on
  zero/close, stale feedback, stale commands or feedback errors. That containment
  also replaced the training ankle damping and parser, so it has been removed:
  the transport now uses the standard EngineAI behaviour described in the current
  baseline above.
- The installed ankle math library was run on the PC with its loaded geometry
  at all 128 recorded poses per side. Motor angles were reconstructed from
  serial poses because raw motor feedback was not recorded. The maximum
  relative Jacobian error against finite differences was 0.173%, maximum
  condition number 2.002, and the reconstructed damping power was nonpositive
  in every sample. This does not support a simple geometric sign reversal or
  singularity in that reconstruction; it does not validate real sensor
  calibration, sampled-loop delay, saturation or stability.
- The loaded motor task period is 2 ms. Ankle damping under the RL parser is
  computed on this onboard task from onboard feedback, so its loop is not a
  round trip through PC policy inference. The inspected bridge runner restores
  the classic parser on exit. Its run path only services ROS callbacks; a
  command-age timeout has not been established. Stopping PC publication must
  not be assumed to stop a previously commanded robot.
- The same robot's loaded `rl_basic/vae/default` uses ankle `kd=0.3`; its
  `rl_amp/walk_stand/release` uses `kd=0.2`. The incident's startup `kd=2.0`
  is 6.7–10 times these references, and the exported Proteus ankle `kd=2.4916`
  is 8.3–12.5 times them. These are controller-specific reference settings,
  not verified universal safe limits. Native passive uses ankle `kd=3` on a
  different control path; that does not validate RL-parser damping of 2.
  Gain mismatch in the sampled torque loop is now a leading hypothesis, not
  a demonstrated complete cause. Changing the policy gains would also change
  the trained controller's dynamics and needs separate validation.
- Both motion computer and Orin rebooted. Independent `/proc/uptime` samples
  placed their boot times about 7.9 seconds apart, after the incident. This
  supports a shared power/reset event but does not identify its trigger. Orin's
  wall clock was unset (1970) and it had no persistent previous-boot journal;
  the comparison used uptime and the PC clock, not those wall-clock dates.

An offline recovery candidate changes the four ankle damping gains to `0.3` and
updates the corresponding log-damping features in the Proteus morphology descriptor.
Changing only YAML would leave the actor conditioned on the original dynamics.
The re-export matches the source actor exactly on 64 sampled inputs. At a 2 ms
simulation step, all four reference motions pass; mean world-body errors are
2.54 cm (aiming), 3.77 cm (dance), 8.00 cm (fight), and 2.58 cm (walk). These are
offline tracking results: the simulation's serial-joint armature and torque clipping
do not establish stability of the physical parallel-motor loop. Commands, hashes
and complete metrics are recorded in `docs/experiments/proteus-pm01-recovery.json`.

The separate `scripts/pm01/bench_damping.py` tool defaults to read-only recording.
Its experimental execution path is limited to at most one second of low ankle
damping. Optional `--ankle-kp` holds the measured initial ankle pose; the default
has no position stiffness. Neither mode requests joint feed-forward torque. It requires fresh
feedback and checks classic damping before starting, then restores and checks
classic damping afterward. A separate process handles worker hangs; both processes
must run onboard to survive loss of the PC connection. This is not a deployment
watchdog and cannot handle failure of Linux, DDS or the motor bus. Hardware
execution also requires read-only access to the
inspected motor driver's actual gain arrays; an exact binary hash rejects other
versions. The installed `JointCommand` feedback does not populate its parser byte,
so that byte alone cannot confirm a parser change. `MotorDebug` arrays contain
24 values followed by a timestamp appended by `DataMonitor`, not a 25th motor.

## Bounded hardware recovery results (2026-11-05)

With the robot fully suspended on a gantry, twelve short ankle trials completed:
low damping from `0.05` to `0.3`, followed by current-pose stiffness up to
`kp=37.701, kd=0.3`. The longest ankle hold was one second. Separate whole-body
shaping probes of one and three seconds completed; the three-second probe moved
the arms about 19 degrees, with peak reported joint torque 3.20 Nm. Every trial
confirmed return to classic damping using actual motor gain readback.

The first real **policy-mode** trial also completed: 510 shaping requests followed
by 51 policy requests over about one second, then a latched return to classic
damping. Peak ankle speed was 0.186 rad/s, ankle feedback torque 0.729 Nm, and
whole-body feedback torque 2.184 Nm. The motion computer's boot ID was unchanged;
post-test readback showed zero stiffness/feed-forward, native damping, and no new
reported fault flags. This is a short static takeover, **not walking, unsupported
standing, long-duration stability, or proof of the original reboot's cause**.
The operator reported no abnormal vibration and the subsequent read-only check
confirmed `passive` with no hardware command publishers.

The subsequent three-second hardware hold also completed: 510 shaping requests
and 151 policy requests over 3.009 seconds. Peak ankle speed was 0.200 rad/s,
ankle feedback torque 1.149 Nm, and whole-body torque 4.253 Nm (right hip roll).
The supervisor confirmed classic damping; the motion computer did not reboot and
no new reported motor faults appeared. The operator reported no abnormal vibration
or sound and returned to `passive`, confirmed by read-only feedback with no command
publishers. Logs are `.scratch/pm01-recovery/static-policy-hold-003.*`.

A separate grounded simulation of the same static reference for approximately
20 seconds passed at a 2 ms physics step: mean body error 3.14 cm, maximum
frame-mean body error 3.82 cm, minimum playback pelvis height 0.812 m. This used
the `kd03/policy-sim.yaml` candidate without the commissioning slew limit, with
measured simulated root position and the robot initialized at the reference pose.
It does **not** validate real hardware handoff, contact loading, or the deploy
configuration's reference-forced root estimate. Do not substitute this configuration
into the suspended hardware trial merely because this offline evaluation passed.

This test used a separate commissioning profile: the re-exported `kd=0.3` actor,
target slew limited to `0.05 rad/s`, and a static reference captured from the
current hanging pose. The slow limiter is unsuitable for ground balance: grounded
rehearsals with these restrictive limits fell. Do not use this profile for walking.
The policy-mode transition preserves the last shaping target so it does not
abruptly remove the arms' gravity-supporting PD torque.

`rehearse_trial.py` now supports the root with a MuJoCo mocap weld. Resetting free-root
position/velocity every substep still allows free-fall acceleration in the dynamics
solve and understates gravity loading, so earlier such results are superseded.
With the weld, the original walking clip's arms-horizontal first frame hit the
5 Nm arm guard during shaping. The static hanging reference passed the same guard
without raising limits (peak simulated ankle speed 0.165 rad/s). The weld is still
an approximation, not a model of the actual cable or motor-loop dynamics.

The experimental tools are separate from the launchers:

- `shadow_policy.py` runs real feedback through the ONNX policy without a hardware
  command publisher (775 frames were checked).
- `trial_policy.py --policy-config <commissioning.yaml> --motion <static.npz>
  --output <new-record.npz>` sends requests only to `/sim2real/pm01/request`.
  Run from the root environment after sourcing `scripts/pm01/env.sh`.
- `policy_bridge.py --execute --config <bridge-config.json> --output <new-log.jsonl>`
  runs **on the inspected motion computer**, with its installed ROS environment
  and root readback access. Copy `bench_damping.py` beside it. Its independent
  onboard supervisor permits about 10 seconds of shaping and, by default, one second of policy
  hold; PC command loss, invalid targets or excessive speed/torque latch classic
  damping. Start the PC client first; it waits for the bridge. `--shaping-only
  --shaping-seconds 1` instead ends before policy mode. No tool plays the reference.
  `--hold-seconds 3` selects the next bounded stage; it keeps the same speed,
  torque and command-loss guards. The three-second static weld rehearsal passed
  (peak joint torque 3.728 Nm), followed by the bounded hardware result above.

These commands require the exact tested gains, joint order, static dataset manifest,
firmware hash and hanging setup; they are not an unrestricted deployment recipe.
The supervisor checks actual native damping on exit and survives PC disconnection,
but cannot cover operating-system, DDS or motor-bus failure. The local evidence is
`.scratch/pm01-recovery/static-policy-hold-001.*`; configuration, artifact hashes,
simulation and hardware metrics are indexed in the recovery experiment JSON.

### Original-walk commissioning entry (experimental)

The original walk reference travels approximately 3.00 m forward and 0.21 m
laterally in its initial heading frame. A following gantry and this clear travel
space are required; a fixed hanging test is not equivalent to grounded walking.
The operator confirmed a following gantry and clear path on this installation.

With feet lightly touching the floor and the sling carrying the torso, the first
hardware attempt to shape into walk frame 0 stopped after 6.8 seconds: left ankle
pitch reached the original 2 Nm guard. Native damping was confirmed afterward;
the operator reported no abnormal vibration or sound and returned to passive.
Active-motion peak joint speed was 0.214 rad/s (the larger 1.245 rad/s peak in the
whole log occurred during the damping return). This did not reach policy mode.
The supported-contact demand estimate requires about 3.06 Nm at the ankle and
6.30 Nm at shoulder roll. It is a model estimate, not permission to bypass a fault.

The separate `--walk-once` entry implements shaping, explicit policy activation,
paused first-frame policy, a separate play barrier, one original-speed reference
playback, and latched native damping. It uses the re-exported ankle `kd=0.3`
actor, body/arm target limits of `6 rad/s`, a head limit of `1 rad/s`, and a
two-second takeover blend from the last shaping command. A `0.5 rad/s` grounded static variant and a four-times-slower walk
with a `2 rad/s` limiter failed offline and must not be deployed.

The walk entry has a separate operating envelope: during shaping the ankle guard
is 4 Nm and shoulder-roll guard 8 Nm. During policy/playing, ankle pitch is limited
to 50 Nm, ankle roll to 20 Nm, joint speeds to 8 rad/s and target steps to 0.12001 rad;
the remaining per-joint torque limits are in `WALK_TORQUE_LIMIT`. These values are
below the prepared model's limits but are **not a certification of hardware safety**.
The actual ankle motor feed-forward arrays are additionally limited to 25 Nm.
Fresh onboard IMU feedback is mandatory; tilt over 0.35 rad or angular speed over
3 rad/s stops the session. Existing feedback, temperature, motor-fault, command-loss,
single-writer and independent-supervisor checks remain active.

Three hardware takeover attempts (`walk-deploy-001` through `003`) stopped
before playback: requested head torque, requested right-shoulder torque, then a
head target-step check after a missing private request sample (40 ms rather than
20 ms between received samples). All confirmed native damping on exit. The
operator corrected the third report to **visible sustained vibration**. Its
approximately 10-second policy hold had shoulder oscillations around 1.95 Hz,
with 0.51/0.48 rad peak-to-peak motion. No new motor fault or reboot was observed.

Extending the simulated supported hold to 12 seconds reproduced the vibration.
The added `2 rad/s` arm limiter caused a limit cycle: simulated shoulder motion
was 0.43/0.53 rad peak-to-peak. An offline comparison at `6 rad/s` reduced this to
0.0084/0.0048 rad; adding the two-second takeover blend gave 0.0212/0.0135 rad.
These comparisons identify a limiter-induced oscillation in the simulated loop.
Do not use the arm-2
profile. Merely completing a simulated walk without a fall was insufficient.
The relay now also checks a two-second shoulder-motion window after four seconds
of paused policy: peak-to-peak motion above 0.25 rad or speed RMS above 0.6 rad/s
latches damping. Playback requires this check to pass. Private requests use
reliable delivery with bounded queues; deadline, ordering, and step checks remain.

The corrected `walk-deploy-004` hardware attempt then held the original first frame
for 45.02 seconds under gantry support. After the first four seconds, shoulder
peak-to-peak motion was 0.00191/0.00038 rad, shoulder speed RMS 0.00224/0.00210 rad/s,
and maximum joint speed 0.0117 rad/s. No oscillation or motor guard fired. It ended
at the play-wait deadline before operator load-transfer confirmation; native
damping was confirmed. **This validates supported first-frame holding, not weight
bearing or walking.** The operator-confirmation window is now 180 seconds; the
100 ms command-loss protection and other motor guards are unchanged.

`walk-deploy-005` held policy for about 57.28 seconds before a joint-feedback-age
timeout. During the final 200 ms before classic damping, peak joint speed was
0.0076 rad/s; the last recorded policy tilt was 0.067 rad. Joint motion began
about 18 ms after the first damping command, and knee flexion increased by about
0.05 rad within 100 ms. A subsequent live read observed torso tilt near 0.70 rad.
The operator reported that weight transfer was in progress, applied the E-stop,
and confirmed the gantry caught the robot and people were safe. No play signal
was sent. Both relay and supervisor confirmed native damping; the motion computer
did not reboot. This establishes the sequence **feedback timeout → removal of
position stiffness → loss of stance support**. The cause of the feedback gap is
not yet established. Pure damping is not a standing fallback; previous simulation
only covered normal playback, not this loaded exit. Hardware walk remains unvalidated.

Offline fault injection now reproduces the missing exit case: after load transfer,
setting zero position stiffness and native damping caused root height to fall from
0.810 m to 0.254 m within two seconds. With an ideal rigid support constraint,
height remained 0.810 m. The serial-joint model and ideal weld do not validate a
physical sling; continuous mechanical support must be established independently.
Read-only timing probes also found a 20.06 ms gap in a lightweight subscription
and a 66.02 ms maximum iteration in the loaded receiver. Batching callbacks reduced
feedback age, but these measurements do not prove the source of the original gap
or qualify a new hardware timing budget. That timing budget was unchanged in this probe. A later experiment added a 20–100 ms feedback grace: retain the last target/gains, pause PC inference, and reject old request epochs; gaps beyond 100 ms still stop. Trial 006 never activated this grace and still produced continuous ankle oscillation.

Actual deploy observations, the full Tracking client, the command gate, first-frame
shaping, a 12-second supported policy hold, idealized support release, and original
playback completed in simulation from two recorded contact poses with the corrected
blend. Peak simulated ankle speed was 6.19 rad/s and ankle torque 45.97 Nm across
both runs; settled shoulder peak-to-peak motion stayed below 0.026 rad. Support
release replaces the weld with a vertical supporting force ramped to zero over
one second, an approximation of physical load transfer. An earlier profile's
installed-ankle-math estimate was 21.86 Nm peak motor command, using reconstructed
motor angles; this is not a measured result for the new candidate. Full physical playback remains unvalidated
until a completed hardware result is recorded.

The PC client takes `--walk-once --policy-signal <new-absent-file>
--play-signal <different-new-absent-file>`; the onboard relay takes `--walk-once`.
The historical client verified the original motion, target limits, and takeover blend; the current client has removed the blend and rejects those old profiles.
After shaping it prints `SHAPING_COMPLETE` and keeps PD support. With feet touching
and the sling still carrying the main weight, write `policy` to the policy file
to enter `POLICY_HOLD`. Do not unload the gantry while only PD is active: that
sequence failed simulation. After policy has settled and the stationary check
passes, transfer load gradually while retaining gantry protection. Only after
stable foot loading and following-gantry readiness are confirmed may the operator
write `play` to the play file. Playback runs once, with
an 8.5-second onboard deadline. No play signal within 180 seconds of policy entry
causes damping, as does a stop or failure. The gantry must remain able to support
the robot on every exit. The independent supervisor has a 270-second overall bound;
neither a stale signal file nor a late request can start another playback.

The vendor's [parallel-ankle control explanation](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
also distinguishes classic motor-side PD from RL torque conversion with zero
motor PD gains. This newer documentation corroborates the semantic distinction;
it does not certify the installed 2025 firmware or a replacement gain set.

Branch `feat/proteus-wamseq-engineai-pm01` supports the **24-joint PM01/pm_v2**
using the released `engineai_pm01` contract and wsq61 WAMSeq actor. Robot aliases:
`pm01`, `pm_v2`, `engineai_pm01`. This is distinct from the T800's 25-joint
contract and from `engineai_pm01_edu_official`, whose defaults and dynamics differ.

## Verified scope

On 2026-11-04, SSH inspection of the connected PM01 found the vendor package
`76445d09e967a16eb191d10aa05fe2ac5df6c19f` (2025-10-16), configured as `pm_v2`.
Its `serial_pm_v2.urdf` has 24 movable joints and matches the vendor ROS workspace
URDF exactly. SDK order is J00–J23: J12 is `J12_WAIST_YAW` and J23 is
`J23_HEAD_YAW`. The IMU site is aligned with the pelvis; deploy YAML exposes
`imu_to_pelvis_wxyz` for a different verified mounting.

The installed `MotionState` has **`current_motion_task_name`**, while the newer
public interface has `current_motion_task` and `available_transition_motions`.
The codec accepts either field, but both DDS endpoints must use matching message
definitions. Merely adding a Python field fallback does not fix a wire-schema mismatch.
`JointState`, `JointCommand`, `ImuInfo`, and `RL_PARSER=1` match the inspected firmware.

This release leaves joint and IMU header stamps at zero. The PM01 codec uses
monotonic local receive time for those streams, keeping the 250 ms feedback-loss
watchdog. Nonzero stamps retain duplicate/backward rejection; a return to zero
after a nonzero stamp is rejected. Zero-stamp streams cannot distinguish freshly
republished old sensor values from a new measurement. The read-only check reports
both header stamps and accepted-feedback ages for diagnosis.

Completed: full ONNX equivalence, four-motion integrated MuJoCo evaluation,
SDK-free mapping/watchdog tests, real ROS loopback serialization with the installed
schema, and PICO IK/ZMQ/ONNX tests with a synthetic XR input fixture.
The actual `start_proteus.sh --record` launcher also passed an isolated loopback
test with zero-stamp synthetic feedback: 10-second shaping, first-frame hold,
playback, damping, clean exit and 878 recorded finite-action frames. This exercises
the installed ROS schemas and ONNX policy, not robot dynamics.
**Sustained physical motion tracking and real PICO hardware have not been validated.**
The bounded static hardware result is documented above. The
initial check found `robotics.service` inactive. After the operator enabled motor
power and selected `joint_bridge`, PC subscriptions received joint/IMU data at
about 500 Hz and mode feedback at 5 Hz, with one compatible command subscriber
and no competing command publisher. The zero-header compatibility fix passed
the hardware read-only readiness check; no Proteus control commands were sent
by that check.

## Prepare assets

Use the root project for inference, and `venv/pico` for IK/teleoperation. Existing
environments are reused with `--no-sync`. Place source checkouts under one parent
directory: `Proteus-wamseq`, `GMR-engineai`, and
`ubisoft-laforge-animation-dataset`. The GMR checkout needs
`assets/engineai_pm01` and the PM01 BVH/SMPL-X IK profiles.

```bash
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects
# Model + policy + neutral PICO reference only:
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects --skip-motions
```

The model, ONNX and motions are local ignored runtime artifacts. The frozen spec
preserves training defaults, gains, limits, armatures and signed wrist actions.
The source spec rounds world joint axes to four decimals, so model verification
uses an explicit `1e-4` tolerance (measured maximum `8.76e-5`); no joint-limit
overrides were needed. Vendor mass differs from the rounded spec by 0.000189 kg.
`third_party/engineai_pm01/validation.json` records source and output hashes.

`checkpoints/proteus/wsq61/engineai_pm01/` contains one complete `policy.onnx`,
the verbatim source contract, comparison report, `policy.yaml` for hardware and
`policy-sim.yaml` for measured-root simulation. Inputs are
`proprioception[10,68]`, `task[6,253]`, `action_history[10,31]`, and
`history_valid[10]`; outputs are `action[24]` and `next_slot_action[31]`.
The 10-step history is oldest first. All 64 seed-0 random comparisons against the
released actor had zero action error. Export uses IR 8/opset 17.

PM01 exports include explicit `soft_joint_pos_limits` for all 24 joints, using
the verified model bounds. These clip both policy targets and the first-frame
shaping target. If an older YAML raises `shaping_target: motion_first_frame ...
set them in the policy YAML`, regenerate it with the updated exporter:

```bash
uv run --no-sync python -m scripts.export_proteus_onnx \
  --contract engineai_pm01 --robot pm01 --release wsq61 \
  --proteus-root ../Proteus-wamseq
```

## PC-to-robot setup

Official SSH endpoints are `user@192.168.0.163` (motion computer) and
`ubuntu@192.168.0.162` (Orin); see the
[vendor connection configuration](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/scripts/sync_src.sh).
The PC runs the policy over Ethernet; no inference installation on Orin is required.

Build the **installed robot's** ROS messages on the PC. This only copies message
definitions and compiles locally; it does not change or start anything on the robot.

```bash
cd /path/to/sim2real
bash scripts/pm01/prepare_ros.sh user@192.168.0.163
source scripts/pm01/env.sh
uv run --no-sync python -m scripts.pm01.check_robot
```

`env.sh` selects ROS Humble, Cyclone DDS, domain 69, Ethernet interface `eno1`, the
local matching message workspace, and both Hugging Face offline exports.
Override `PM01_NETWORK_INTERFACE`, `PM01_ROS_DOMAIN_ID`, or `PM01_ROS_SETUP` as
needed before sourcing. An existing `CYCLONEDDS_URI` takes precedence.
`PM01_REMOTE_PROTOCOL` overrides the remote installed message-package path for
`prepare_ros.sh`. Do not mix incompatible interface overlays in one shell.

The check only subscribes. It exits 0 when it receives fresh 24-joint feedback,
IMU and `joint_bridge` mode, finds a compatible command subscriber, and finds no
other command publisher; otherwise it prints the missing conditions and exits 2.
An inactive service is expected to fail this check.
`ready: true` verifies ROS communication and mode ownership only. These messages
do not expose motor enable status, so it does not prove the actuators are enabled
or following commands.

## Run on hardware

Keep the PM01 supported by a gantry with its hardware stop available. Computer
power and motor power are separate. For the PM01 supplied with a separate white
emergency-stop/power-on button, the [official user manual, pages 6 and 8](https://cn-engineai-1304599088.cos.ap-guangzhou.myqcloud.com/uploads/upload/files/20251031/c15bec68ae58e2f5cd2901aad04e0750.pdf#page=6)
instructs pressing that white button after the computer has booted. Two short
beeps and the rear/hip white light turning on indicate joint power is enabled.
Do not use the red neck emergency stop to enable power, or repeatedly press the
white button after successful power-on; it can also cut joint power.

Start the existing vendor service on the motion computer if it is inactive:

```bash
ssh user@192.168.0.163
sudo systemctl start robotics.service
systemctl status monitor.service robotics.service --no-pager
```

On the tested PM01 firmware, `monitor.service` runs a motor check before
`robotics.service`. With motor power off, the check can wait indefinitely and
`systemctl start` remains blocked. Check `systemctl list-jobs` and
`journalctl -b -u monitor.service --no-pager` in a second SSH session. Once motor
power is enabled, the pending start can continue automatically; there is no need
to submit it again or bypass the motor check. Healthy startup leaves the monitor
`active (exited)` and robotics `active (running)`.

Motor power alone does not enable control. With the external policy stopped,
gantry support secured and people clear, use **LB + Start** for the vendor startup
(`idle`) sequence, then **LB + RB** for `passive` before `pd_stand`. If already in
`joint_bridge`, first return to `passive` with LB + RB: this release only allows
entry to `idle` from `passive`. The installed IdleRunner requests motor enable
when leaving `idle`; entering `pd_stand`/`joint_bridge` alone does not perform that
startup handshake. A read-only inspection during a no-motion incident found all
24 motor disable flags set even with motor power on and `joint_bridge` reported.
Verify enable before restarting the external policy. Do not restore enable while
a policy is running with targets accumulated against an unmoving robot.

On the tested onboard release `76445d09` (2025-10-16), the loaded
`robot_release` mode uses `task_motion/release`. Its controller mappings are:

| Buttons | Target mode |
| --- | --- |
| LB + Start | `idle` startup / motor-enable handshake |
| LB + RB | `passive` |
| LB + A | `pd_stand` |
| LB + D-pad left | `joint_bridge` (`CROSS_Y_LEFT`) |

The newer vendor ROS2 documentation lists BACK + A for joint bridge, but that
combination does not trigger a transition on this release. Keep gantry support
and verify each transition with `/motion/motion_state`; this firmware reports
`current_motion_task_name`. The launchers do not switch robot modes.

If `pd_stand` returns to `passive`, inspect the service log. This release rejects
entry when a joint differs from its standing target by more than 1.2 rad. With
gantry support and the controller left in `passive`, correct a folded or twisted
starting posture without forcing joints, then retry after hands are clear. If
the physical posture looks correct but reported angles disagree, investigate
feedback/calibration instead of bypassing the entry check.

Back on the PC:

```bash
cd /path/to/sim2real
bash scripts/pm01/start_proteus.sh \
  checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

The launcher sources the PM01 environment and runs the read-only check before
opening the command transport. Default inference is CPU at 50 Hz; the independent
ROS sender runs at 500 Hz. Override the policy with `PROTEUS_POLICY_CONFIG`.

Keys: **i** shapes to the motion's first frame over 10 seconds; **]** enables
policy Hold; **Space** plays/pauses; **o** enters damping; **Ctrl+C** exits.
Keep gantry support through shaping and the initial Hold. Starting the transport
in `joint_bridge` sends damping before shaping, so it does not support a standing
robot by itself. An explicit mode change gives ownership back to the firmware.

The sender maps joints by name, converts serial ankle commands via the classic parser (`parallel_parser_type=0`),
and rejects wrong dimensions/nonfinite commands. Commands or joint/IMU feedback
older than 250 ms, a mode heartbeat older than 1 s, or a latched error produce
zero stiffness/velocity/feedforward torque with damping 2. Position increments
are limited to 0.2 rad per publication. Normal shutdown repeats damping commands;
process death or power loss requires firmware-side protection.

## PICO

```bash
# PC terminal 1: XRoboToolkit PC Service and calibrated body/leg trackers
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# PC terminal 2: robot policy; publisher runs on this same PC
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

For simulation, use `start_pico.sh sim` and `start_pico.sh policy-sim 127.0.0.1`.
**A** shapes to neutral; release it, then **A+B** enables Hold; **X** plays/pauses;
**B alone** enters damping. Motion/button TCP ports are 28701/5592. Only one robot
uses the default ports at a time. Live XR scaling accounts for operator height;
IK limits joint speed to 12 rad/s. Missing future samples are padded with received
references. Losing the body stream holds the last reference while feedback stays
healthy; it does not automatically enter damping. No new hand driver is included.

## Reproduce validation

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
```

Seed 0, policy 50 Hz, physics 0.000625 s, one-second initial and final policy holds,
and eight-second real LAFAN1 excerpts retargeted to PM01. Each completed with finite
states and per-frame mean tracked-body error below 0.5 m. This is simulation evidence.

| Motion | Mean world body error (cm) | Mean local body error (cm) | World root endpoint error (cm) |
| --- | ---: | ---: | ---: |
| aiming1_subject1 | 2.60 | 1.71 | 3.97 |
| dance2_subject1 | 3.00 | 1.98 | 2.29 |
| fightAndSports1_subject4 | 4.77 | 2.01 | 7.01 |
| walk1_subject1 | 2.30 | 1.85 | 2.42 |

See `docs/experiments/proteus-pm01.json` for exact commands, hashes and trajectories.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m pytest \
  tests/test_pm01.py tests/test_proteus_transport.py tests/test_proteus_observations.py \
  tests/test_robot_io.py tests/test_command_sender.py -q
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py -q -k pm01
# ROS loopback test: isolates itself on localhost, domain 171, never the robot.
source scripts/pm01/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py -q
# Full launcher, keyboard shaping/hold/play/damping, inference and recording.
# Requires prepared assets; forces loopback/domain 171, with synthetic feedback.
PM01_RUN_LAUNCH_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
  python -m pytest tests/test_pm01_launch.py -q -s
```
