#!/usr/bin/env python3
"""Rehearse the X2 onboard procedure on the PC: the real aimdk_bridge + Tracking processes, MuJoCo for the robot.

The AimDK HAL is replaced by MuJoCo (PD torques from the bridge's commands at 500 Hz, pelvis IMU from the free
joint) with a virtual gantry: a stiff pelvis spring until the operator lowers it. The operator presses come from
OperatorKeyPublisher exactly as scripts/et1/operator_keyboard.py sends them:

  a (stand / shaping) -> a (2 s reference transition, Hold) -> lower gantry -> s (play) -> space (stop)
  --lower-before-hold: a (shaping) -> lower gantry -> a (Hold on the floor) -> s -> space

  python scripts/x2/bridge_sim_test.py --policy-dir checkpoints/proteus/proteus_wamseq_62/lingxi_x2 \
      --motion outputs/x2/lafan/11_stand_walk_straight --play-s 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sim2real.config.robots import X2_CFG  # noqa: E402
from sim2real.rl_policy.robot_io.base import RobotState  # noqa: E402
from sim2real.rl_policy.robot_io.x2 import X2StateCommand  # noqa: E402
from sim2real.sim_env.utils.mjcf import load_sim_model  # noqa: E402
from sim2real.utils.operator_keys import KEY_A, KEY_SELECT, KEY_START, OperatorKeyPublisher  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts/x2'))
import aimdk_bridge  # noqa: E402

SIM_DT = 0.002


class MujocoHal:
    """Stands in for AimdkHal: same read()/publish() as the bridge uses."""

    def __init__(self, cfg, start_qpos=None, gantry_z_offset=0.02, gantry_gravity=None):
        self.model = load_sim_model(cfg)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.codec = X2StateCommand(cfg, timeout_s=1.0)
        names = cfg.joint_names
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in names])
        self.vadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in names])
        self.effort = np.array([cfg.joint_effort_limit[n] for n in names])
        self.data.qpos[:7] = cfg.default_qpos[:7]
        self.data.qpos[self.qadr] = cfg.default_qpos[7:] if start_qpos is None else start_qpos
        self.gantry_z = float(cfg.default_qpos[2]) + gantry_z_offset
        # Orientation the gantry holds the pelvis at: upright, or tilted so the IMU sees `gantry_gravity`.
        self.gantry_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if gantry_gravity is not None:
            from scipy.spatial.transform import Rotation
            g = np.asarray(gantry_gravity, float) / np.linalg.norm(gantry_gravity)
            rot, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [g])  # body-frame gravity -> world down
            x, y, z, w = rot.as_quat()
            self.gantry_quat = np.array([w, x, y, z])
        self.data.qpos[3:7] = self.gantry_quat
        self.data.qpos[2] = self.gantry_z
        mujoco.mj_forward(self.model, self.data)
        self.pelvis = self.model.body('pelvis').id
        self.cmd = None
        self.gantry = True
        self.lock = threading.Lock()
        self.stamp = 0.0
        self.published = 0
        self.stop = threading.Event()
        self.wall_s = 0.0
        self.resets = 0
        self.min_height = np.inf
        self.trace = []  # (sim time, pelvis z, gantry) at 50 Hz
        self.max_ang_vel = 0.0  # pelvis angular speed while on the gantry, after `track_ang_vel` is set
        self.track_ang_vel = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # Physics follows the wall clock: each pass runs every step that is due (PD per step), so a slow pass
        # never stretches simulated time.
        start = time.monotonic()
        while not self.stop.is_set():
            due = min(int((time.monotonic() - start) / SIM_DT) - round(self.data.time / SIM_DT), 50)
            if due <= 0:
                time.sleep(SIM_DT / 2)
                continue
            for _ in range(due):
                self._step()
        self.wall_s = time.monotonic() - start

    def _step(self):
        if True:
            with self.lock:
                d = self.data
                d.qfrc_applied[:] = 0.0
                d.xfrc_applied[self.pelvis] = 0.0
                if self.cmd is not None:
                    q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
                    tau = self.cmd.kp * (self.cmd.q_target - q) + self.cmd.kd * (self.cmd.dq_target - dq) + self.cmd.tau_ff
                    d.qfrc_applied[self.vadr] = np.clip(tau, -self.effort, self.effort)
                if self.gantry:  # pelvis spring-damper to the hanging pose, upright
                    pos, vel = d.qpos[:3], d.qvel[:3]
                    force = 4000.0 * (np.array([0.0, 0.0, self.gantry_z]) - pos) - 300.0 * vel
                    force[2] += self.model.body_subtreemass[self.pelvis] * 9.81
                    d.xfrc_applied[self.pelvis, :3] = force
                    # Upright spring-damper, soft enough for explicit integration at SIM_DT.
                    tilt = np.zeros(3)
                    mujoco.mju_subQuat(tilt, self.gantry_quat, d.qpos[3:7])
                    world_w = np.zeros(3)
                    mujoco.mju_rotVecQuat(world_w, d.qvel[3:6], d.qpos[3:7])
                    d.xfrc_applied[self.pelvis, 3:] = 150.0 * tilt - 10.0 * world_w
                before = d.time
                mujoco.mj_step(self.model, d)
                if d.time < before:
                    self.resets += 1  # MuJoCo reset the data after a numerical blow-up
                self.stamp = d.time
                if self.gantry and self.track_ang_vel:
                    self.max_ang_vel = max(self.max_ang_vel, float(np.linalg.norm(d.qvel[3:6])))
                if not self.gantry:
                    self.min_height = min(self.min_height, float(d.qpos[2]))
                if round(d.time / SIM_DT) % 10 == 0:
                    self.trace.append((d.time, float(d.qpos[2]), self.gantry))

    def read(self):
        with self.lock:
            n = len(self.qadr)
            qpos = np.zeros(7 + n, np.float32)
            qvel = np.zeros(6 + n, np.float32)
            qpos[3:7] = self.data.qpos[3:7]
            qpos[7:] = self.data.qpos[self.qadr]
            qvel[3:6] = self.data.qvel[3:6]  # free-joint angular velocity is in the body frame
            qvel[6:] = self.data.qvel[self.vadr]
            return RobotState(qpos=qpos, qvel=qvel, joint_torque=np.zeros(n, np.float32), tick=0), self.stamp

    def publish(self, cmd):
        with self.lock:
            q = np.clip(cmd.q_target, self.codec.lower, self.codec.upper)
            self.cmd = type(cmd)(q, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd)
            self.published += 1

    def height(self):
        with self.lock:
            return float(self.data.qpos[2])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy-dir', default='checkpoints/proteus/proteus_wamseq_62/lingxi_x2')
    p.add_argument('--policy-file', default='policy-pc-damped.yaml', help='config in --policy-dir, as on the robot')
    p.add_argument('--motion', default='outputs/x2/pico_hold')
    p.add_argument('--play-s', type=float, default=8.0)
    p.add_argument('--start-pose', choices=['default', 'hanging'], default='hanging',
                   help='hanging: arms/legs slightly off the stand pose, as on a gantry')
    p.add_argument('--out', type=Path, default=Path('outputs/x2/rehearsal'))
    p.add_argument('--gantry-z-offset', type=float, default=0.02,
                   help='pelvis height above the standing height while hanging (feet this far off the floor)')
    p.add_argument('--gantry-gravity', type=lambda s: [float(v) for v in s.split(',')],
                   help='pelvis-frame gravity the gantry holds, e.g. 0.22,0.10,-0.97 as hung on the real X2')
    p.add_argument('--hold-on-gantry-s', type=float, default=3.0, help='policy Hold time before lowering')
    p.add_argument('--lower-before-hold', action='store_true',
                   help='lower the gantry after shaping and press the second a with the feet on the floor')
    args = p.parse_args()
    sys.setswitchinterval(0.0005)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = replace(X2_CFG, low_state_bind_addr='127.0.0.1', low_cmd_host='127.0.0.1')
    start = None
    if args.start_pose == 'hanging':
        start = np.asarray(X2_CFG.default_qpos[7:], dtype=float).copy()
        rng = np.random.default_rng(0)
        start += rng.uniform(-0.15, 0.15, start.shape)
        start = np.clip(start, [X2_CFG.joint_pos_lower_limit[n] for n in X2_CFG.joint_names],
                        [X2_CFG.joint_pos_upper_limit[n] for n in X2_CFG.joint_names])
    hal = MujocoHal(cfg, start, args.gantry_z_offset, args.gantry_gravity)
    bridge = aimdk_bridge.X2Bridge(cfg, hal, rate_hz=500.0, timeout_s=0.2, takeover_cmd='true')  # max_kd 25 as on the robot
    bridge_thread = threading.Thread(target=bridge.run, kwargs=dict(state_log=str(args.out / 'bridge-state.json')),
                                     daemon=True)
    bridge_thread.start()
    env = dict(os.environ, SIM2REAL_IO_TIMEOUT_S='0.2', SIM2REAL_ORT_NUM_THREADS='1', OMP_NUM_THREADS='1')
    log = open(args.out / 'tracker.log', 'w')
    tracker = subprocess.Popen([sys.executable, '-u', 'sim2real/rl_policy/tracking.py', '--robot', 'x2',
                                '--robot-io', 'zmq', '--inference-backend', 'onnx-cpu', '--controller',
                                'bridge_joystick', '--policy-config', f'{args.policy_dir}/{args.policy_file}',
                                '--motion-path', args.motion, '--record', '--record-output',
                                str(args.out / 'tracking.npz')], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    op = OperatorKeyPublisher()
    events = []

    def note(text):
        events.append((round(time.monotonic() - t0, 2), round(float(hal.data.time), 2), text, round(hal.height(), 3),
                       bridge.guard.reason))
        e = events[-1]
        print(f'{e[0]:7.2f}s (sim {e[1]:6.2f}s)  {text:34s} pelvis z={e[3]:.3f}  bridge: {e[4]}', flush=True)

    t0 = time.monotonic()
    try:
        deadline = time.monotonic() + 120
        while not bridge._tracker_ready_logged:
            if tracker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Tracker did not become ready; see tracker.log')
            time.sleep(0.2)
        note('Tracker and LowState ready')
        assert hal.published == 0, 'bridge wrote the HAL before the operator A'
        op.press(KEY_A)
        note('a: takeover + stand (shaping)')
        time.sleep(12)
        note('shaping done')
        if args.lower_before_hold:
            # Lower until the feet are flat on the floor with the rope still taut (the robot hangs upright, the
            # rope carries part of the weight), then let the policy take over and release the rope.
            with hal.lock:
                hal.gantry_z = float(X2_CFG.default_qpos[2]) - 0.01
                hal.gantry_quat = np.array([1.0, 0.0, 0.0, 0.0])
            note('lowered: feet on the floor, rope taut')
            time.sleep(3)
            note('standing on the floor under shaping')
            op.press(KEY_A)
            note('a: reference transition')
            hal.track_ang_vel = True
            time.sleep(args.hold_on_gantry_s)
            hal.track_ang_vel = False
            note('Hold, releasing the rope')
            hal.gantry = False
            time.sleep(2)
            note('standing on its own')
        else:
            op.press(KEY_A)
            note('a: reference transition')
            hal.track_ang_vel = True
            time.sleep(args.hold_on_gantry_s)
            hal.track_ang_vel = False
            note('Hold ready, lowering gantry')
            hal.gantry = False
            time.sleep(2)
            note('standing on its own')
        op.press(KEY_START)
        note('s: play')
        time.sleep(args.play_s)
        note('motion done')
        hal.gantry = True  # the gantry rope takes the robot when the stop drops posture support
        op.press(KEY_SELECT)
        time.sleep(1)
        note('space: stop latched')
    finally:
        op.close()
        tracker.terminate()
        tracker.wait(timeout=20)
        hal.stop.set()
        hal.thread.join(timeout=2)
        log.close()
    fell = hal.min_height < 0.6 * float(X2_CFG.default_qpos[2]) or hal.resets > 0
    report = dict(policy_dir=args.policy_dir, policy_file=args.policy_file, motion=args.motion, start_pose=args.start_pose, events=events,
                  min_pelvis_height_after_gantry_m=hal.min_height, fell=bool(fell), hal_publishes=hal.published,
                  bridge_final_reason=bridge.guard.reason, gantry_z_offset=args.gantry_z_offset,
                  gantry_gravity=args.gantry_gravity, lower_before_hold=args.lower_before_hold, max_pelvis_ang_vel_on_gantry=hal.max_ang_vel, takeover_s=bridge.takeover_s,
                  physics_resets=hal.resets, realtime_factor=float(hal.data.time / max(hal.wall_s, 1e-9)))
    np.save(args.out / 'pelvis_trace.npy', np.array(hal.trace, dtype=float))
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, indent=2))
    return 1 if fell else 0


if __name__ == '__main__':
    raise SystemExit(main())
