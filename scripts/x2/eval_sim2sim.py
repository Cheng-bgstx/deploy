#!/usr/bin/env python3
"""Headless X2 sim2sim evaluation of policy x motion, nominal or under one perturbation.

Uses the runtime's own IntegratedSim2Sim (observations, targets, PD at sim_dt) and reports, over reference
playback, the mean position error of the contract's tracked bodies (world and pelvis-local), the minimum root
height and whether the robot fell. Perturbations follow the Proteus test-motion checks: kp or kd x0.8 / x1.2,
masses x1.1, half floor friction, one control step of action delay, and 60 N lateral pelvis pushes.

  python scripts/x2/eval_sim2sim.py --policy-config checkpoints/proteus/proteus_60/lingxi_x2/policy-pc.yaml \
      --motion-path outputs/x2/lafan/11_stand_walk_straight --perturbation kp0.8 --out outputs/x2/eval/run.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401  # native library ordering on ARM
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs  # noqa: E402
from sim2real.utils.math import quat_rotate_inverse_numpy  # noqa: E402

PERTURBATIONS = ('nominal', 'kp0.8', 'kp1.2', 'kd0.8', 'kd1.2', 'mass1.1', 'friction0.5', 'delay1', 'push60',
                 'delay2', 'armkp0.5', 'armkp0.5+delay1', 'armkp0.5+delay2')
ARM_KEYS = ('shoulder', 'elbow', 'wrist')
FALL_HEIGHT_FRACTION = 0.6  # of the standing root height
PUSH_N, PUSH_S, PUSH_EVERY_S = 60.0, 0.2, 3.0


class PerturbedSim(IntegratedSim2Sim):
    def __init__(self, args, perturbation: str, cmd_delay_ms: float = 0.0, arm_torque_scale: float = 1.0):
        super().__init__(args)
        self.perturbation = perturbation
        model, sim = self.sim.mj_model, self.sim
        if perturbation == 'mass1.1':
            model.body_mass[:] *= 1.1
            model.body_inertia[:] *= 1.1
        elif perturbation == 'friction0.5':
            model.geom_friction[:, 0] *= 0.5
        kp_scale = {'kp0.8': .8, 'kp1.2': 1.2}.get(perturbation, 1.)
        kd_scale = {'kd0.8': .8, 'kd1.2': 1.2}.get(perturbation, 1.)
        # armkp0.5: the arm joints' kp and kd halved (the real X2 arms measured about half the commanded torque).
        arm = np.array([any(k in n for k in ARM_KEYS) for n in self.robot_cfg.joint_names])
        arm_scale = np.where(arm, 0.5, 1.0) if perturbation.startswith('armkp0.5') else np.ones(len(arm))
        # --arm-torque-scale: the arm PD torque the motors deliver, as a fraction of the commanded one.
        arm_scale = arm_scale * np.where(arm, arm_torque_scale, 1.0)
        pending = []  # (apply at sim time, command) for --cmd-delay-ms, applied at physics-step resolution
        delay = {'delay1': 1, 'delay2': 2, 'armkp0.5+delay1': 1, 'armkp0.5+delay2': 2}.get(perturbation, 0)
        apply = sim.apply_command
        previous = []

        def apply_command(q, dq, tau, kp, kd, *rest, **kw):
            command = (np.array(q), np.array(dq), np.array(tau), np.asarray(kp) * kp_scale * arm_scale,
                       np.asarray(kd) * kd_scale * arm_scale)
            if delay:  # in policy steps (20 ms)
                previous.append(command)
                command = previous.pop(0) if len(previous) > delay else previous[0]
            if cmd_delay_ms > 0:  # measured from the state the policy saw to the command reaching the motors
                pending.append((sim.mj_data.time + cmd_delay_ms / 1000 - 1e-9, command, rest, kw))
                return None
            return apply(*command, *rest, **kw)

        sim.apply_command = apply_command
        self.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        step = sim.sim_step

        def sim_step():
            while pending and pending[0][0] <= sim.mj_data.time:
                _, command, rest, kw = pending.pop(0)
                apply(*command, *rest, **kw)
            sim.mj_data.xfrc_applied[self.pelvis, :3] = 0.
            if perturbation == 'push60' and self.playback_started:
                t = sim.mj_data.time - self._push_t0
                k = int(t // PUSH_EVERY_S)
                if k >= 1 and t - k * PUSH_EVERY_S < PUSH_S:
                    angle = k * np.pi / 2  # +y, -x, -y, +x, ...
                    sim.mj_data.xfrc_applied[self.pelvis, :2] = PUSH_N * np.array([np.cos(angle + np.pi / 2),
                                                                                    np.sin(angle + np.pi / 2)])
            return step()

        sim.sim_step = sim_step
        self._push_t0 = 0.
        self.arm_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                 for n, a in zip(self.robot_cfg.joint_names, arm) if a])
        self.arm_speed_sq, self.arm_speed_n = 0.0, 0
        self.arm_qvel = []  # (time, arm qvel) during playback, for the oscillation frequency and growth
        leg = np.array([any(k in n for k in ('hip', 'knee', 'ankle', 'waist')) for n in self.robot_cfg.joint_names])
        self.leg_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                 for n, a in zip(self.robot_cfg.joint_names, leg) if a])
        self.leg_qvel = []
        step2 = sim.sim_step

        def sim_step_arm():
            out = step2()
            if self.playback_started:
                self.arm_speed_sq += float(np.square(sim.mj_data.qvel[self.arm_dof]).mean())
                self.arm_speed_n += 1
                self.arm_qvel.append((sim.mj_data.time, sim.mj_data.qvel[self.arm_dof].copy()))
                self.leg_qvel.append((sim.mj_data.time, sim.mj_data.qvel[self.leg_dof].copy()))
            return out

        sim.sim_step = sim_step_arm

    def _start_motion(self):
        super()._start_motion()
        self._push_t0 = self.sim.mj_data.time


def arm_oscillation(samples) -> dict:
    """Dominant arm-speed frequency above 1 Hz and how the arm speed grows: rms in the first vs last second."""
    if len(samples) < 50:
        return {}
    t = np.array([x[0] for x in samples])
    v = np.stack([x[1] for x in samples])
    dt = float(np.median(np.diff(t)))
    spec = np.abs(np.fft.rfft(v - v.mean(0), axis=0)).sum(1)
    f = np.fft.rfftfreq(len(v), dt)
    k = f > 1.0
    first, last = v[t < t[0] + 1.0], v[t > t[-1] - 1.0]
    return {'arm_osc_freq_hz': float(f[k][spec[k].argmax()]),
            'arm_speed_rms_first_s': float(np.sqrt(np.square(first).mean())),
            'arm_speed_rms_last_s': float(np.sqrt(np.square(last).mean()))}


def summarize(trajectory_path: Path, policy_config: str, stand_height: float) -> dict:
    tr = np.load(trajectory_path)
    cfg = yaml.safe_load(Path(policy_config).read_text())
    names = cfg['proteus']['body_names']
    all_names = tr['body_names'].tolist()
    idx = [all_names.index(n) for n in names]
    robot, motion = tr['robot_body_pos_w'][:, idx], tr['motion_body_pos_w'][:, idx]

    def local(body, prefix):
        quat = tr[f'{prefix}_root_quat_w']
        centered = body - tr[f'{prefix}_root_pos_w'][:, None]
        return quat_rotate_inverse_numpy(np.broadcast_to(quat[:, None], (*centered.shape[:2], 4)).copy(), centered)

    world = np.linalg.norm(robot - motion, axis=-1).mean(-1)
    loc = np.linalg.norm(local(robot, 'robot') - local(motion, 'motion'), axis=-1).mean(-1)
    height = tr['robot_root_pos_w'][:, 2]
    fell = bool(height.min() < FALL_HEIGHT_FRACTION * stand_height)
    return {'frames': int(len(world)), 'mean_world_error_m': float(world.mean()),
            'max_world_error_m': float(world.max()), 'mean_local_error_m': float(loc.mean()),
            'max_local_error_m': float(loc.max()), 'min_root_height_m': float(height.min()), 'fell': fell}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy-config', required=True)
    p.add_argument('--motion-path', required=True)
    p.add_argument('--perturbation', choices=PERTURBATIONS, default='nominal')
    p.add_argument('--duration', type=float, help='sim seconds (default: motion length + 2)')
    p.add_argument('--initial-pause-s', type=float, default=1.)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cmd-delay-ms', type=float, default=0.0,
                   help='extra delay from policy step to motor command, at physics-step resolution')
    p.add_argument('--arm-torque-scale', type=float, default=1.0, help='fraction of the arm PD torque delivered')
    p.add_argument('--sim-dt', type=float, help='physics step (default: the runtime default, 0.005 s)')
    args = p.parse_args()
    manifest = json.loads((Path(args.motion_path) / 'manifest.json').read_text())
    motion_files = sorted((Path(args.motion_path) / manifest.get('motions_subdir', 'motions')).glob('*.npz'))
    frames = len(np.load(motion_files[0])['qpos'])
    motion_s = frames * float(manifest['timestep'])
    duration = args.duration or args.initial_pause_s + motion_s + 1.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trajectory = args.out.with_suffix('.trajectory.npz')
    sim_args = IntegratedSim2SimArgs(robot='x2', policy_config=args.policy_config, motion_path=args.motion_path,
                                     headless=True, initial_pause_s=args.initial_pause_s, max_runtime_s=duration,
                                     seed=args.seed, trajectory_output=str(trajectory),
                                     root_trajectory_output=str(args.out.with_suffix('.root.npz')),
                                     **({'sim_dt': args.sim_dt} if args.sim_dt else {}))
    sim = PerturbedSim(sim_args, args.perturbation, args.cmd_delay_ms, args.arm_torque_scale)
    stand = float(sim.robot_cfg.default_qpos[2])
    sim.run()
    if hasattr(sim, 'save_trajectory'):
        sim.save_trajectory()
    report = {'policy_config': args.policy_config, 'motion': args.motion_path, 'perturbation': args.perturbation,
              'motion_s': motion_s, 'sim_s': duration, **summarize(trajectory, args.policy_config, stand),
              'arm_joint_speed_rms': float(np.sqrt(sim.arm_speed_sq / max(sim.arm_speed_n, 1))),
              'cmd_delay_ms': args.cmd_delay_ms, 'arm_torque_scale': args.arm_torque_scale, **arm_oscillation(sim.arm_qvel),
              **{k.replace('arm', 'leg'): v for k, v in arm_oscillation(sim.leg_qvel).items()}}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
