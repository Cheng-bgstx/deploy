"""No ROS: rehearse the bounded client/gate from a recorded hanging PM01 pose."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.pm01.policy_bridge import PolicyGate
from scripts.pm01.bench_damping import NATIVE_DAMPING
from scripts.pm01.trial_policy import TrialTracking
from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.rl_policy.tracking import TrackingArgs
from sim2real.sim_env.integrated_sim2sim import IntegratedSimRuntime


class SimIO(RobotIO):
    def __init__(self, sim, cfg, gate):
        self.sim, self.cfg, self.gate = sim, cfg, gate
        self.now, self.mode = 10., 'zero'
        self.commands = 0
        self.last_q = None
        self.playing = False
        self.shaped = False

    def read_state(self):
        n = len(self.cfg.joint_names)
        s = NS(root_pos_w=np.zeros(3), root_quat_w=np.zeros(4), root_lin_vel_w=np.zeros(3),
               root_ang_vel_b=np.zeros(3), joint_pos=np.zeros(n), joint_vel=np.zeros(n), joint_torque=np.zeros(n))
        self.sim.sync_policy_state(s)
        return RobotState(np.r_[np.zeros(3), s.root_quat_w, s.joint_pos],
                          np.r_[np.zeros(3), s.root_ang_vel_b, s.joint_vel], s.joint_torque, round(self.now*1000))

    def set_control_mode(self, mode, generation):
        self.mode = mode

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        s = self.read_state()
        self.commands += 1
        self.last_q = np.asarray(q_target).copy()
        phase = 'play' if self.playing else 'hold' if self.shaped and self.mode == 'init' else self.mode
        self.gate.accept(self.now, self.now, phase, [q_target, dq_target, tau_ff, kp, kd],
                         s.qpos[7:], s.qvel[6:])


class SimTracking(TrialTracking):
    def _create_robot_io(self):
        return self.provided_io


def main(args):
    cfg = get_robot_cfg('pm01')
    config = json.loads(args.bridge_config.read_text())
    gate = PolicyGate(config, 'offline', hold_seconds=args.hold_seconds,
                      shaping_only=10. if args.walk_shaping else False, walk_shaping=args.walk_shaping,
                      walk_once=args.walk_once)
    live = np.load(args.pose_record, allow_pickle=True)['data'].item()  # Locally generated trusted recording.
    names = live['robot']['joint_names']
    q = live['robot']['joint_pos'][-1, [names.index(n) for n in cfg.joint_names]]
    root = np.r_[0., 0., cfg.default_qpos[2]+.15, live['robot']['root_quat_w'][-1]]
    # A weld supplies the support reaction during the dynamics solve. Resetting
    # free-root qpos/qvel each step would still allow free-fall acceleration and
    # incorrectly unload the joints' gravity torques.
    source = cfg.resolve_mjcf_path()
    xml = ET.parse(source)
    ET.SubElement(xml.getroot().find('worldbody'), 'body', name='pm01_gantry_target', mocap='true',
                  pos=' '.join(map(str, root[:3])), quat=' '.join(map(str, root[3:])))
    equality = xml.getroot().find('equality')
    if equality is None:
        equality = ET.SubElement(xml.getroot(), 'equality')
    ET.SubElement(equality, 'weld', name='pm01_gantry_weld', body1='pm01_gantry_target',
                  body2='LINK_BASE', relpose='0 0 0 1 0 0 0', solref='.004 1', solimp='.99 .99 .001')
    with tempfile.NamedTemporaryFile(suffix='.xml', dir=source.parent) as temp:
        xml.write(temp.name)
        sim = IntegratedSimRuntime(replace(cfg, mjcf_path=Path(temp.name)),
                                   sim_dt=.002, headless=True, key_callback=None)
    sim.mj_data.qpos[:7] = root
    sim.mj_data.qpos[sim.qpos_adrs] = q
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    if args.feet_touch:
        # Place the lowest box corner of either foot at the plane while the
        # weld still carries the torso. This models light contact, not unloading
        # the gantry or a compliant physical sling.
        foot_ids = {sim.mj_model.body(n).id for n in ('LINK_FOOT_L', 'LINK_FOOT_R')}
        bottoms = []
        for i in range(sim.mj_model.ngeom):
            if sim.mj_model.geom_bodyid[i] in foot_ids and sim.mj_model.geom_contype[i]:
                if sim.mj_model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError('Unsupported foot collision geometry')
                r = sim.mj_data.geom_xmat[i].reshape(3, 3)
                bottoms.append(sim.mj_data.geom_xpos[i, 2]-abs(r[2])@sim.mj_model.geom_size[i])
        if not bottoms:
            raise ValueError('Missing foot collision geometry')
        root[2] -= min(bottoms)
        sim.mj_data.qpos[:7] = root
        sim.mj_data.mocap_pos[0] = root[:3]
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
    sim.apply_command(q, np.zeros(24), np.zeros(24), np.zeros(24), NATIVE_DAMPING)
    io = SimIO(sim, cfg, gate)
    SimTracking.provided_io = io
    policy = SimTracking(TrackingArgs(robot='pm01', robot_io='inline', controller='keyboard',
        policy_config=str(args.policy_config), inference_backend='onnx-cpu', rl_rate=50.,
        motion_backend='npz', motion_path=str(args.motion)))
    policy.state_dict = {'action': np.zeros(policy.num_actions, np.float32), 'paused': True, 'control_mode': 'zero'}
    policy.total_inference_cnt, policy.perf_dict = 0, {}
    frames = []
    release_step = 585+round(args.supported_policy_seconds/.02)
    play_step = release_step+100
    try:
        for step in range(play_step+650 if args.walk_once else 535+int((args.hold_seconds+1.)/.02)):
            io.now = 10.+step*.02
            if step == 25: policy.set_init_mode(source='offline bounded trial')
            if step == 535 and not args.walk_once: policy.set_policy_mode(source='offline bounded trial')
            if args.walk_once and step == 535:
                io.shaped = True
            if args.walk_once and step == 585: policy.set_policy_mode(source='offline feet-contact policy barrier')
            if args.walk_once and step == release_step:
                # Release the kinematic fixture, then transfer vertical load
                # over one second. This is an explicit idealized external sling
                # force, never a policy feed-forward or hardware command.
                sim.mj_data.eq_active[sim.mj_model.equality('pm01_gantry_weld').id] = 0
            if args.walk_once and step >= release_step:
                support = max(0., 1.-(step-release_step)/50.)
                sim.mj_data.xfrc_applied[sim.mj_model.body('LINK_BASE').id, 2] = (
                    support*sim.mj_model.body_mass.sum()*abs(sim.mj_model.opt.gravity[2]))
            if args.walk_once and step >= play_step and not io.playing and gate.hold_stable:
                policy.toggle_paused(source='offline grounded play barrier')
                io.playing = True
            policy.step()
            if gate.reason:
                break
            if gate.command is not None:
                sim.apply_command(*gate.command)
            for substep in range(10):
                sim.sim_step()
                s = io.read_state()
                now = io.now+(substep+1)*.002
                gate.observe(now, s.qpos[7:], s.qvel[6:], s.joint_torque, 0.)
                if args.walk_once:
                    quat = sim.mj_data.qpos[3:7]
                    tilt = np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1))
                    if tilt > .35 or sim.mj_data.qpos[2] < .6:
                        gate.stop('ground handoff tilt or height limit')
                frames.append(dict(time=now, mode=io.mode, q=s.qpos[7:].copy(), dq=s.qvel[6:].copy(),
                                   tau=s.joint_torque.copy(), root=sim.mj_data.qpos[:7].copy(),
                                   target=io.last_q.copy()))
                if gate.reason:
                    break
            policy.total_inference_cnt += 1
            if gate.reason:
                break
    finally:
        sim.stop()
        policy.controller.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {k: np.array([f[k] for f in frames]) for k in frames[0]}
    np.savez_compressed(args.output.with_suffix('.npz'), **data)
    stationary_quality = None
    if args.walk_once:
        settled = ((data['time'] >= 10.+585*.02+2.)
                   & (data['time'] < 10.+release_step*.02))
        if settled.sum() >= 500:
            p2p = np.ptp(data['q'][settled], axis=0)
            rms = np.sqrt(np.mean(data['dq'][settled]**2, axis=0))
            stationary_quality = dict(
                passed=bool(np.max(p2p) <= .12 and np.max(rms) <= .3),
                joint_names=cfg.joint_names,
                joint_peak_to_peak_rad=p2p.tolist(), joint_speed_rms_rad_s=rms.tolist(),
                peak_to_peak_limit_rad=.12, speed_rms_limit_rad_s=.3,
                scope='Whole-body settled supported first-frame hold; offline qualification, not a hardware rating')
    report = dict(reason=gate.reason,
        passed=(gate.reason == ('completed walk' if args.walk_once else
                               'completed shaping probe' if args.walk_shaping else 'completed first-frame hold')
                and (not args.walk_once or (stationary_quality is not None and stationary_quality['passed']))),
        stationary_quality=stationary_quality,
        max_joint_speed_rad_s=float(abs(data['dq']).max()),
        max_ankle_speed_rad_s=float(abs(data['dq'][:, [4,5,10,11]]).max()),
        max_joint_torque_Nm=float(abs(data['tau']).max()),
        max_ankle_torque_Nm=float(abs(data['tau'][:, [4,5,10,11]]).max()),
        max_root_position_error_m=float(np.linalg.norm(data['root'][:, :3]-root[:3], axis=1).max()),
        policy_config=str(args.policy_config), pose_record=str(args.pose_record),
        motion=str(args.motion), hold_seconds=args.hold_seconds, walk_shaping=args.walk_shaping,
        feet_touch=args.feet_touch, walk_once=args.walk_once, hardware_tested=False,
        supported_policy_seconds=args.supported_policy_seconds,
        limitation=(f'Idealized load transfer starts {args.supported_policy_seconds:g} seconds after policy entry, replacing the weld with vertical weight support ramped to zero over one second; ' if args.walk_once else 'Mocap weld supplies gantry reaction; ') + 'serial-joint dynamics, no physical cable or measured motor inertia')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-config', type=Path, required=True)
    p.add_argument('--pose-record', type=Path, required=True)
    p.add_argument('--bridge-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hold-seconds', type=float, choices=[1., 3.], default=1.)
    p.add_argument('--walk-shaping', action='store_true')
    p.add_argument('--feet-touch', action='store_true')
    p.add_argument('--walk-once', action='store_true')
    p.add_argument('--supported-policy-seconds', type=float, default=12.,
                   help='Offline-only policy hold under the gantry before load transfer (1 to 20 s)')
    p.add_argument('--motion', type=Path, default=Path('checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz'))
    args = p.parse_args()
    if not 1. <= args.supported_policy_seconds <= 20.:
        p.error('--supported-policy-seconds must be between 1 and 20')
    if args.walk_once and not args.feet_touch:
        p.error('--walk-once rehearsal requires --feet-touch')
    raise SystemExit(main(args))
