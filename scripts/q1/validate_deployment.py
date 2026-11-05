"""Offline Q1 rehearsal of Tracking, named-joint commands and an ideal fixed-root support.

No ROS context, publisher or real robot connection is created. This checks the
Shaping -> Hold transition; it does not validate IMU mounting or gantry dynamics.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import tyro
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec, NamedRosRobotIO
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.sim_env.integrated_sim2sim import IntegratedSimRuntime
from sim2real.sim_env.utils.mjcf import load_sim_model


class Operator(ControllerBase):
    name = 'offline keyboard'
    requested = None

    def get_control_mode(self):
        value, self.requested = self.requested, None
        return value


class OfflineIO(NamedRosRobotIO):
    def __init__(self, sim, cfg, wire_names, **codec_options):
        # Deliberately bypass the ROS constructor; exercise the real codec and
        # keyword-command path with MuJoCo feedback in the vendor wire layout.
        self.sim, self.cfg = sim, cfg
        self.codec = NamedJointCodec(cfg, {'body': cfg.joint_names}, damping=5., allow_init_from_measured=True,
                                     **codec_options)
        self._armed = False
        self._preserve_feedback_order = True
        self.wire_names = list(wire_names)
        self._sequence = 0
        self._array_type = lambda: NS(header=NS(stamp=None), joints=[])
        self._joint_type = lambda: NS()
        self._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: self.tick)))
        self._publishers = {'body': NS(publish=self.capture_command)}
        self.last_command = None
        self.tick = 0

    def capture_command(self, message):
        # Emulate a HAL that consumes array slots: the wire layout must agree
        # with feedback even when joint.name is present. Do not reorder by name
        # at this receiver, which hid the crossed-leg fault in the first rehearsal.
        assert [j.name for j in message.joints] == self.wire_names, 'Q1 wire slot mismatch'
        rows = np.array([[j.position, j.velocity, j.effort, j.stiffness, j.damping] for j in message.joints]).T
        self.last_command = rows[:, [self.wire_names.index(n) for n in self.cfg.joint_names]]

    def read_state(self):
        self.tick += 1
        stamp = NS(stamp=NS(sec=0, nanosec=self.tick))
        data = self.sim.mj_data
        joints = {n: NS(name=n, position=data.qpos[qa], velocity=data.qvel[va], effort=0.)
                  for n, qa, va in zip(self.cfg.joint_names, self.sim.qpos_adrs, self.sim.qvel_adrs)}
        self.codec.receive_group('body', NS(header=stamp, joints=[joints[n] for n in self.wire_names]))
        q, w = data.qpos[3:7], data.qvel[3:6]
        self.codec.receive_imu(NS(header=stamp, orientation=NS(w=q[0], x=q[1], y=q[2], z=q[3]),
                                  angular_velocity=NS(x=w[0], y=w[1], z=w[2])))
        return super().read_state()

    def close(self):
        pass


class OfflineTracking(Tracking):
    def __init__(self, args, io):
        self.offline_io = io
        super().__init__(args)

    def _create_robot_io(self):
        return self.offline_io

    def _build_controller(self):
        self.keyboard_controller = self.joystick_controller = self.pico_controller = None
        return Operator()


@dataclass
class Args:
    policy_config: str = 'checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml'
    motion_path: str = 'checkpoints/primebot_q1/motions/motions/walk1_subject1.npz'
    initial_feedback: Path | None = None
    """Optional ros-readonly.json containing the measured /aima/hal/joint/state snapshot."""
    initial_tracking_record: Path | None = None
    """Replay initial joint pose and supported root orientation from a local Tracking record."""
    use_recorded_orientation: bool = True
    """False keeps the motion's root orientation while replaying the recorded joint pose."""
    output: Path = Path('outputs/q1/deployment-rehearsal')


def add_gantry_constraint(sim):
    # Resetting a free root's pose each substep does NOT supply a support force:
    # its instantaneous free-fall acceleration incorrectly removes joint sag.
    # Use a real equality constraint against a movable, kinematic gantry body.
    def customize(spec):
        spec.worldbody.add_body(name='q1_gantry', mocap=True)
        spec.add_equality(name='q1_gantry_weld', type=mujoco.mjtEq.mjEQ_WELD,
            name1='pelvis', name2='q1_gantry', objtype=mujoco.mjtObj.mjOBJ_BODY,
            data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=[.003, 1])
    sim.mj_model = load_sim_model(sim.robot_cfg, customize_spec=customize)
    sim.mj_model.opt.timestep = sim.sim_dt
    sim.mj_data = mujoco.MjData(sim.mj_model)
    sim._init_mappings()
    return sim.mj_model.equality('q1_gantry_weld').id, int(sim.mj_model.body('q1_gantry').mocapid[0])


def main(args):
    cfg = get_robot_cfg('q1')
    sim = IntegratedSimRuntime(cfg, sim_dt=.000625, headless=True, key_callback=None)
    weld_id, mocap_id = add_gantry_constraint(sim)
    feedback = None
    if args.initial_feedback:
        feedback = json.loads(args.initial_feedback.read_text())['latest']['/aima/hal/joint/state']
    # Default to a deliberately different wire layout; captured feedback supplies
    # the actual hardware layout when provided.
    wire_names = [j['name'] for j in feedback['joints']] if feedback else list(reversed(cfg.joint_names))
    # The hardware codec reads these from robot_io_options; mirror them here.
    io_options = yaml.safe_load(Path(args.policy_config).read_text()).get('robot_io_options') or {}
    limit_options = {k: io_options[k] for k in ('policy_limit_mode', 'limit_feedforward_band_rad') if k in io_options}
    io = OfflineIO(sim, cfg, wire_names, **limit_options)
    tracker = OfflineTracking(TrackingArgs(robot='q1', policy_config=args.policy_config,
        motion_path=args.motion_path, robot_io='inline', inference_backend='onnx-cpu'), io)
    tracker.total_inference_cnt = 0
    frame, _, bodies = tracker._motion_first_frame('offline Q1 rehearsal')
    root = bodies.index('pelvis')
    reference_root = frame.body_pos_w[0, 0, root].copy()
    quat = frame.body_quat_w[0, 0, root].copy()
    quat /= np.linalg.norm(quat)
    sim.mj_data.qpos[:] = cfg.default_qpos
    if feedback:
        positions = {j['name']: j['position'] for j in feedback['joints']}
        sim.mj_data.qpos[sim.qpos_adrs] = [positions[n] for n in cfg.joint_names]
    if args.initial_tracking_record:
        record = np.load(args.initial_tracking_record, allow_pickle=True)['data'].item()
        entry = int(np.flatnonzero(record['runtime']['control_mode'] == 'init')[0])
        names = record['robot']['joint_names']
        sim.mj_data.qpos[sim.qpos_adrs] = record['robot']['joint_pos'][entry, [names.index(n) for n in cfg.joint_names]]
        if args.use_recorded_orientation:
            quat = record['robot']['root_quat_w'][entry].copy()
            quat /= np.linalg.norm(quat)
    sim.mj_data.qpos[:7] = [reference_root[0], reference_root[1], reference_root[2] + .15, *quat]
    sim.mj_data.mocap_pos[mocap_id] = sim.mj_data.qpos[:3]
    sim.mj_data.mocap_quat[mocap_id] = quat
    initial_q = sim.mj_data.qpos[sim.qpos_adrs].copy()
    records, transitions = [], []
    last_mode = None
    first_command = None
    guarded_entry = tracker.policy_config.get('policy_entry_max_joint_error_rad') is not None
    try:
        for step in range(1000):
            t = step * .02
            if step == 0:
                # An early policy request tests the optional entry gate only.
                # Manual-entry configurations would immediately arm on this key.
                tracker.controller.requested = 'policy' if guarded_entry else None
            elif step == 1:
                tracker.controller.requested = 'init'
            elif (step == 20 and guarded_entry) or (
                    step in (520, 600, 650) and tracker.state_dict['control_mode'] == 'init'):
                tracker.controller.requested = 'policy'
            # The weld carries body weight and transmits joint reaction forces.
            # It still idealizes a real cable's compliance and attachment point.
            if t < 15.:
                height = reference_root[2] + .15 * (1 - np.clip((t-13.) / 1., 0, 1))
                sim.mj_data.mocap_pos[mocap_id] = [reference_root[0], reference_root[1], height]
            else:
                sim.mj_data.eq_active[weld_id] = False
            start = time.perf_counter()
            tracker.step()
            mode = tracker.state_dict['control_mode']
            if mode != last_mode:
                transitions.append({'time_s': t, 'mode': mode})
                last_mode = mode
            io._publish()
            if io.codec.error:
                raise RuntimeError('Offline Q1 wire serialization failed') from io.codec.error
            cmd = io.last_command
            if cmd is not None:
                if first_command is None:
                    first_command = cmd[0].copy()
                sim.apply_command(*cmd)
            if step > 0:
                for _ in range(32):
                    sim.sim_step()
            records.append({'time_s': t, 'mode': mode, 'root': sim.mj_data.qpos[:7].copy(),
                'joint_pos': sim.mj_data.qpos[sim.qpos_adrs].copy(),
                'target': cmd[0].copy() if cmd is not None else np.full(22, np.nan),
                'joint_error': float(np.max(np.abs(sim.mj_data.qpos[sim.qpos_adrs] - tracker.shaping_target))),
                'step_ms': (time.perf_counter()-start)*1000})
            tracker.total_inference_cnt += 1
        free = [r for r in records if r['time_s'] >= 15.]
        report = {
            'hardware_tested': False, 'support': 'mocap weld carrying body weight, lowered before release',
            'wire_receiver': 'consumes feedback-order slots; checks the real ROS serializer',
            'initial_feedback': str(args.initial_feedback), 'transitions': transitions,
            'initial_tracking_record': str(args.initial_tracking_record),
            'use_recorded_orientation': args.use_recorded_orientation,
            'entry_mode': 'guarded' if guarded_entry else 'manual',
            'policy_limit_mode': io.codec.policy_limit_mode,
            'joint_error_before_policy_rad': records[519]['joint_error'],
            'first_command_max_change_rad': float(np.max(np.abs(first_command-initial_q))),
            'joint_error_at_10_4_s_rad': records[520]['joint_error'],
            'finite': bool(all(np.isfinite(r['root']).all() and np.isfinite(r['joint_pos']).all() for r in records)),
            'min_free_root_height_m': float(min(r['root'][2] for r in free)),
            'min_free_upright': float(min(1-2*(r['root'][4]**2+r['root'][5]**2) for r in free)),
        }
        report['passed'] = bool(report['finite'] and [r['mode'] for r in transitions] == ['zero','init','policy']
            and report['first_command_max_change_rad'] < 1e-5
            and report['min_free_root_height_m'] > .3 and report['min_free_upright'] > .8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix('.npz'), **{k: np.asarray([r[k] for r in records]) for k in records[0]})
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        if not report['passed']:
            raise SystemExit(1)
    finally:
        sim.stop()
        tracker.controller.close()
        io.close()


if __name__ == '__main__':
    main(tyro.cli(Args))
