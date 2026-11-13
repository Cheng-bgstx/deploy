"""Run Proteus against live PM01 feedback without creating command publishers.

The same Tracking/observation/ONNX/action path runs, but write_command only checks
and records arrays in memory. This is not a closed-loop dynamics test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.robot_io.base import RobotIO
from sim2real.rl_policy.robot_io.pm01 import PM01Codec, PM01_SDK_JOINT_NAMES
from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec
from sim2real.rl_policy.tracking import Tracking, TrackingArgs


class SnapshotCodec(PM01Codec):
    def _fresh(self):
        # Read-only observation is allowed in native passive/idle modes too.
        return NamedJointCodec._fresh(self)


class ReadOnlyIO(RobotIO):
    def __init__(self, cfg):
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from interface_protocol.msg import JointState, ImuInfo, MotionState
        self.codec = SnapshotCodec(cfg, {'body': PM01_SDK_JOINT_NAMES})
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node('pm01_read_only_policy', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(JointState, '/hardware/joint_state',
                                     lambda m: self.codec.receive_group('body', m), qos)
        self.node.create_subscription(ImuInfo, '/hardware/imu_info', self.codec.receive_imu, qos)
        self.node.create_subscription(MotionState, '/motion/motion_state', self.codec.receive_motion_state, qos)
        self.commands = 0
        self.mode = 'zero'
        self.last_q = None

    def read_state(self):
        for _ in range(8):
            self.executor.spin_once(timeout_sec=0.)
        return self.codec.read_state()

    def set_control_mode(self, mode, generation):
        self.mode = mode

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        arrays = [np.asarray(a) for a in (q_target, dq_target, tau_ff, kp, kd)]
        if any(a.shape != (24,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid shadow command')
        self.commands += 1
        self.last_q = np.asarray(q_target).copy()

    def close(self):
        self.executor.shutdown(timeout_sec=1.)
        self.node.destroy_node()
        self.context.shutdown()


class NoInput(ControllerBase):
    def get_control_mode(self):
        return None


class ShadowTracking(Tracking):
    def _create_robot_io(self):
        return ReadOnlyIO(self.robot_cfg)

    def _build_controller(self):
        self.keyboard_controller = self.joystick_controller = self.pico_controller = None
        return NoInput()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = ShadowTracking(TrackingArgs(robot='pm01', robot_io='inline', controller='keyboard',
        policy_config=str(args.policy_config), inference_backend='onnx-cpu', rl_rate=50.,
        motion_backend='npz', motion_path=str(args.motion), record=True, record_output=str(args.output)))
    policy.state_dict = {'action': np.zeros(policy.num_actions, np.float32),
                         'paused': True, 'control_mode': 'zero'}
    policy.total_inference_cnt = 0
    policy.perf_dict = {}
    phases = [(0, 'zero'), (25, 'init'), (550, 'policy'), (650, 'play'), (750, 'zero')]
    try:
        until = time.monotonic()+5.
        while policy.robot_io.read_state() is None and time.monotonic() < until:
            time.sleep(.01)
        if policy.robot_io.read_state() is None:
            raise RuntimeError('No fresh real feedback')
        step_ms = []
        for step in range(775):
            deadline = time.monotonic()+.02
            for at, phase in phases:
                if at == step:
                    if phase == 'play': policy.toggle_paused(source='read-only test')
                    elif phase == 'init': policy.set_init_mode(source='read-only test')
                    elif phase == 'policy': policy.set_policy_mode(source='read-only test')
                    else: policy.set_zero_mode(source='read-only test')
            begin = time.perf_counter()
            policy.step()
            step_ms.append((time.perf_counter() - begin) * 1e3)
            policy.total_inference_cnt += 1
            time.sleep(max(0., deadline-time.monotonic()))
        if policy.robot_io.commands != 775:
            raise RuntimeError(f'Incomplete inference: {policy.robot_io.commands}/775 frames')
        # Whole Tracking step (observation, ONNX, action) against the 20 ms control period; the policy
        # phase (steps 550+) is the one that runs the actor every step.
        active = np.asarray(step_ms[550:])
        timing = {'policy_step_ms_p50': round(float(np.percentile(active, 50)), 2),
                  'policy_step_ms_p99': round(float(np.percentile(active, 99)), 2),
                  'policy_step_ms_max': round(float(active.max()), 2),
                  'steps_over_20ms': int((np.asarray(step_ms) > 20.).sum())}
        result = {'read_only': True, 'motor_commands_sent': 0, 'frames': policy.robot_io.commands, **timing,
                  'motion_mode': policy.robot_io.codec.motion_mode,
                  'command_publishers': [p.node_name for p in policy.robot_io.node.get_publishers_info_by_topic('/hardware/joint_command')]}
        print(json.dumps(result, indent=2))
        args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    finally:
        try:
            policy._save_recording()
        finally:
            policy.controller.close()
            policy.robot_io.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--motion', type=Path, default=Path('checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
