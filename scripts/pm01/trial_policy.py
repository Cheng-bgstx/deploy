"""Bounded candidate client with separately signaled policy and walk entry.

Creates a publisher only on /sim2real/pm01/request. It cannot command motors
without the separately supervised onboard policy_bridge, which selects and
enforces the hold/play deadlines. Walk requires two explicit session signals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from scripts.pm01.shadow_policy import ReadOnlyIO, ShadowTracking
from scripts.pm01.policy_bridge import REQUEST_TOPIC, STATUS_TOPIC, WALK_CLIENT_LIMIT_S
from sim2real.rl_policy.robot_io.pm01 import PM01_SDK_JOINT_NAMES
from sim2real.rl_policy.tracking import TrackingArgs


class RequestIO(ReadOnlyIO):
    def __init__(self, cfg):
        super().__init__(cfg)
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from interface_protocol.msg import JointCommand
        from std_msgs.msg import String
        self.message_type = JointCommand
        self.indices = [cfg.joint_names.index(n) for n in PM01_SDK_JOINT_NAMES]
        self.status = None
        self.playing = False
        self.shaped = False
        self.status_at = -np.inf
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        request_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.node.create_publisher(JointCommand, REQUEST_TOPIC, request_qos)
        def status(msg):
            data = json.loads(msg.data)
            if not isinstance(data.get('token'), str) or len(data['token']) != 32:
                raise ValueError('Invalid bridge session')
            self.status, self.status_at = data, time.monotonic()
        self.node.create_subscription(String, STATUS_TOPIC, status, qos)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        super().write_command(q_target, dq_target, tau_ff, kp, kd)
        if (not self.status or not self.status['ready'] or self.status.get('feedback_hold')
                or time.monotonic()-self.status_at > .1):
            return
        msg = self.message_type()
        ns = time.monotonic_ns()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        phase = ('play' if self.playing and self.mode == 'policy' else
                 'hold' if self.shaped and self.mode == 'init' else self.mode)
        msg.header.frame_id = self.status['token']+'/'+str(self.status['feedback_epoch'])+'/'+phase
        msg.position, msg.velocity, msg.feed_forward_torque, msg.stiffness, msg.damping = [
            np.asarray(a)[self.indices].astype(float).tolist() for a in (q_target, dq_target, tau_ff, kp, kd)]
        msg.torque = [0.]*24
        msg.parallel_parser_type = 1
        self.publisher.publish(msg)

    def resume_target(self):
        target = np.asarray(self.status['last_target_sdk'], dtype=float)
        if target.shape != (24,) or not np.isfinite(target).all():
            raise ValueError('Missing validated target for feedback recovery')
        result = np.empty(24)
        result[self.indices] = target
        return result


class TrialTracking(ShadowTracking):
    """Use the training inference path without experimental target transforms."""
    def __init__(self, args):
        config = yaml.safe_load(Path(args.policy_config).read_text())
        removed = {'soft_joint_pos_limits', 'max_target_velocity_rad_s',
                   'pm01_head_target_velocity_rad_s', 'pm01_arm_target_velocity_rad_s',
                   'pm01_policy_blend_seconds'} & config.keys()
        if removed:
            raise ValueError(f'Retired PM01 experiment configuration: {sorted(removed)}; '
                             'regenerate the original training policy with scripts.export_proteus_onnx')
        super().__init__(args)

    def _create_robot_io(self):
        return RequestIO(self.robot_cfg)



def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.walk_once:
        if (args.play_signal is None or args.play_signal.exists()
                or args.policy_signal is None or args.policy_signal.exists()
                or args.play_signal == args.policy_signal):
            raise ValueError('Walk requires distinct, new --policy-signal and --play-signal files')
        original = Path('checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz')
        if hashlib.sha256(args.motion.read_bytes()).digest() != hashlib.sha256(original.read_bytes()).digest():
            raise ValueError('Walk session requires the verified original walk motion')
    policy = TrialTracking(TrackingArgs(robot='pm01', robot_io='inline', controller='keyboard',
        policy_config=str(args.policy_config), inference_backend='onnx-cpu', rl_rate=50.,
        motion_backend='npz', motion_path=str(args.motion), record=True, record_output=str(args.output)))
    policy.state_dict = {'action': np.zeros(policy.num_actions, np.float32),
                         'paused': True, 'control_mode': 'zero'}
    policy.total_inference_cnt = 0
    policy.perf_dict = {}
    started, ready_at, init_steps = time.monotonic(), None, 0
    holding_at = None
    feedback_epoch = 0
    try:
        print('Candidate loaded; waiting for supervised onboard bridge', flush=True)
        while time.monotonic()-started < (WALK_CLIENT_LIMIT_S if args.walk_once else 45):
            deadline = time.monotonic()+.02
            policy.robot_io.read_state()
            status = policy.robot_io.status
            if status and status.get('reason'):
                print(json.dumps(status, indent=2), flush=True)
                if status['reason'] in ('completed first-frame hold', 'completed shaping probe', 'completed walk'):
                    return 0
                raise RuntimeError('Onboard bridge stopped: '+status['reason'])
            if status and status['ready'] and ready_at is None:
                ready_at = time.monotonic()
            if status and status.get('feedback_hold'):
                # Pause inference/reference advancement while the onboard motor
                # task retains the last validated target through a short gap.
                time.sleep(max(0., deadline-time.monotonic()))
                continue
            if status and status['feedback_epoch'] != feedback_epoch:
                feedback_epoch = status['feedback_epoch']
            if ready_at is not None and time.monotonic()-ready_at > .3 and policy.state_dict['control_mode'] == 'zero':
                policy.set_init_mode(source='bounded onboard trial')
            if init_steps >= 510 and policy.state_dict['control_mode'] == 'init':
                if args.walk_once and not policy.robot_io.shaped:
                    policy.robot_io.shaped = True
                    print('SHAPING_COMPLETE: holding PD pose; keep gantry support, verify foot contact before policy signal', flush=True)
                if not args.walk_once or args.policy_signal.is_file():
                    if args.walk_once and args.policy_signal.read_text().strip() != 'policy':
                        raise ValueError('Invalid policy signal contents')
                    policy.set_policy_mode(source='bounded first-frame hold')
                    holding_at = time.monotonic()
                    if args.walk_once:
                        print('POLICY_HOLD: original walk frame 0; waiting for the play barrier', flush=True)
            if (args.walk_once and holding_at is not None and not policy.robot_io.playing
                    and time.monotonic()-holding_at >= 4.
                    and status.get('stationary_hold_stable') and args.play_signal.is_file()):
                if args.play_signal.read_text().strip() != 'play':
                    raise ValueError('Invalid play signal contents')
                policy.toggle_paused(source='explicit guarded walk play signal')
                policy.robot_io.playing = True
                print('PLAYING: original walk once; onboard deadline remains active', flush=True)
            before = policy.robot_io.commands
            policy.step()
            if policy.state_dict['control_mode'] == 'init' and policy.robot_io.commands > before:
                init_steps += 1
            policy.total_inference_cnt += 1
            time.sleep(max(0., deadline-time.monotonic()))
        raise RuntimeError('Bounded policy client timed out')
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
    parser.add_argument('--walk-once', action='store_true')
    parser.add_argument('--play-signal', type=Path)
    parser.add_argument('--policy-signal', type=Path)
    raise SystemExit(run(parser.parse_args()))
