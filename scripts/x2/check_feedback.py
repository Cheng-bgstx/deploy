#!/usr/bin/env python3
"""Check X2 HAL feedback without creating command publishers or taking control."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from sim2real.config.robots import X2_CFG
from sim2real.config.robots.x2 import X2_AIMDK_GROUPS
from sim2real.rl_policy.robot_io.x2 import X2StateCommand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--policy-config', type=Path, default=Path(
        'checkpoints/proteus/wsq61/lingxi_x2/policy.yaml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-exclusive-control', action='store_true',
                        help='Fail if another node publishes HAL joint commands; does not take control.')
    parser.add_argument('--require-joint-limits', action='store_true',
                        help='Fail if measured joints exceed configured limits by more than 0.01 rad.')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive and finite')

    import rclpy
    import aimdk_msgs
    from aimdk_msgs.msg import JointState, JointStateArray
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu

    options = yaml.safe_load(args.policy_config.read_text()).get('robot_io_options', {})
    codec = X2StateCommand(X2_CFG, **{k: v for k, v in options.items()
                                   if k in {'timeout_s', 'head_pitch_enabled', 'imu_to_pelvis_wxyz'}})
    stats = {}

    def received(key, msg):
        now = time.monotonic()
        item = stats.setdefault(key, {'count': 0, 'first': now, 'last': now, 'max_gap_s': 0.})
        item['max_gap_s'] = max(item['max_gap_s'], now - item['last'])
        item['last'] = now
        item['count'] += 1
        if key == 'imu':
            codec.receive_imu(msg)
            item['frame_id'] = msg.header.frame_id
        else:
            item['joint_names'] = [j.name for j in msg.joints]
            group_state = getattr(msg, 'state', None)
            item['group_state'] = getattr(group_state, 'value', None)
            codec.receive_group(key, msg)

    rclpy.init()
    node = rclpy.create_node('sim2real_x2_feedback_check')
    subscriptions = []
    try:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for group in X2_AIMDK_GROUPS:
            subscriptions.append(node.create_subscription(
                JointStateArray, f'/aima/hal/joint/{group}/state',
                lambda msg, key=group: received(key, msg), qos))
        subscriptions.append(node.create_subscription(
            Imu, options.get('imu_topic', '/aima/hal/imu/torso/state'),
            lambda msg: received('imu', msg), qos))
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(.05, max(0., end - time.monotonic())))
        now = time.monotonic()
        error = None
        try:
            state = codec.read_state()
        except RuntimeError as exc:
            state = None
            error = str(exc.__cause__ or exc)
        report = {
            'feedback_valid': state is not None, 'error': error,
            'message_package': aimdk_msgs.__file__,
            'joint_state_fields': JointState.get_fields_and_field_types(),
            'joint_state_array_fields': JointStateArray.get_fields_and_field_types(),
            'command_publishers': {}, 'joint_limit_violations': {},
            'streams': {},
        }
        for group in X2_AIMDK_GROUPS:
            report['command_publishers'][group] = [
                {'name': info.node_name, 'namespace': info.node_namespace}
                for info in node.get_publishers_info_by_topic(f'/aima/hal/joint/{group}/command')]
        for key in [*X2_AIMDK_GROUPS, 'imu']:
            item = stats.get(key)
            if item is None:
                report['streams'][key] = {'count': 0}
                continue
            span = item['last'] - item['first']
            report['streams'][key] = {
                k: v for k, v in item.items() if k not in {'first', 'last'}}
            report['streams'][key].update(
                rate_hz=(item['count'] - 1) / span if span > 0 else 0.,
                age_s=now - item['last'])
        if state is not None:
            report.update(joint_position_rad=dict(zip(X2_CFG.joint_names, map(float, state.qpos[7:]))),
                          pelvis_quaternion_wxyz=state.qpos[3:7].tolist(),
                          pelvis_gyro_rad_s=state.qvel[3:6].tolist())
            for i, name in enumerate(X2_CFG.joint_names):
                q = float(state.qpos[7 + i])
                lo, hi = float(codec.lower[i]), float(codec.upper[i])
                if q < lo - .01 or q > hi + .01:
                    report['joint_limit_violations'][name] = {
                        'position_rad': q, 'lower_rad': lo, 'upper_rad': hi}
        passed = (state is not None
                  and (not args.require_exclusive_control or not any(report['command_publishers'].values()))
                  and (not args.require_joint_limits or not report['joint_limit_violations']))
        report['requested_checks_passed'] = passed
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n')
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
